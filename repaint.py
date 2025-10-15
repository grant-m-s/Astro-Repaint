# Copyright (c) 2022 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This repository was forked from https://github.com/openai/guided-diffusion, which is under the MIT license

"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import os
import sys 
import argparse
import torch as th
import torch.nn.functional as F
import time
import numpy as np
import conf_mgt
from utils import yamlread
from guided_diffusion import dist_util_repaint
from guided_diffusion.image_datasets import load_data
from PIL import Image
import blobfile as bf
import matplotlib.pyplot as plt
# from sklearn.metrics import mean_squared_error
from copy import deepcopy
import json
from time import time
from tqdm import tqdm

import asyncio

from types import SimpleNamespace
print("imported most...")
# Workaround
try:
    import ctypes
    libgcc_s = ctypes.CDLL('libgcc_s.so.1')
except:
    pass

conf_arg = None

from guided_diffusion.script_util_repaint import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    select_args,
)  # noqa: E402

print("Imports Complete...")

if not os.path.isdir("repaint_data"):
    os.mkdir("repaint_data")
subdirs = ["masks","images"]
for s in subdirs:
    if not os.path.isdir(f"repaint_data/{s}"):
        os.mkdir(f"repaint_data/{s}")

def toU8(sample):
    if sample is None:
        return sample

    sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    sample = sample.detach().cpu().numpy()
    return sample

def imread(path):
    with bf.BlobFile(path, "rb") as f:
        pil_image = Image.open(f)
        pil_image.load()
    return pil_image
    

def main(conf: conf_mgt.Default_Conf_Repaint, progress = None, batch_size = None, file_index=None, data_dir=None):
    print("Start", conf['name'])

    device = "cuda"

    model, diffusion = create_model_and_diffusion(
        **select_args(conf, model_and_diffusion_defaults().keys()), conf=conf
    )
    model.load_state_dict(
        dist_util_repaint.load_state_dict(os.path.expanduser(
            conf.model_path), map_location="cpu")
    )
    # model= th.nn.DataParallel(model)

    print("test device:'CUDA'")
    test_tensor1 = th.ones((5,5),device="cuda")
    print(test_tensor1)
    print("test tensor .to(device)")
    test_tensor2 = th.ones((5,5)).to(device)
    print(test_tensor2)
    print("to device")
    model.to(device)
    print("model on device")
    
    if conf.use_fp16:
        model.convert_to_fp16()

    model.eval()

    print(model)

    show_progress = conf.show_progress

    if conf.classifier_scale > 0 and conf.classifier_path:
        print("loading classifier...")
        classifier = create_classifier(
            **select_args(conf, classifier_defaults().keys()))
        classifier.load_state_dict(
            dist_util_repaint.load_state_dict(os.path.expanduser(
                conf.classifier_path), map_location="cpu")
        )

        classifier.to(device)
        if conf.classifier_use_fp16:
            classifier.convert_to_fp16()
        classifier.eval()

        def cond_fn(x, t, y=None, gt=None, **kwargs):
            assert y is not None
            with th.enable_grad():
                x_in = x.detach().requires_grad_(True)
                logits = classifier(x_in, t)
                log_probs = F.log_softmax(logits, dim=-1)
                selected = log_probs[range(len(logits)), y.view(-1)]
                return th.autograd.grad(selected.sum(), x_in)[0] * conf.classifier_scale
    else:
        cond_fn = None

    def model_fn(x, t, y=None, gt=None, **kwargs):
        assert y is not None
        return model(x, t, y if conf.class_cond else None, gt=gt)

    print("making masks...")
    os.makedirs("repaint_data/masks/", exist_ok=True)
    r_start = 27
    r_end = 36

    c_start = 28
    c_end = 37

    for s in [1,3,5,7,9]:
        for r_idx in range(9):

            for c_idx in range(9):

                mask = np.ones((64,64,3), dtype=np.uint8)*255

                c_offset = int(np.ceil(s/2)-1)

                mask[r_start+r_idx-(c_offset):r_start+r_idx-(c_offset)+s,c_start-(c_offset)+c_idx:c_start+c_idx-(c_offset)+s,:] = 0
                im = Image.fromarray(mask)
                # im = im.convert('RGB')
                im.save(f"repaint_data/masks/{r_start+r_idx}-{c_start+c_idx}-{s}.png")
    
    mask = np.ones((64,64,3), dtype=np.uint8)*255
    mask[:,::2,:] = 0
    im = Image.fromarray(mask)
    im.save(f"repaint_data/masks/alt_cols.png")

    mask = np.ones((64,64,3), dtype=np.uint8)*255
    mask[::2,:,:] = 0
    im = Image.fromarray(mask)
    im.save(f"repaint_data/masks/alt_rows.png")
    
    print("sampling...")
    all_images = []

    eval_name = conf.get_default_eval_name()
    print(eval_name)

    # dl = conf.get_dataloader(dset=dset, dsName=eval_name)

    if data_dir == None:
        data_dir = "data"
    else:
        if data_dir[-1] == '/':
            data_dir = data_dir[:-1]

    dl = load_data(
        data_dir=data_dir,
        ids_file="batch_source_full.json",
        batch_size=conf.data['eval']['paper_face_mask']['npy_files_at_once'],
        image_size=64,
        deterministic=True,
        set=conf.data['eval']['paper_face_mask']['dset'],
        file_index=file_index)
    
    print("data_loader: ", dl)

    with open(f"{data_dir}/batch_source_full.json",'r') as fp:
        batch_ids = json.load(fp)

    print("batch_id keys: ",list(batch_ids.keys()))

    if batch_size is None:
        use_b = 1024
    else:
        use_b = batch_size

    o_pixels = []
    c_pixels = []
    c_pixels_norm = []
    mask_centres = []
    
    all_ids = []

    if os.path.isfile(f"repaint_data/res_diffusion_{conf_arg.data['eval']['paper_face_mask']['inpaint_size']}_{conf_arg.schedule_jump_params['t_T']}_{conf_arg.schedule_jump_params['jump_length']}_{conf_arg.schedule_jump_params['jump_n_sample']}_{conf.data['eval']['paper_face_mask']['dset']}.json"):
        with open(f"repaint_data/res_diffusion_{conf_arg.data['eval']['paper_face_mask']['inpaint_size']}_{conf_arg.schedule_jump_params['t_T']}_{conf_arg.schedule_jump_params['jump_length']}_{conf_arg.schedule_jump_params['jump_n_sample']}_{conf.data['eval']['paper_face_mask']['dset']}.json",'r') as fp:
            res = json.load(fp)

        all_ids = res["ids"]
        o_pixels = res["o_pixels"]
        c_pixels = res["c_pixels"]
        c_pixels_norm = res["c_pixels_norm"]
        mask_centres = res["centre_pixels"]


    if progress is not None:
        progress.value = 0
        progress.active = True
    
    print("use_b: ", use_b)

    for b_idx, batch_ in enumerate(iter(dl)):

        mask_paths = []
        centre_pixels = []

        batch, ids = batch_

        ids_ = [x[0] for x in ids]

        
        remaining_indexs = [index_x for index_x, x in enumerate(ids_) if x not in all_ids][:use_b]
        if progress is not None:
            rm = [index_x for index_x, x in enumerate(ids_) if x not in all_ids]

            progress.value = int(((len(ids_)-len(rm))/(len(ids_)))*100)

        batch = batch[:,remaining_indexs,:,:,:]
        ids = [x for index_x, x in enumerate(ids) if index_x in remaining_indexs]

        print("ids: ", ids)

        if not len(remaining_indexs):
            break

        if len(ids) == 0:
            break

        batch = batch.view(batch.shape[0]*batch.shape[1],batch.shape[-3],batch.shape[-2], batch.shape[-1])
        max_values = batch.view(batch.shape[0], -1).max(dim=1, keepdim=True)[0]
        min_values = batch.view(batch.shape[0], -1).min(dim=1, keepdim=True)[0]

        # Reshape max_values to make it compatible for division
        max_values = max_values.view(batch.shape[0], 1, 1, 1)
        # print("max_values: ", max_values.shape)
        min_values = min_values.view(batch.shape[0], 1, 1, 1)
        # print("min_values: ", min_values.shape)

        max_values += abs(min_values)+1e-8

        ids_temp = []
        for i in range(len(ids[0])):
            ids_temp += [x[i] for x in ids]

        ids = ids_temp 

        a = 0

        for i in batch:

            cutout = i[0]

            cutout = cutout[r_start:r_end,c_start:c_end]

            max_val = np.unravel_index(np.argmax(cutout), cutout.shape)

            if conf.data['eval']['paper_face_mask']['diffusion_test']:
                mask_paths.append(f"repaint_data/masks/alt_cols.png")
            else:
                mask_paths.append(f"repaint_data/masks/{r_start+max_val[0]}-{c_start+max_val[1]}-{conf.data['eval']['paper_face_mask']['inpaint_size']}.png")
            
            centre_pixels.append([r_start+max_val[0],c_start+max_val[1]])

        print(mask_paths[:5])

        model_kwargs = {}

        model_kwargs["gt"] = batch.to(device)

        masks = np.zeros((1,1,64,64))
        for m_idx,m in enumerate(mask_paths):
            a = np.array(imread(m))
            a = a.astype(np.float16) / 255.0
            if m_idx == 0:
                masks = a[None,None,:,:,0]
            else:
                masks = np.vstack((masks,a[None,None,:,:,0]))

        gt_keep_mask = th.tensor(masks,dtype=th.int8).to(device)

        if gt_keep_mask is not None:
            model_kwargs['gt_keep_mask'] = gt_keep_mask

        batch_size = model_kwargs["gt"].shape[0]

        if conf.cond_y is not None:
            classes = th.ones(batch_size, dtype=th.long, device=device)
            model_kwargs["y"] = classes * conf.cond_y
        else:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(batch_size,), device=device
            )
            model_kwargs["y"] = ids

        sample_fn = (
            diffusion.p_sample_loop if not conf.use_ddim else diffusion.ddim_sample_loop
        )

        start = time()

        result = sample_fn(
            model_fn,
            (batch_size, 1, conf.image_size, conf.image_size),
            clip_denoised=conf.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            device=device,
            progress=show_progress,
            return_all=True,
            conf=conf
        )
        end = time()

        srs = result['sample'].cpu().numpy()
        gts = model_kwargs["gt"].cpu().numpy()

        masks_ = model_kwargs.get('gt_keep_mask').cpu().numpy()

        lrs = (gts * masks_) + (srs * (1 - masks_))

        s = conf.data['eval']['paper_face_mask']['inpaint_size']
        to_show = [x for x in range(0,1024,32)]

        for i in tqdm(range(len(batch))):

            c_offset = int(np.ceil(s/2)-1)

            if not conf.data['eval']['paper_face_mask']['diffusion_test']:
                c_srs = srs[i,0, centre_pixels[i][0]-(c_offset):centre_pixels[i][0]-(c_offset)+s,centre_pixels[i][1]-(c_offset):centre_pixels[i][1]-(c_offset)+s]
                c_gts = gts[i,0, centre_pixels[i][0]-(c_offset):centre_pixels[i][0]-(c_offset)+s,centre_pixels[i][1]-(c_offset):centre_pixels[i][1]-(c_offset)+s]
            else:
                c_srs = srs[i,0, :, ::2]
                c_gts = gts[i,0, :, ::2]

            mask_full = np.zeros_like(srs[i,0])
            mask_full[centre_pixels[i][0]-(c_offset):centre_pixels[i][0]-(c_offset)+s,centre_pixels[i][1]-(c_offset):centre_pixels[i][1]-(c_offset)+s] = 1

            if not conf.data['eval']['paper_face_mask']['diffusion_test']:
                output_save_dir = f"repaint_data/outputs_{conf.data['eval']['paper_face_mask']['dset']}/{conf.schedule_jump_params['t_T']}_{conf.schedule_jump_params['jump_length']}_{conf.schedule_jump_params['jump_n_sample']}"
            else:
                # output_save_dir = f"repaint_data/outputs/Tests"
                output_save_dir = f"repaint_data/outputs/{conf.data['eval']['paper_face_mask']['dset']}"


            os.makedirs(output_save_dir, exist_ok=True)

            orig_without_centre = gts[i][0].copy()
            output_without_centre = srs[i][0].copy()

            orig_without_centre[centre_pixels[i][0]-(c_offset):centre_pixels[i][0]-(c_offset)+s,centre_pixels[i][1]-(c_offset):centre_pixels[i][1]-(c_offset)+s] = np.nan
            output_without_centre[centre_pixels[i][0]-(c_offset):centre_pixels[i][0]-(c_offset)+s,centre_pixels[i][1]-(c_offset):centre_pixels[i][1]-(c_offset)+s] = np.nan

            scale = (np.nanmax(orig_without_centre)-np.nanmin(orig_without_centre))/(np.nanmax(output_without_centre)-np.nanmin(output_without_centre))
            offset = np.nanmin(orig_without_centre) - (scale*np.nanmin(output_without_centre))


            new_output = (srs[i][0]*scale) + offset
            c_output = new_output[centre_pixels[i][0]-(c_offset):centre_pixels[i][0]-(c_offset)+s,centre_pixels[i][1]-(c_offset):centre_pixels[i][1]-(c_offset)+s]
            if i in to_show:
                mosaic = [["A","CN","DN","E","GN"]]
                
                plt.close()

                fig, axes = plt.subplot_mosaic(mosaic,figsize=(25,5), gridspec_kw={'wspace':0.02, 'hspace':0.02})
                axes["A"].imshow(gts[i][0],cmap="Greys_r")
                axes["A"].text(1 ,gts[i][0].shape[1]-1, str(ids[i]), color='white', ha='left', va='bottom',fontsize=20)
                axes["A"].axis('off')
  
                a_max = np.max(np.array([gts[i][0].max(), new_output.max()]))
                a_min = np.min(np.array([gts[i][0].min(), new_output.min()]))

                axes["CN"].imshow(new_output,cmap="Greys_r",clim=(a_min,a_max))
                axes["CN"].axis('off')

                axes["DN"].imshow(gts[i][0]-srs[i][0],cmap="seismic",clim=(-1.0*a_max,a_max))
                axes["DN"].axis('off')

                a_max = np.max(np.array([c_gts.max(), c_output.max()]))
                a_min = np.min(np.array([c_gts.min(), c_output.min()]))

                axes["E"].imshow(c_gts,cmap="Greys_r",clim=(a_min,a_max))
                axes["E"].axis('off')

                axes["GN"].imshow(c_output,cmap="Greys_r",clim=(a_min,a_max))
                axes["GN"].axis('off')

                if conf.data['eval']['paper_face_mask']['diffusion_test']:
                    fig.suptitle(f"Timesteps:{conf.schedule_jump_params['t_T']} | jump len:{conf.schedule_jump_params['jump_length']} | samples:{conf.schedule_jump_params['jump_n_sample']} | {(end-start)/60:.2f} mins")
                    plt.savefig(f"{output_save_dir}/{i}_{conf.schedule_jump_params['t_T']}_{conf.schedule_jump_params['jump_length']}_{conf.schedule_jump_params['jump_n_sample']}.png", bbox_inches="tight")

                else:
                    plt.savefig(f"{output_save_dir}/{ids[i]}_{s}.pdf", bbox_inches="tight")

            plt.close()

            all_ids.append(f"{ids[i]}")
            o_pixels.append([str(x) for x in c_gts.flatten().tolist()])
            c_pixels.append([str(x) for x in c_srs.flatten().tolist()])
            c_pixels_norm.append([str(x) for x in c_output.flatten().tolist()])
            mask_centres.append([centre_pixels[i][0]-(c_offset),centre_pixels[i][0]-(c_offset)+s,centre_pixels[i][1]-(c_offset),centre_pixels[i][1]-(c_offset)+s])

        res = {}
        res["ids"] = all_ids
        res["o_pixels"] = o_pixels
        res["c_pixels"] = c_pixels
        res["c_pixels_norm"] = c_pixels_norm

        res["centre_pixels"] = np.array(mask_centres).tolist()

        with open(f"repaint_data/res_diffusion_{s}_{conf.schedule_jump_params['t_T']}_{conf.schedule_jump_params['jump_length']}_{conf.schedule_jump_params['jump_n_sample']}_{conf.data['eval']['paper_face_mask']['dset']}.json",'w') as fp:
            json.dump(res,fp,indent=2)

        print("saved")

        yield res

    print("sampling complete")
    yield res
    return res

def setup(t_T = None, js = None, jl = None, inpaint_size = None):

    print("Initialising Argparse...")

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_path', type=str, required=False, default='confs/galaxy.yml', help="Path to configuration file.")
    parser.add_argument('--inpaint_size','-is', type=int, required=False, default=5, help="The size of the square mask used for inpainting.")
    parser.add_argument('--t_T','-t', type=int, required=False, default=60, help="The number of inference timesteps. Larger numbers will produce more resiliant outputs but will take longer per batch.")
    parser.add_argument('--jump_n_sample','-js', type=int, required=False, default=3, help="The number of resamples made in the repaint process (see repaint paper for details). Larger numbers will produce more resiliant outputs but will take longer per batch.")
    parser.add_argument('--jump_length','-jl', type=int, required=False, default=1, help="The number of steps jumped before resampling. Smaller numbers will produce more resiliant outputs but will take longer per batch.")
    parser.add_argument('--diffusion_test','-dt', type=bool, required=False, default=False, help="A flag for testing the diffusion model on more extreme masks. This is only recommended for testing.")
    parser.add_argument('--batch_size','-bs', type=int, required=False, default=1024, help="The number of images to inpaint at once in an iteration. Memory is main bottleneck for inference and so only increase if you have sufficient GPU Memory.")

    args, _ = parser.parse_known_args()


    args = vars(args)

    if t_T is not None:
        args['t_T'] = t_T

    if js is not None:
        args['jump_n_sample'] = js

    if jl is not None:
        args['jump_length'] = jl

    if inpaint_size is not None:
        args['inpaint_size'] = inpaint_size


    global conf_arg

    conf_arg = conf_mgt.conf_base.Default_Conf_Repaint()
    conf_arg.update(yamlread(args.get('conf_path')))
    print(args)
    print()


    conf_arg.data['eval']['paper_face_mask']['diffusion_test'] = args['diffusion_test']
    if conf_arg.data['eval']['paper_face_mask']['diffusion_test']:
        print("\n\n==========\nRUNNING DIFFUSION TEST\n==========\n")

    if args['t_T'] is not None:
        o_ = conf_arg['schedule_jump_params']['t_T'] 
        # print(args['t_T'])
        conf_arg['schedule_jump_params']['t_T'] = args['t_T']
        print(f"Updated t_T: {o_} -> {conf_arg['schedule_jump_params']['t_T']}")

    if args['jump_n_sample'] is not None:
        o_ = conf_arg['schedule_jump_params']['jump_n_sample'] 

        conf_arg['schedule_jump_params']['jump_n_sample'] = args['jump_n_sample']
        print(f"Updated jump_n_sample: {o_} -> {conf_arg['schedule_jump_params']['jump_n_sample']}")
    
    if args['inpaint_size'] is not None:
        o_ = conf_arg.data['eval']['paper_face_mask']['inpaint_size'] 

        conf_arg.data['eval']['paper_face_mask']['inpaint_size'] = args['inpaint_size']
        print(f"Updated inpaint_size: {o_} -> {conf_arg.data['eval']['paper_face_mask']['inpaint_size']}")
    
    if args['jump_length'] is not None:
        o_ = conf_arg['schedule_jump_params']['jump_length'] 

        conf_arg['schedule_jump_params']['jump_length'] = args['jump_length']
        print(f"Updated jump_length: {o_} -> {conf_arg['schedule_jump_params']['jump_length']}")
    
    if args['batch_size'] is not None:
        conf_arg['batch_size'] = args['batch_size']
        print(f"Updated batch_size: 1024 -> {conf_arg['batch_size']}")

    print()
    
    print("GPU Tests:")
    print(th.cuda.device_count())

    print("Done...")


    return conf_arg

if __name__ == "__main__":

    print("==================")
    print("RUNNING FROM CLI")
    print("==================")

    conf_arg = setup()
    print("Conf Arg Loaded")
    for r in main(conf_arg, batch_size=conf_arg['batch_size']):
        pass
    print("After Main")