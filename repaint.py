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
print("imported most...")
# Workaround
try:
    import ctypes
    libgcc_s = ctypes.CDLL('libgcc_s.so.1')
except:
    pass


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
    

def main(conf: conf_mgt.Default_Conf_Repaint):

    print("Start", conf['name'])

    # assert False
    # device = dist_util_repaint.dev(conf.get('device'))
    device = "cuda"
    #for i in range(th.cuda.device_count()):
    #    print(th.cuda.get_device_properties(i).name)
    #assert False
    model, diffusion = create_model_and_diffusion(
        **select_args(conf, model_and_diffusion_defaults().keys()), conf=conf
    )
    model.load_state_dict(
        dist_util_repaint.load_state_dict(os.path.expanduser(
            conf.model_path), map_location="cpu")
    )
    # model= th.nn.DataParallel(model)

    
    # free, total = th.cuda.mem_get_info(0)
    # memory_before = (total - free) / 1024 ** 3
    # print("before self.model_mean_type == ModelVarType.LEARNED:", memory_before)
    print("test device:'CUDA'")
    test_tensor1 = th.ones((5,5),device="cuda")
    print(test_tensor1)
    print("test tensor .to(device)")
    test_tensor2 = th.ones((5,5)).to(device)
    print(test_tensor2)
    print("to device")
    model.to(device)
    print("model on device")
    
    # free, total = th.cuda.mem_get_info(0)
    # memory_after = (total - free) / 1024 ** 3
    # print("model size:", memory_after-memory_before)
    if conf.use_fp16:
        model.convert_to_fp16()
        # free, total = th.cuda.mem_get_info(0)
        # memory_16 = (total - free) / 1024 ** 3
        # print("model size after fp16:", memory_16-memory_before)

    # assert False

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
    r_start = 27
    r_end = 36

    c_start = 28
    c_end = 37
    # cutout = mask[r_start:r_end,c_start:c_end]
    for s in [1,3,5,7,9]:
        for r_idx in range(9):
            # if r_start + r_idx + 3 > r_end:
            #     continue
            for c_idx in range(9):
                # if c_start + c_idx + 3 > c_end:
                #     continue
                mask = np.ones((64,64,3), dtype=np.uint8)*255
                # mask[r_start:r_end,c_start:c_end,:] = 128
                c_offset = int(np.ceil(s/2)-1)
                # print(c_offset)
                # assert False
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
    # print()
    # assert False
    dl = load_data(
        data_dir="/shared/cutouts_benchmarking/batches",
        ids_file="batch_source_full.json",
        batch_size=conf.data['eval']['paper_face_mask']['batch_size'],
        image_size=64,
        deterministic=True,
        set=conf.data['eval']['paper_face_mask']['dset'])
    print(dl)

    with open("/shared/cutouts_benchmarking/batches/batch_source_full.json",'r') as fp:
        batch_ids = json.load(fp)

    # print(list(batch_ids.keys()))

    # batch_ids = batch_ids["1"]
    # print(batch_ids[:5])
    # assert False
    
    # use_b = 5
    use_b = 1024


    o_pixels = []
    c_pixels = []
    c_pixels_norm = []
    

    # c_summed = []
    # c_mse = []

    # c_summed_norm = []
    # c_mse_norm = []

    # c_range = []
    # c_range_norm = []

    # r1 = []
    # r1_norm = []
    # r2 = []
    # r2_norm = []


    # all_summed = []
    # all_mse = []

    # all_summed_norm = []
    # all_mse_norm = []


    all_ids = []

    if os.path.isfile(f"repaint_data/res_diffusion_{conf_arg.data['eval']['paper_face_mask']['inpaint_size']}_{conf_arg.schedule_jump_params['t_T']}_{conf_arg.schedule_jump_params['jump_length']}_{conf_arg.schedule_jump_params['jump_n_sample']}_{conf.data['eval']['paper_face_mask']['dset']}.json"):
        with open(f"repaint_data/res_diffusion_{conf_arg.data['eval']['paper_face_mask']['inpaint_size']}_{conf_arg.schedule_jump_params['t_T']}_{conf_arg.schedule_jump_params['jump_length']}_{conf_arg.schedule_jump_params['jump_n_sample']}_{conf.data['eval']['paper_face_mask']['dset']}.json",'r') as fp:
            res = json.load(fp)

        all_ids = res["ids"]
        o_pixels = res["o_pixels"]
        c_pixels = res["c_pixels"]
        c_pixels_norm = res["c_pixels_norm"]

    for b_idx, batch_ in enumerate(iter(dl)):

        # if b_idx >= 50:
        #     break

        mask_paths = []
        centre_pixels = []

        batch, ids = batch_

        batch, ids = batch[:use_b], ids[:use_b]
        
        print(ids[0:3])
        print(ids[1024:1027])
        if ids[0][0] in all_ids:
            continue
        print(f"batch.shape for index {b_idx}: ", batch.shape)
        print(len(ids))
        if len(ids) == 0:
            break
        # curr_ids = []
        # curr_ids += 
        # all_ids += [x[0] for x in ids]

        # print(all_ids[:10])
        # assert False
 
        # print(batch.shape)
        # batch = (batch - th.min(batch))/(th.max(batch)-th.min(batch)+1e-8)

        batch = batch.view(batch.shape[0]*batch.shape[1],batch.shape[-3],batch.shape[-2], batch.shape[-1])
        max_values = batch.view(batch.shape[0], -1).max(dim=1, keepdim=True)[0]
        min_values = batch.view(batch.shape[0], -1).min(dim=1, keepdim=True)[0]
        # print("max_values: ", max_values.shape)

        # max_values shape will be (1024, 1) after max

        # Reshape max_values to make it compatible for division
        max_values = max_values.view(batch.shape[0], 1, 1, 1)
        # print("max_values: ", max_values.shape)
        min_values = min_values.view(batch.shape[0], 1, 1, 1)
        # print("min_values: ", min_values.shape)

        max_values += abs(min_values)+1e-8
        # print("max_values: ", max_values.shape)

        # Divide each image by its respective maximum value
        # batch = (batch + abs(min_values)+1e-8) / max_values
        #print("01:", ids[0][1])
        #print("10:", ids[1][0])
        #plt.imshow(batch[1][0],cmap="Greys_r")
        #plt.savefig("second.png")
        #plt.close()
        #assert False
        print("ids.shape:",len(ids),len(ids[0]))
        ids_temp = []
        for i in range(len(ids[0])):
            ids_temp += [x[i] for x in ids]
        #[00,10,20,..,01,11,21,...,02
        ids = ids_temp 

        #batch = batch[:use_b]
        print("batch.shape: ", batch.shape)
        print("ids: ",len(ids))
        # print(batch.shape)

        # print(srs.shape)
        # print(gts.shape)
        # print(batch.shape)
        a = 0
        
        for i in batch:
            # print(i.shape)
            # mosaic = [["A","B","C"]]
            # fig, axes = plt.subplot_mosaic(mosaic)
            cutout = i[0]
            # print(cutout.shape)
            # axes["A"].imshow(cutout,cmap="Greys_r")


            # c_out = deepcopy(cutout)
            # subset = deepcopy(c_out[r_start:r_end,c_start:c_end])
            # c_out[r_start:r_end,c_start:c_end] = th.zeros_like(c_out[r_start:r_end,c_start:c_end])
            # cutout [r_start:r_end,c_start:c_end] = th.zeros_like(c_out[r_start:r_end,c_start:c_end])
            cutout = cutout[r_start:r_end,c_start:c_end]
            # print(cutout.shape)
            # assert False
            # axes["B"].imshow(c_out,cmap="Greys_r")

            max_val = np.unravel_index(np.argmax(cutout), cutout.shape)

            # plt.imshow()


            # cutout[r_start+max_val[0],c_start+max_val[1]] = 1000

            # print(max_val)
            # print(r_start+max_val[0],c_start+max_val[1])
            # assert False


            # max_val = np.unravel_index(np.argmax(cutout), cutout.shape)
            # axes["C"].imshow(cutout,cmap="Greys_r")
            # axes["A"].imshow(cutout,cmap="Greys_r")

            # plt.show()

            # print(max_val)
            # print(f"{r_start+max_val[0]}-{c_start+max_val[1]}")
            # print(f"{r_start+max_val[0]-1}:{r_start+max_val[0]-1+3}-{c_start+max_val[1]-1}:{c_start+max_val[1]-1+3}")
            if conf.data['eval']['paper_face_mask']['diffusion_test']:
                mask_paths.append(f"repaint_data/masks/alt_cols.png")
            else:
                mask_paths.append(f"repaint_data/masks/{r_start+max_val[0]}-{c_start+max_val[1]}-{conf.data['eval']['paper_face_mask']['inpaint_size']}.png")
            # m_ = imread(f"repaint_data/masks/{r_start+max_val[0]}-{c_start+max_val[1]}-3.png")
            # m_ = np.asarray(m_)[:,:,0]

            # if a < 10:
            #     mosaic = [["A","B","C","D"]]

            #     fig, axes = plt.subplot_mosaic(mosaic)
            #     full = i[0].cpu().numpy()
            #     full[:r_start] = 0
            #     full[:,:c_start] = 0
            #     full[:,c_end:] = 0
            #     full[r_end:] = 0

                
            #     test = m_ * i[0].cpu().numpy()
            #     axes["A"].imshow(test)
            #     axes["B"].imshow(full)

            #     axes["C"].imshow(i[0].cpu().numpy()[r_start+max_val[0]-2:r_start+max_val[0]+3,r_start+max_val[1]-2:r_start+max_val[1]+3])
            #     axes["D"].imshow(test[r_start+max_val[0]-2:r_start+max_val[0]+3,r_start+max_val[1]-2:r_start+max_val[1]+3])
            #     # plt.imshow(test)
            #     plt.show()
            #     a+=1
            # else:
            #     assert False
            
            centre_pixels.append([r_start+max_val[0],c_start+max_val[1]])

            # assert False
        print(mask_paths[:5])
        # assert False
        # print(len(mask_paths))
        # assert False
        # print(ids)
        # assert False

        # for k in batch.keys():
        #     if isinstance(batch[k], th.Tensor):
        #         batch[k] = batch[k].to(device)

        model_kwargs = {}
        # print(batch.dtype)
        # assert False
        # before
        # th._C._cuda_clearCublasWorkspaces()
        # memory_before = th.cuda.memory_allocated(device)

        # your tensor
        # data = torch.randn((10000,100),device=device)

        model_kwargs["gt"] = batch.to(device)
        # after
        # memory_after = th.cuda.memory_allocated(device)
        # latent_size = memory_after - memory_before
        # print("batch: ",latent_size/(1024**3))
        masks = np.zeros((1,1,64,64))
        for m_idx,m in enumerate(mask_paths):
            a = np.array(imread(m))
            a = a.astype(np.float16) / 255.0
            if m_idx == 0:
                masks = a[None,None,:,:,0]
            else:
                masks = np.vstack((masks,a[None,None,:,:,0]))
                # print(masks.shape)
                # assert False
        # memory_before = th.cuda.memory_allocated(device)

        gt_keep_mask = th.tensor(masks,dtype=th.int8).to(device)

        # memory_after = th.cuda.memory_allocated(device)
        # latent_size = memory_after - memory_before
        # print("masks: ",latent_size/(1024**3))

        # memory_before = th.cuda.memory_allocated(device)

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
        # after
        # memory_after = th.cuda.memory_allocated(device)
        # latent_size = memory_after - memory_before
        # print("cond: ",latent_size/(1024**3))
        # free, total = th.cuda.mem_get_info(0)
        # memory_before = (total - free) / 1024 ** 3
        # memory_before = th.cuda.memory_allocated(device)
        # print(conf.use_ddim)
        # print(th.cuda.memory_allocated("cuda")/(1024**3))
        sample_fn = (
            diffusion.p_sample_loop if not conf.use_ddim else diffusion.ddim_sample_loop
        )
        # free, total = th.cuda.mem_get_info(0)
        # memory_after = (total - free) / 1024 ** 3
        # latent_size = memory_after - memory_before
        # print("sample_fn: ",latent_size/(1024**3), memory_after)
        # assert False
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
#['-556629272489603255', '-632776876468028387', '2708791418636299385', '-541497046267981377', '2673165896651963039', '-536608986262815699', '-634373019486682663', '-634470759471104100', '2697493600656504684', '2726737091665822700', '2639810088647520060', '-546245776291251279', '2712000623685703191', '-614431404497676875', '-516229662281599789', '-534348504295891656', '-529176656264415880', '-581580627491826837', '-541110174282446080', '-518532302288237103']
#['-556629272489603255', '-632776876468028387', '2708791418636299385', '-541497046267981377', '2673165896651963039', '-536608986262815699', '-634373019486682663', '-634470759471104100', '2697493600656504684', '2726737091665822700', '2639810088647520060', '-546245776291251279', '2712000623685703191', '-614431404497676875', '-516229662281599789', '-534348504295891656', '-529176656264415880', '-581580627491826837', '-541110174282446080', '-518532302288237103']        
        print(ids[:10])
        # assert False
        # srs = toU8(result['sample'])
        # gts = toU8(result['gt'])
        # lrs = toU8(result.get('gt') * model_kwargs.get('gt_keep_mask') + (-1) *
        #            th.ones_like(result.get('gt')) * (1 - model_kwargs.get('gt_keep_mask')))

        # gt_keep_masks = toU8((model_kwargs.get('gt_keep_mask') * 2 - 1))

        srs = result['sample'].cpu().numpy()
        gts = model_kwargs["gt"].cpu().numpy()

        # r_start = 27
        # r_end = 36

        # c_start = 28
        # c_end = 37
        # for i in range(10):
        #     print("---")
        #     print(np.unravel_index(np.argmax(gts[i][0]), gts[i][0].shape))
        #     print(centre_pixels[i])
        #     centre_max = np.unravel_index(np.argmax(gts[i][0,r_start:r_end,c_start:c_end]), gts[i][0,r_start:r_end,c_start:c_end].shape)
        #     print([r_start+centre_max[0],c_start+centre_max[1]])

        #     plt.imshow(gts[i][0,centre_pixels[i][0]-1:centre_pixels[i][0]+2,centre_pixels[i][1]-1:centre_pixels[i][1]+2])
        #     plt.show()


        # assert False

        # print(result['pred_xstart'])
        # import matplotlib.pyplot as plt
        # mosaic = [["A","B","C","D"]]
        # for img in range(10):
        #     plt.close()
        #     fig, axes = plt.subplot_mosaic(mosaic)
        #     axes["A"].imshow(result['sample'].cpu().numpy()[img][0],cmap="Greys_r")
        #     axes["A"].set_title("output", fontsize=10)
        #     axes["B"].imshow(model_kwargs["gt"].cpu().numpy()[img][0],cmap="Greys_r")
        #     axes["B"].set_title("orig", fontsize=10)
        #     axes["C"].imshow(model_kwargs["gt_keep_mask"].cpu().numpy()[img][0],cmap="Greys_r")
        #     axes["C"].set_title(result['ids'][img], fontsize=10)
        #     axes["D"].imshow(model_kwargs["gt"].cpu().numpy()[img][0]-result['sample'].cpu().numpy()[img][0],cmap="Greys_r")
        #     axes["D"].set_title("diff", fontsize=10)


        #     plt.show()
        
        # assert False
        # print(srs.shape)
        # print(gts.shape)

        # lrs = result.get('gt').cpu() * model_kwargs.get('gt_keep_mask').cpu() + (
        #     (-1) * th.ones_like(result.get('gt').cpu()) * (1 - model_kwargs.get('gt_keep_mask').cpu()))

        masks_ = model_kwargs.get('gt_keep_mask').cpu().numpy()
        # for i in range(10):
        #     plt.imshow(masks_[i][0])
        #     plt.show()

        lrs = (gts * masks_) + (srs * (1 - masks_))

        # mosaic = [["A","B","C"]]
        # for i in range(20):
        #     plt.close()
        #     fig,axes = plt.subplot_mosaic(mosaic)
        #     axes["A"].imshow(srs[i][0].cpu(),cmap="Greys_r")
        #     axes["B"].imshow(gts[i][0].cpu(),cmap="Greys_r")
        #     axes["C"].imshow(lrs[i][0].cpu(),cmap="Greys_r")
        #     plt.savefig(f"repaint_data/outputs/{i}.png")
            

            # print(srs.shape)
            # print(gts.shape)
            # print(lrs.shape)

        s = conf.data['eval']['paper_face_mask']['inpaint_size']
        to_show = [x for x in range(0,1024,32)]

        for i in tqdm(range(len(batch))):

            c_offset = int(np.ceil(s/2)-1)
            # print(c_offset)
            # assert False
            # print(srs.shape)
            if not conf.data['eval']['paper_face_mask']['diffusion_test']:
                c_srs = srs[i,0, centre_pixels[i][0]-(c_offset):centre_pixels[i][0]-(c_offset)+s,centre_pixels[i][1]-(c_offset):centre_pixels[i][1]-(c_offset)+s]
                c_gts = gts[i,0, centre_pixels[i][0]-(c_offset):centre_pixels[i][0]-(c_offset)+s,centre_pixels[i][1]-(c_offset):centre_pixels[i][1]-(c_offset)+s]
            else:
                c_srs = srs[i,0, :, ::2]
                c_gts = gts[i,0, :, ::2]
            # print(c_gts.shape)
            mask_full = np.zeros_like(srs[i,0])
            mask_full[centre_pixels[i][0]-(c_offset):centre_pixels[i][0]-(c_offset)+s,centre_pixels[i][1]-(c_offset):centre_pixels[i][1]-(c_offset)+s] = 1
            # full_orig = gts[i,0]
            # full_generated = srs[i,0]]
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
            # print(scale)
            offset = np.nanmin(orig_without_centre) - (scale*np.nanmin(output_without_centre))
            # print(offset)

            new_output = (srs[i][0]*scale) + offset
            c_output = new_output[centre_pixels[i][0]-(c_offset):centre_pixels[i][0]-(c_offset)+s,centre_pixels[i][1]-(c_offset):centre_pixels[i][1]-(c_offset)+s]
            if i in to_show:
                # mosaic = [["A","B","C","CN","D","DN","E","F","GN"]]
                mosaic = [["A","CN","DN","E","GN"]]
                
                # for i in range(20):
                plt.close()

                fig, axes = plt.subplot_mosaic(mosaic,figsize=(25,5), gridspec_kw={'wspace':0.02, 'hspace':0.02})
                axes["A"].imshow(gts[i][0],cmap="Greys_r")
                axes["A"].text(1 ,gts[i][0].shape[1]-1, str(ids[i]), color='white', ha='left', va='bottom',fontsize=20)
                axes["A"].axis('off')
                # axes["B"].imshow(srs[i][0],cmap="Greys_r")
                # axes["B"].axis('off')
                # axes["C"].imshow(srs[i][0],cmap="Greys_r",clim=(gts[i][0].min(),gts[i][0].max()))
                # axes["C"].axis('off')
                # axes["C"].imshow(new_output,cmap="Greys_r")
                # axes["C"].axis('off')
                a_max = np.max(np.array([gts[i][0].max(), new_output.max()]))
                a_min = np.min(np.array([gts[i][0].min(), new_output.min()]))

                axes["CN"].imshow(new_output,cmap="Greys_r",clim=(a_min,a_max))
                axes["CN"].axis('off')
                # axes["D"].imshow(gts[i][0]-srs[i][0],cmap="Greys_r")
                # axes["D"].axis('off')
                # largest = np.max(np.abs(gts[i][0]-srs[i][0]))

                axes["DN"].imshow(gts[i][0]-srs[i][0],cmap="seismic",clim=(-1.0*a_max,a_max))
                axes["DN"].axis('off')

                a_max = np.max(np.array([c_gts.max(), c_output.max()]))
                a_min = np.min(np.array([c_gts.min(), c_output.min()]))

                axes["E"].imshow(c_gts,cmap="Greys_r",clim=(a_min,a_max))
                axes["E"].axis('off')
                # axes["F"].imshow(c_srs,cmap="Greys_r")
                # axes["F"].axis('off')
                # axes["G"].imshow(c_output,cmap="Greys_r")
                # axes["G"].axis('off')

                axes["GN"].imshow(c_output,cmap="Greys_r",clim=(a_min,a_max))
                axes["GN"].axis('off')

                # print(f"---{i}---")
                # print("c_gts:",np.min(c_gts), np.max(c_gts))
                # print("c_srs:",np.min(c_srs), np.max(c_srs))
                # print("c_output:",np.min(c_output), np.max(c_output))


                # print("gts:",np.min(gts[i][0]), np.max(gts[i][0]))
                # print("srs:",np.min(srs[i][0]), np.max(srs[i][0]))
                # print("new_output:",np.min(new_output), np.max(new_output))

                # print("------")

                # if (i == 4) or (i==13) or (i==14):
                #     print(len(gts[i]))
                #     plt.show()


                # axes["C"].imshow(lrs[i,0],cmap="Greys_r")
                if conf.data['eval']['paper_face_mask']['diffusion_test']:
                    fig.suptitle(f"Timesteps:{conf.schedule_jump_params['t_T']} | jump len:{conf.schedule_jump_params['jump_length']} | samples:{conf.schedule_jump_params['jump_n_sample']} | {(end-start)/60:.2f} mins")
                    plt.savefig(f"{output_save_dir}/{i}_{conf.schedule_jump_params['t_T']}_{conf.schedule_jump_params['jump_length']}_{conf.schedule_jump_params['jump_n_sample']}.png", bbox_inches="tight")

                else:
                    plt.savefig(f"{output_save_dir}/{ids[i]}_{s}.pdf", bbox_inches="tight")
            # else:
            #     if conf.data['eval']['paper_face_mask']['diffusion_test']:
            #         assert False

            plt.close()

            all_ids.append(f"{ids[i]}")
            o_pixels.append([str(x) for x in c_gts.flatten().tolist()])
            c_pixels.append([str(x) for x in c_srs.flatten().tolist()])
            c_pixels_norm.append([str(x) for x in c_output.flatten().tolist()])

            # c_summed.append(np.sum(c_gts)-np.sum(c_srs))
            # c_mse.append(mean_squared_error(c_gts.flatten(),c_srs.flatten()))
            # all_mse.append(mean_squared_error(gts[i].flatten(),srs[i].flatten()))
            # all_summed.append(np.sum(gts[i])-np.sum(srs[i]))

        res = {}
        res["ids"] = all_ids
        res["o_pixels"] = o_pixels
        res["c_pixels"] = c_pixels
        res["c_pixels_norm"] = c_pixels_norm

        # res["all_mse"] = [str(x) for x in all_mse]
        # res["all_summed"] = [str(x) for x in all_summed]
        # res["c_mse"] = [str(x) for x in c_mse]
        # res["c_summed"] = [str(x) for x in c_summed]
        #assert False
        with open(f"repaint_data/res_diffusion_{s}_{conf.schedule_jump_params['t_T']}_{conf.schedule_jump_params['jump_length']}_{conf.schedule_jump_params['jump_n_sample']}_{conf.data['eval']['paper_face_mask']['dset']}.json",'w') as fp:
            json.dump(res,fp,indent=2)

        # assert False
        # gt_keep_masks = model_kwargs.get('gt_keep_mask') * 2 - 1

        # conf.eval_imswrite(
        #     srs=srs, gts=gts, lrs=lrs, gt_keep_masks=gt_keep_masks,
        #     img_names=batch['GT_name'], dset=dset, name=eval_name, verify_same=False)

    print("sampling complete")


if __name__ == "__main__":
    print("Initialising Argparse...")

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_path', type=str, required=False, default=None)
    parser.add_argument('--inpaint_size','-is', type=int, required=False, default=None)
    parser.add_argument('--t_T','-t', type=int, required=False, default=None)
    parser.add_argument('--jump_n_sample','-js', type=int, required=False, default=None)
    parser.add_argument('--jump_length','-jl', type=int, required=False, default=None)
    parser.add_argument('--diffusion_test','-dt', type=bool, required=False, default=False)
    print("Entered Main")
    #   t_T: 30
    #   n_sample: 1
    #   jump_length: 15 ## more is faster
    #   jump_n_sample: 1 ## more is slower

    args = vars(parser.parse_args())

    conf_arg = conf_mgt.conf_base.Default_Conf_Repaint()
    conf_arg.update(yamlread(args.get('conf_path')))
    print(args)
    print()

    # if args['diffusion_test']:
    # o_ = conf_arg['schedule_jump_params']['t_T'] 
    # print(args['t_T'])
    conf_arg.data['eval']['paper_face_mask']['diffusion_test'] = args['diffusion_test']
    if conf_arg.data['eval']['paper_face_mask']['diffusion_test']:
        print("\n\n==========\nRUNNING DIFFUSION TEST\n==========\n")
    # print(f"")

    if args['t_T'] is not None:
        o_ = conf_arg['schedule_jump_params']['t_T'] 
        # print(args['t_T'])
        conf_arg['schedule_jump_params']['t_T'] = args['t_T']
        print(f"Updated t_T: {o_} -> {conf_arg['schedule_jump_params']['t_T']}")

    if args['jump_n_sample'] is not None:
        o_ = conf_arg['schedule_jump_params']['jump_n_sample'] 
        # print(args['jump_n_sample'])
        conf_arg['schedule_jump_params']['jump_n_sample'] = args['jump_n_sample']
        print(f"Updated jump_n_sample: {o_} -> {conf_arg['schedule_jump_params']['jump_n_sample']}")
    
    if args['inpaint_size'] is not None:
        o_ = conf_arg.data['eval']['paper_face_mask']['inpaint_size'] 
        # print(args['inpaint_size'])
        conf_arg.data['eval']['paper_face_mask']['inpaint_size'] = args['inpaint_size']
        print(f"Updated inpaint_size: {o_} -> {conf_arg.data['eval']['paper_face_mask']['inpaint_size']}")
    
    if args['jump_length'] is not None:
        o_ = conf_arg['schedule_jump_params']['jump_length'] 
        # print(args['jump_length'])
        conf_arg['schedule_jump_params']['jump_length'] = args['jump_length']
        print(f"Updated jump_length: {o_} -> {conf_arg['schedule_jump_params']['jump_length']}")
    print()
    
    print("GPU Tests:")
    print(th.cuda.device_count())

    #for i in range(th.cuda.device_count()):
        #print(th.cuda.get_device_properties(i).name)


    print("Done...")

    
    # if os.path.isfile(f"repaint_data/res_diffusion_{conf_arg.data['eval']['paper_face_mask']['inpaint_size']}_{conf_arg.schedule_jump_params['t_T']}_{conf_arg.schedule_jump_params['jump_length']}_{conf_arg.schedule_jump_params['jump_n_sample']}.json"):
    #     if not conf_arg.data['eval']['paper_face_mask']['diffusion_test']:
    #         print(f"skipping {conf_arg.data['eval']['paper_face_mask']['inpaint_size']}_{conf_arg.schedule_jump_params['t_T']}_{conf_arg.schedule_jump_params['jump_length']}_{conf_arg.schedule_jump_params['jump_n_sample']}")
    #         assert False
    # assert False
    main(conf_arg)
