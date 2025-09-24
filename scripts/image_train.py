"""
Train a diffusion model on images.
"""

import argparse

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop


def main():
    args = create_argparser().parse_args()
    d_set_locs = {
        "default":"/shared/cutouts_benchmarking/batches",
        "no_pointlike":"/shared/cutouts_benchmarking/batches/no_pointlike"
    }

    # assert False

    dist_util.setup_dist()
    logger.configure(dir=f"./training_ds-{args.dataset}_dp-{args.data_processing}_ls-{args.loss_function}")

    logger.log("creating model and diffusion...")

    args.dataset_dir = d_set_locs[args.dataset]
    args.training_dir = logger.get_dir()
    print(args)

    # assert False
    # data = load_data(
    #     data_dir=args.data_dir,
    #     batch_size=args.batch_size,
    #     image_size=args.image_size,
    #     class_cond=args.class_cond,
    # )
    # import matplotlib.pyplot as plt
    # from copy import deepcopy
    # import cv2
    # import numpy as np
    # for i,j in data:
    #     for k_idx,k in enumerate(i):
    #         norm_image = cv2.normalize(k.permute(1, 2, 0).cpu().numpy(), None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)

    #         norm_image = norm_image.astype(np.uint8)
    #         # print(img.permute(1, 2, 0).shape)
    #         cv2.imwrite(f"../RePaint/data/datasets/gts/galaxys/{k_idx}.png", norm_image) 



    #     assert False
    # print(args)
    model, diffusion = create_model_and_diffusion(args=args,
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)



    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.dataset_dir,
        ids_file="batch_source_full.json",
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )

    logger.log("training...")
    # print(model)
    # from torchsummary import summary
    # print(model)
    # assert False
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        args=args
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=50,
        save_interval=5000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)

    parser.add_argument('--dataset', type=str, default="default",choices=["default","no_pointlike"])
    parser.add_argument('--data_processing', type=str, default="default",choices=["default","norm_max_pixel","min_max"])
    parser.add_argument('--loss_function', type=str, default="default",choices=["default","1_over_pixel","min_max","huber"])
    # parser.add_argument('--loss_function', type=str, default="default",choices=["default","1_over_pixel","min_max","huber"])

    return parser


if __name__ == "__main__":
    main()
