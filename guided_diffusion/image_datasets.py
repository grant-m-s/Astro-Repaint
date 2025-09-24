# Galaxy Zoo Code used with permission from Mike Walmsley
# https://github.com/mwalmsley/galaxy-datasets

import math
import random

from PIL import Image
import blobfile as bf
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T


from json import load
from pathlib import Path
import os
import torch

# from .shared import gz2, gz_candels, gz_decals_5, gz_hubble,demo_rings, tidal

from . import galaxy_dataset

# TODO could refactor these into same class if needed

# class GZCandels(galaxy_dataset.GalaxyDataset):
    
#     def __init__(self, root, train=True, download=False, transform=None, target_transform=None):

#         catalog, label_cols = gz_candels(root, train, download)

#         super().__init__(catalog, label_cols, transform, target_transform)


class VIS_Cutouts(Dataset):
    
    def __init__(self, files_dir, ids_file, transform=None, target_transform=None, set="train"):
        
        self.files_dir = files_dir
        with open(f"{files_dir}/{ids_file}",'r') as fp:
            print(f" loading IDs from {files_dir}/{ids_file}")
            self.img_ids = load(fp)
        self.img_dir = files_dir
        self.npy_list = Path(files_dir).glob('*.npy')
        # print(list(self.npy_list))
        # assert False
        self.path_list = [str(x) for x in self.npy_list]
        print(self.path_list[:10])
        print("Full cutout dataset has ",1024*len(self.path_list)," images.")
        self.set = set
        import pickle
        if set == "train":
            with open(f"{files_dir}/train_set_files","rb") as fp:
                b = pickle.load(fp)
            p_list = [f"{files_dir}/{x}" for x in b]
            print(p_list[:20])
            # assert False
            self.path_list = p_list
        elif set == "test":
            with open(f"{files_dir}/test_set_files","rb") as fp:
                b = pickle.load(fp)
            p_list = [f"{files_dir}/{x}" for x in b]
            print(p_list[:20])
            self.path_list = p_list

            #assert False


        print("Used dataset has ",1024*len(self.path_list)," images.")
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):

        if self.set != "noise":
            img_path = self.path_list[idx]
            idx_num = img_path[len(self.files_dir)+1:img_path.index(".")]
            # print(img_path)
            # print(idx," vs ",idx_num)
            # assert False
            with open(img_path, 'rb') as fp:
                image = np.load(fp)

            labels = self.img_ids[f"{idx_num}"]
            
        else:
            noise_samples = 3
            mask_size = 5
            noise_steps = 60

            subset_name = "all"
            split = "test"

            with open(f"elsa_test_data/res_diffusion_{mask_size}_{noise_steps}_1_{noise_samples}_{split}_local.json",'r') as fp:
                res = load(fp)

            all_o = []

            for i in range(len(res["ids"])):
                curr_o =      np.array([float(x) for x in res["o_pixels"][i]]).reshape((mask_size,mask_size))
                all_o.append(curr_o)

            all_o = np.array(all_o)

            nonzero_o = all_o.flatten() > 0

            test_ = abs(np.random.normal(0,1,(1024,128,128)).astype(np.float32))
        
            # print(test_.min())
            all_o_copy = all_o.copy()

            nonzeros = [x for x in range(1024) if np.all(all_o_copy[x] > 0)]

            # print(len(nonzeros))

            # nonzero = all_o_copy > 0

            all_o_flattend = all_o_copy[nonzeros].flatten()
            all_o_flattend.sort()
            # print(all_o_copy[nonzeros,:,:].shape)
            rs = [all_o_flattend[x] for x in range(0,1024*25,32*25)]

            # print(len(rs))

            for i in range(32):
                for j in range(32):
                    test_[(i*32)+j] *= 5*rs[i]

            # print(test_.shape)

            image = test_
            
            labels = [f"{x}" for x in range(1024)]
        # img_max = np.max(image)
        # img_min = np.min(image)
        # image = (image-img_min) / (img_max - img_min)
        # print("---")
        # print(idx_num)
        # print(len(labels))
        # assert False
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            labels = self.target_transform(labels)

        B, H,W = image.shape
        image = image.view(B,1,H,W)
        # print(image.shape)
        # image = image.expand(-1,3,-1,-1)
        # print(image.shape)

        # assert False
        return image, labels



# class GZDecals5(galaxy_dataset.GalaxyDataset):
    
#     def __init__(self, root, train=True, download=False, transform=None, target_transform=None):

#         catalog, label_cols = gz_decals_5(root, train, download)

#         super().__init__(catalog, label_cols, transform, target_transform)



# class GZ2(galaxy_dataset.GalaxyDataset):
    
#     def __init__(self, root, train=True, download=False, transform=None, target_transform=None):

#         catalog, label_cols = gz2(root, train, download)  # no train arg

#         super().__init__(catalog, label_cols, transform, target_transform)


# class GZHubble(galaxy_dataset.GalaxyDataset):
    
#     def __init__(self, root, train=True, download=False, transform=None, target_transform=None):

#         catalog, label_cols = gz_hubble(root, train, download)

#         super().__init__(catalog, label_cols, transform, target_transform)


# class DemoRings(galaxy_dataset.GalaxyDataset):
    
#     def __init__(self, root, train=True, download=False, transform=None, target_transform=None):

#         catalog, label_cols = demo_rings(root, train, download)

#         super().__init__(catalog, label_cols, transform, target_transform)

# class Tidal(galaxy_dataset.GalaxyDataset):
    
#     def __init__(self, root, train=True, download=False, transform=None, target_transform=None, label_mode='coarse'):

#         catalog, label_cols = tidal(root, train, download, label_mode=label_mode)

#         super().__init__(catalog, label_cols, transform, target_transform)




# from .shared import gz_desi, gz_rings, gz_cosmic_dawn

# class GZDesi(galaxy_dataset.GalaxyDataset):
    
#     def __init__(self, root, train=True, download=False, transform=None, target_transform=None):

#         catalog, label_cols = gz_desi(root, train, download)

#         super().__init__(catalog, label_cols, transform, target_transform)

# class GZRings(galaxy_dataset.GalaxyDataset):
    
#     def __init__(self, root, train=True, download=False, transform=None, target_transform=None):

#         catalog, label_cols = gz_rings(root, train, download)

#         super().__init__(catalog, label_cols, transform, target_transform)

# class GZCosmic(galaxy_dataset.GalaxyDataset):
    
#     def __init__(self, root, train=True, download=False, transform=None, target_transform=None):

#         catalog, label_cols = gz_cosmic_dawn(root, train, download)

#         super().__init__(catalog, label_cols, transform, target_transform)



# temporarily deprecated
# class Legs(galaxy_dataset.GalaxyDataset):
    
#     # based on https://pytorch.org/vision/stable/generated/torchvision.datasets.STL10.html
#     def __init__(self, root=None, split='train', download=False, transform=None, target_transform=None, train=None):
#         # train=None is just an exception-raising parameter to avoid confused users using the train=False api

#         catalog, label_cols = legs(root, split, download, train)

#         # paths are not adjusted as cannot be downloaded
#         # catalog = _temp_adjust_catalog_paths(catalog)
#         # catalog = adjust_catalog_dtypes(catalog, label_cols)

#         super().__init__(catalog, label_cols, transform, target_transform)

class CustomTransform:
    def __init__(self, crop_margin=32, H=128):
        self.crop_margin = crop_margin
        self.transforms = T.Compose([
            T.Lambda(lambda x: torch.from_numpy(x)),  # Step 1: Convert np array to tensor
            # T.Lambda(self.reshape_tensor),            # Step 2: Reshape the tensor
            T.CenterCrop((self.crop_size, self.crop_size))  # Step 3: Center crop
        ])

    def reshape_tensor(self, tensor):
        # Original shape is [B, SB, H, W]
        B, SB, H, W = tensor.shape
        # Reshape to [B*SB, 1, H, W]
        return tensor.view(B * SB, 1, H, W)

    @property
    def crop_size(self):
        # Define the new height and width after cropping
        return self.calculate_crop_size()

    def calculate_crop_size(self):
        # Calculate the crop size based on the original height and width minus 32 pixels
        return 2 * self.crop_margin  # Assuming H and W are equal

    def __call__(self, np_array):
        return self.transforms(np_array)

def load_data(
    *,
    data_dir,
    ids_file,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=True,
    set="train",
    args=None,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    print("loading data...")
    # dataset = GZDecals5(
    #     "datasets/Decals", train=True, download=True, transform=None, target_transform=None
    # )

    # print("Finished...")
    # assert False

    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    # dataset = ImageDataset(
    #     image_size,
    #     all_files,
    #     classes=classes,
    #     shard=MPI.COMM_WORLD.Get_rank(),
    #     num_shards=MPI.COMM_WORLD.Get_size(),
    #     random_crop=random_crop,
    #     random_flip=random_flip,)
    # dataset = GZDecals5(
    #     "datasets/Decals", train=True, download=True, transform=None, target_transform=None
    # )
    # assert False
    transform = CustomTransform()
    memory_before = torch.cuda.memory_allocated("cuda")

    dataset = VIS_Cutouts(
        data_dir, ids_file, transform=transform, target_transform=None, set=set
    )
    memory_after = torch.cuda.memory_allocated("cuda")
    # latent_size = memory_after - memory_before
    # print("after dataset: ",latent_size/(1024**3))
    # print(f"Is Deterministic - workers=10: {deterministic}")
    if deterministic:
        print("Is Deterministic")
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        print("Isnt Deterministic")
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=10, drop_last=True
        )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
