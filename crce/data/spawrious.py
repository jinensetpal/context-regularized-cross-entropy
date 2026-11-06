#!/usr/bin/env python3

"""
The code for the Spawrious dataset was adapted from the Spawrious codebase:
    https://github.com/aengusl/spawrious/blob/651e90415bfb414a5b6579cb1f53b83e6bb8c165/spawrious/torch.py
"""

from typing import Any, Tuple
import urllib.request
import tarfile
import urllib
import os


from torch.utils.data import ConcatDataset, Dataset
import torchvision.transforms.v2.functional as F
from torchvision.datasets import ImageFolder
import torchvision.transforms.v2 as T
from PIL import ImageFile
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np
import itertools
import random
import torch
import timm

from crce.utils import DataLoader
from crce import const

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _extract_dataset_from_tar(
    tar_file_name: str, data_dir: str, remove_tar_after_extracting: bool = True
) -> None:
    tar_file_dst = os.path.join(data_dir, tar_file_name)
    print("Extracting dataset...")
    tar = tarfile.open(tar_file_dst, "r:gz")
    tar.extractall(os.path.dirname(tar_file_dst))
    tar.close()
    print("Dataset extracted. Delete tar file.")
    if remove_tar_after_extracting:
        os.remove(tar_file_dst)


def _download_dataset_if_not_available(
    dataset_name: str, data_dir: str, remove_tar_after_extracting: bool = True
) -> None:
    """
    datasets.txt file, which is present in the data_dir, is used to check if the dataset is already extracted. If the dataset is already extracted, then the tar file is not downloaded again.
    """
    data_dir = data_dir.split("/spawrious224/")[
        0
    ]  # in case people pass in the wrong root_dir
    os.makedirs(data_dir, exist_ok=True)
    dataset_name = dataset_name.lower()
    if dataset_name.split("_")[0] == "m2m":
        dataset_name = "entire_dataset"
    url_dict = {
        "entire_dataset": "https://www.dropbox.com/s/hofkueo8qvaqlp3/spawrious224__entire_dataset.tar.gz?dl=1",
        "o2o_easy": "https://www.dropbox.com/s/kwhiv60ihxe3owy/spawrious224__o2o_easy.tar.gz?dl=1",
        "o2o_medium": "https://www.dropbox.com/s/x03gkhdwar5kht4/spawrious224__o2o_medium.tar.gz?dl=1",
        "o2o_hard": "https://www.dropbox.com/s/p1ry121m2gjj158/spawrious224__o2o_hard.tar.gz?dl=1",
        # "m2m": "https://www.dropbox.com/s/5usem63nfub266y/spawrious__m2m.tar.gz?dl=1",
    }
    tar_file_name = f"spawrious224__{dataset_name}.tar.gz"
    tar_file_dst = os.path.join(data_dir, tar_file_name)
    url = url_dict[dataset_name]

    # check if the dataset is already extracted
    if _check_images_availability(data_dir, dataset_name):
        print("Dataset already downloaded and extracted.")
        return
    # check if the tar file is already downloaded
    else:
        if os.path.exists(tar_file_dst):
            print("Dataset already downloaded. Extracting...")
            _extract_dataset_from_tar(
                tar_file_name, data_dir, remove_tar_after_extracting
            )
            return
        # download the tar file and extract from it
        else:
            print("Dataset not found. Downloading...")
            response = urllib.request.urlopen(url)
            total_size = int(response.headers.get("Content-Length", 0))
            block_size = 1024
            # Track progress of download
            progress_bar = tqdm(total=total_size, unit="iB", unit_scale=True)
            with open(tar_file_dst, "wb") as f:
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    f.write(buffer)
                    progress_bar.update(len(buffer))
            progress_bar.close()
            print("Dataset downloaded. Extracting...")
            _extract_dataset_from_tar(
                tar_file_name, data_dir, remove_tar_after_extracting
            )
            return


class CustomImageFolder(Dataset):
    """
    A class that takes one folder at a time and loads a set number of images in a folder and assigns them a specific class
    """

    def __init__(
        self, folder_path, class_index, location_index, return_location_label, add_mask_noise, mix_object_contexts, comb_list, limit=None, transform=None, split='test'
    ):
        self.folder_path = folder_path
        root_index = self.folder_path.index('spawrious224/') + len('spawrious224/')

        self.split = split
        self.class_index = class_index
        self.location_index = location_index
        self.return_location_label = return_location_label
        self.image_mask_paths = [
                (os.path.join(folder_path, img), os.path.join(self.folder_path[:root_index], 'masks', self.folder_path[root_index:], img))
                for img in os.listdir(folder_path)
                if img.endswith((".png", ".jpg", ".jpeg"))
        ]
        if limit:
            self.image_mask_paths = self.image_mask_paths[:limit]

        self.mask_transform = T.Compose([T.Resize(const.CAM_SIZE, T.InterpolationMode.NEAREST_EXACT), T.ToImage(), T.ToDtype(torch.float32, scale=True)])

        if len(transform.transforms) == 5:  # pull img+mask augment stages
            self.pre_transform = transform.transforms[:2]
            self.transform = T.Compose(transform.transforms[2:])
        else:
            self.transform = transform
            self.pre_transform = None

        self.y = torch.zeros(4)
        self.y[self.class_index] = 1

        self.add_mask_noise = add_mask_noise
        if self.add_mask_noise:
            self.mask_noise_transform = T.RandomAffine(5, translate=(0.075, 0.075), scale=(0.7, 1.3))

        self.mix_object_contexts = mix_object_contexts
        if self.mix_object_contexts:
            folder_path = Path(folder_path)

            other_classes = [x.name for x in folder_path.parent.glob('*') if x.name != folder_path.name]
            other_locations = [x[0] for x in comb_list if x[0] != folder_path.parent.name]

            other_images = []
            for other_location, other_class in itertools.product(other_locations, other_classes):
                other_images.extend(list((folder_path.parent.parent / other_location / other_class).glob('*')))

            self.alternate_contexts = random.choices(other_images, k=len(self) // 2)

    def __len__(self):
        return (1 + int(self.mix_object_contexts)) * len(self.image_mask_paths)

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        index_mix_object_contexts = False
        if index >= len(self.image_mask_paths):  # load the context-shifted image
            index_mix_object_contexts = True
            assert self.mix_object_contexts
            index -= len(self.image_mask_paths)

        img_path = self.image_mask_paths[index][0]
        img = Image.open(img_path).convert('RGB')

        mask_path = self.image_mask_paths[index][1]
        mask = Image.open(mask_path)

        if index_mix_object_contexts:
            alternate_context = Image.open(self.alternate_contexts[index]).convert('RGB')

            mask_arr = np.expand_dims(np.array(mask), axis=2) != 0
            img = np.array(img) * mask_arr + np.array(alternate_context) * (1 - mask_arr)
            img = Image.fromarray(img.astype(np.uint8), mode='RGB')

        if self.pre_transform is not None:
            i, j, h, w = self.pre_transform[0].get_params(img, self.pre_transform[0].scale, self.pre_transform[0].ratio)

            img = F.resized_crop(img, i, j, h, w, self.pre_transform[0].size, self.pre_transform[0].interpolation)
            mask = F.resized_crop(mask, i, j, h, w, self.pre_transform[0].size, self.pre_transform[0].interpolation)

            state = torch.get_rng_state()
            img = self.pre_transform[1](img)
            torch.set_rng_state(state)
            mask = self.pre_transform[1](mask)

        if self.transform: img = self.transform(img)

        if self.add_mask_noise:
            mask = self.mask_noise_transform(mask)
            mask_arr = np.array(mask)

        mask = self.mask_transform(mask)[0]
        mask = (mask > .5).to(torch.uint8)

        if mask.sum() == 0 and self.split == 'train': return None  # mask[:] = 1.

        location_label = torch.tensor(self.location_index, dtype=torch.long)
        if self.return_location_label: return img, (mask, location_label, self.y)
        return img, (mask, self.y)


class MultipleDomainDataset:
    N_STEPS = 5001  # Default, subclasses may override
    CHECKPOINT_FREQ = 100  # Default, subclasses may override
    ENVIRONMENTS = None  # Subclasses should override
    INPUT_SHAPE = None  # Subclasses should override

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)


def build_combination(benchmark_type, group, test, filler=None):
    total = 3168
    combinations = {}
    if "m2m" in benchmark_type:
        counts = [total, total]
        combinations["train_combinations"] = {
            ("bulldog",): [(group[0], counts[0]), (group[1], counts[1])],
            ("dachshund",): [(group[1], counts[0]), (group[0], counts[1])],
            ("labrador",): [(group[2], counts[0]), (group[3], counts[1])],
            ("corgi",): [(group[3], counts[0]), (group[2], counts[1])],
        }
        combinations["test_combinations"] = {
            ("bulldog",): [test[0], test[1]],
            ("dachshund",): [test[1], test[0]],
            ("labrador",): [test[2], test[3]],
            ("corgi",): [test[3], test[2]],
        }
    else:
        counts = [int(0.97 * total), int(0.87 * total)]
        combinations["train_combinations"] = {
            ("bulldog",): [(group[0], counts[0]), (group[0], counts[1])],
            ("dachshund",): [(group[1], counts[0]), (group[1], counts[1])],
            ("labrador",): [(group[2], counts[0]), (group[2], counts[1])],
            ("corgi",): [(group[3], counts[0]), (group[3], counts[1])],
            ("bulldog", "dachshund", "labrador", "corgi"): [
                (filler, total - counts[0]),
                (filler, total - counts[1]),
            ],
        }
        combinations["test_combinations"] = {
            ("bulldog",): [test[0], test[0]],
            ("dachshund",): [test[1], test[1]],
            ("labrador",): [test[2], test[2]],
            ("corgi",): [test[3], test[3]],
        }
    return combinations


def _get_combinations(benchmark_type: str) -> Tuple[dict, dict]:
    combinations = {
        "o2o_easy": (
            ["desert", "jungle", "dirt", "snow"],
            ["dirt", "snow", "desert", "jungle"],
            "beach",
        ),
        "o2o_medium": (
            ["mountain", "beach", "dirt", "jungle"],
            ["jungle", "dirt", "beach", "snow"],
            "desert",
        ),
        "o2o_hard": (
            ["jungle", "mountain", "snow", "desert"],
            ["mountain", "snow", "desert", "jungle"],
            "beach",
        ),
        "m2m_hard": (
            ["dirt", "jungle", "snow", "beach"],
            ["snow", "beach", "dirt", "jungle"],
            None,
        ),
        "m2m_easy": (
            ["desert", "mountain", "dirt", "jungle"],
            ["dirt", "jungle", "mountain", "desert"],
            None,
        ),
        "m2m_medium": (
            ["beach", "snow", "mountain", "desert"],
            ["desert", "mountain", "beach", "snow"],
            None,
        ),
    }
    if benchmark_type not in combinations:
        raise ValueError("Invalid benchmark type")
    group, test, filler = combinations[benchmark_type]
    return build_combination(benchmark_type, group, test, filler)


class SpawriousBenchmark(MultipleDomainDataset):
    ENVIRONMENTS = ["Test", "SC_group_1", "SC_group_2"]
    input_shape = (3, 224, 224)
    num_classes = 4
    class_list = ["bulldog", "corgi", "dachshund", "labrador"]
    locations_list = ["desert", "jungle", "dirt", "mountain", "snow", "beach"]

    def __init__(self, benchmark, root_dir, return_location_label, add_mask_noise, mix_object_contexts):
        combinations = _get_combinations(benchmark.lower())
        self.type1 = benchmark.lower().startswith("o2o")
        train_datasets, test_datasets = self._prepare_data_lists(
            combinations["train_combinations"],
            combinations["test_combinations"],
            root_dir,
            return_location_label,
            add_mask_noise,
            mix_object_contexts,
        )
        self.datasets = [ConcatDataset(test_datasets)] + train_datasets

    def get_train_dataset(self):
        return torch.utils.data.ConcatDataset(self.datasets[1:])

    def get_test_dataset(self):
        return self.datasets[0]

    # Prepares the train and test data lists by applying the necessary transformations.
    def _prepare_data_lists(
        self, train_combinations, test_combinations, root_dir, return_location_label, add_mask_noise, mix_object_contexts
    ):
        backbone = timm.create_model(
            "swin_base_patch4_window7_224.ms_in22k_ft_in1k",
            pretrained=True,
            num_classes=0,
        ).eval()
        self.data_config = timm.data.resolve_model_data_config(backbone)
        transforms = timm.data.create_transform(
            **self.data_config, is_training=False
        )

        train_data_list = self._create_data_list(
            train_combinations, root_dir, transforms, return_location_label, 'train', add_mask_noise, mix_object_contexts
        )
        test_data_list = self._create_data_list(
            test_combinations, root_dir, transforms, return_location_label, 'test', False, False
        )

        return train_data_list, test_data_list

    # Creates a list of datasets based on the given combinations and transformations.
    def _create_data_list(self, combinations, root_dir, transforms, return_location_label, split, add_mask_noise, mix_object_contexts):
        data_list = []
        if isinstance(combinations, dict):
            # Build class groups for a given set of combinations, root directory, and transformations.
            for_each_class_group = []
            cg_index = 0
            for classes, comb_list in combinations.items():
                for_each_class_group.append([])
                for ind, location_limit in enumerate(comb_list):
                    if isinstance(location_limit, tuple):
                        location, limit = location_limit
                    else:
                        location, limit = location_limit, None
                    cg_data_list = []
                    for cls in classes:
                        path = os.path.join(
                            root_dir,
                            "spawrious224",
                            f"{0 if not self.type1 else ind}/{location}/{cls}",
                        )
                        data = CustomImageFolder(
                            folder_path=path,
                            class_index=self.class_list.index(cls),
                            location_index=self.locations_list.index(location),
                            return_location_label=return_location_label,
                            add_mask_noise=add_mask_noise,
                            mix_object_contexts=mix_object_contexts,
                            comb_list=comb_list,
                            limit=limit,
                            transform=transforms,
                            split=split,
                        )
                        cg_data_list.append(data)

                    for_each_class_group[cg_index].append(ConcatDataset(cg_data_list))
                cg_index += 1

            for group in range(len(for_each_class_group[0])):
                data_list.append(
                    ConcatDataset(
                        [
                            for_each_class_group[k][group]
                            for k in range(len(for_each_class_group))
                        ]
                    )
                )
        else:
            for location in combinations:
                path = os.path.join(root_dir, f"{0}/{location}/")
                data = ImageFolder(root=path, transform=transforms)
                data_list.append(data)

        return data_list


def _check_images_availability(root_dir: str, dataset_type: str) -> bool:
    # Get the combinations for the given dataset type
    root_dir = root_dir.split("/spawrious224/")[
        0
    ]  # in case people pass in the wrong root_dir
    if dataset_type == "entire_dataset":
        for dataset in ["0", "1", "domain_adaptation_ds"]:
            for location in ["snow", "jungle", "desert", "dirt", "mountain", "beach"]:
                for cls in ["bulldog", "corgi", "dachshund", "labrador"]:
                    path = os.path.join(
                        root_dir, "spawrious224", f"{dataset}/{location}/{cls}"
                    )
                    if not os.path.exists(path) or not any(
                        img.endswith((".png", ".jpg", ".jpeg"))
                        for img in os.listdir(path)
                    ):
                        return False
        return True
    combinations = _get_combinations(dataset_type.lower())

    # Extract the train and test combinations
    train_combinations = combinations["train_combinations"]
    test_combinations = combinations["test_combinations"]

    # Check if the relevant images for each combination are present in the root directory
    for combination in [train_combinations, test_combinations]:
        for classes, comb_list in combination.items():
            for ind, location_limit in enumerate(comb_list):
                if isinstance(location_limit, tuple):
                    location, _ = location_limit
                else:
                    location, _ = location_limit, None

                for cls in classes:
                    path = os.path.join(
                        root_dir,
                        "spawrious224",
                        f"{0 if not dataset_type.lower().startswith('o2o') else ind}/{location}/{cls}",
                    )

                    # If the path does not exist or there are no relevant images, return False
                    if not os.path.exists(path) or not any(
                        img.endswith((".png", ".jpg", ".jpeg"))
                        for img in os.listdir(path)
                    ):
                        return False

    # If all the required images are present, return True
    return True


def get_spawrious_dataset(root_dir, dataset_name, return_location_label, add_mask_noise, mix_object_contexts):
    """
    Returns the dataset as a torch dataset, and downloads dataset if dataset is not already available.

    By default, the entire dataset is downloaded, which is necessary for m2m experiments, and domain adaptation experiments
    """
    root_dir = root_dir.split("/spawrious224/")[
        0
    ]  # in case people pass in the wrong root_dir
    assert dataset_name.lower() in {
        "o2o_easy",
        "o2o_medium",
        "o2o_hard",
        "m2m_easy",
        "m2m_medium",
        "m2m_hard",
        "m2m",
        "entire_dataset",
    }, f"Invalid dataset type: {dataset_name}"
    _download_dataset_if_not_available(dataset_name, root_dir)
    return SpawriousBenchmark(dataset_name, root_dir, return_location_label, add_mask_noise, mix_object_contexts)


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def get_generators(benchmark_type, return_location_label=False):
    base_dataset = get_spawrious_dataset(const.DATA_DIR.as_posix(), benchmark_type, return_location_label, add_mask_noise=const.ADD_MASK_NOISE, mix_object_contexts=const.MIX_OBJECT_CONTEXTS)

    val_split = 0.2
    train_set = base_dataset.get_train_dataset()
    val_size = int(len(train_set) * val_split)

    datasets = [*torch.utils.data.random_split(train_set, [len(train_set) - val_size, val_size]), base_dataset.get_test_dataset()]
    return [DataLoader(dataset, shuffle=True if idx < 2 else False, num_workers=const.N_WORKERS, pin_memory=True, batch_size=const.BATCH_SIZE, collate_fn=collate_fn) for idx, dataset in enumerate(datasets)]


def generate_masks():
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model
    from glob import glob

    sam3 = build_sam3_image_model()
    processor = Sam3Processor(sam3)

    image_list = glob((const.DATA_DIR / 'spawrious224' / 'masks' / '**' / '*.png').as_posix(), recursive=True)  # must copy all images to mask as placeholder first

    failed_mask_count = 0
    for image_path in tqdm(image_list):
        image = Image.open(image_path)
        inference_state = processor.set_image(image)

        res = processor.set_text_prompt(state=inference_state, prompt=f'{image_path.split("/")[-2]} or dog')
        mask = res['masks'][res['scores'] > .5].sum(0).clamp(max=1).to(torch.float)
        if mask.sum() == 0:
            failed_mask_count += 1
        T.ToPILImage()(mask).save(image_path)

    print("Failed mask %:", failed_mask_count / len(image_list))


if __name__ == '__main__':
    generate_masks()
