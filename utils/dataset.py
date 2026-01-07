import json
import os
from pathlib import Path

import lmdb
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

from utils.lmdb import get_array_shape_from_lmdb, retrieve_row_from_lmdb


class TextDataset(Dataset):
    def __init__(self, prompt_path, extended_prompt_path=None):
        with open(prompt_path, encoding="utf-8") as f:
            self.prompt_list = [line.rstrip() for line in f]

        if extended_prompt_path is not None:
            with open(extended_prompt_path, encoding="utf-8") as f:
                self.extended_prompt_list = [line.rstrip() for line in f]
            assert len(self.extended_prompt_list) == len(self.prompt_list)
        else:
            self.extended_prompt_list = None

    def __len__(self):
        return len(self.prompt_list)

    def __getitem__(self, idx):
        batch = {
            "prompts": self.prompt_list[idx],
            "idx": idx,
        }
        if self.extended_prompt_list is not None:
            batch["extended_prompts"] = self.extended_prompt_list[idx]
        return batch


class TextFolderDataset(Dataset):
    def __init__(self, data_path, max_count=30000):
        self.texts = []
        count = 1
        for file in os.listdir(data_path):
            if file.endswith(".txt"):
                with open(os.path.join(data_path, file), "r") as f:
                    text = f.read().strip()
                    self.texts.append(text)
                    count += 1
                    if count > max_count:
                        break

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {"prompts": self.texts[idx], "idx": idx}


class ODERegressionLMDBDataset(Dataset):
    def __init__(self, data_path: str, max_pair: int = int(1e8)):
        self.env = lmdb.open(
            data_path, readonly=True, lock=False, readahead=False, meminit=False
        )

        self.latents_shape = get_array_shape_from_lmdb(self.env, "latents")
        self.max_pair = max_pair

    def __len__(self):
        return min(self.latents_shape[0], self.max_pair)

    def __getitem__(self, idx):
        """
        Outputs:
            - prompts: List of Strings
            - latents: Tensor of shape (num_denoising_steps, num_frames, num_channels, height, width). It is ordered from pure noise to clean image.
        """
        latents = retrieve_row_from_lmdb(
            self.env, "latents", np.float16, idx, shape=self.latents_shape[1:]
        )

        if len(latents.shape) == 4:
            latents = latents[None, ...]

        prompts = retrieve_row_from_lmdb(self.env, "prompts", str, idx)
        return {
            "prompts": prompts,
            "ode_latent": torch.tensor(latents, dtype=torch.float32),
        }


class ShardingLMDBDataset(Dataset):
    def __init__(self, data_path: str, max_pair: int = int(1e8)):
        self.envs = []
        self.index = []

        for fname in sorted(os.listdir(data_path)):
            path = os.path.join(data_path, fname)
            env = lmdb.open(
                path, readonly=True, lock=False, readahead=False, meminit=False
            )
            self.envs.append(env)

        self.latents_shape = [None] * len(self.envs)
        for shard_id, env in enumerate(self.envs):
            self.latents_shape[shard_id] = get_array_shape_from_lmdb(env, "latents")
            for local_i in range(self.latents_shape[shard_id][0]):
                self.index.append((shard_id, local_i))

            # print("shard_id ", shard_id, " local_i ", local_i)

        self.max_pair = max_pair

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        """
        Outputs:
            - prompts: List of Strings
            - latents: Tensor of shape (num_denoising_steps, num_frames, num_channels, height, width). It is ordered from pure noise to clean image.
        """
        shard_id, local_idx = self.index[idx]

        latents = retrieve_row_from_lmdb(
            self.envs[shard_id],
            "latents",
            np.float16,
            local_idx,
            shape=self.latents_shape[shard_id][1:],
        )

        if len(latents.shape) == 4:
            latents = latents[None, ...]

        prompts = retrieve_row_from_lmdb(self.envs[shard_id], "prompts", str, local_idx)

        img = retrieve_row_from_lmdb(
            self.envs[shard_id], "img", np.uint8, local_idx, shape=(480, 832, 3)
        )
        img = Image.fromarray(img)
        img = TF.to_tensor(img).sub_(0.5).div_(0.5)

        return {
            "prompts": prompts,
            "ode_latent": torch.tensor(latents, dtype=torch.float32),
            "img": img,
        }


class TextImagePairDataset(Dataset):
    def __init__(
        self, data_dir, transform=None, eval_first_n=-1, pad_to_multiple_of=None
    ):
        """
        Args:
            data_dir (str): Path to the directory containing:
                - target_crop_info_*.json (metadata file)
                - */ (subdirectory containing images with matching aspect ratio)
            transform (callable, optional): Optional transform to be applied on the image
        """
        self.transform = transform
        data_dir = Path(data_dir)

        # Find the metadata JSON file
        metadata_files = list(data_dir.glob("target_crop_info_*.json"))
        if not metadata_files:
            raise FileNotFoundError(f"No metadata file found in {data_dir}")
        if len(metadata_files) > 1:
            raise ValueError(f"Multiple metadata files found in {data_dir}")

        metadata_path = metadata_files[0]
        # Extract aspect ratio from metadata filename (e.g. target_crop_info_26-15.json -> 26-15)
        aspect_ratio = metadata_path.stem.split("_")[-1]

        # Use aspect ratio subfolder for images
        self.image_dir = data_dir / aspect_ratio
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        # Load metadata
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

        eval_first_n = eval_first_n if eval_first_n != -1 else len(self.metadata)
        self.metadata = self.metadata[:eval_first_n]

        # Verify all images exist
        for item in self.metadata:
            image_path = self.image_dir / item["file_name"]
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

        self.dummy_prompt = "DUMMY PROMPT"
        self.pre_pad_len = len(self.metadata)
        if (
            pad_to_multiple_of is not None
            and len(self.metadata) % pad_to_multiple_of != 0
        ):
            # Duplicate the last entry
            self.metadata += [self.metadata[-1]] * (
                pad_to_multiple_of - len(self.metadata) % pad_to_multiple_of
            )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        """
        Returns:
            dict: A dictionary containing:
                - image: PIL Image
                - caption: str
                - target_bbox: list of int [x1, y1, x2, y2]
                - target_ratio: str
                - type: str
                - origin_size: tuple of int (width, height)
        """
        item = self.metadata[idx]

        # Load image
        image_path = self.image_dir / item["file_name"]
        image = Image.open(image_path).convert("RGB")

        # Apply transform if specified
        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "prompts": item["caption"],
            "target_bbox": item["target_crop"]["target_bbox"],
            "target_ratio": item["target_crop"]["target_ratio"],
            "type": item["type"],
            "origin_size": (item["origin_width"], item["origin_height"]),
            "idx": idx,
        }


class ImageEditDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        """
        Args:
            csv_path (str or Path): Path to the metadata_edit.csv file.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.csv_path = Path(csv_path)
        # The images are usually stored relative to the CSV file's directory
        self.root_dir = self.csv_path.parent

        print(f"[Dataset] Initializing from: {self.csv_path}")

        # Read the CSV
        try:
            self.meta_df = pd.read_csv(self.csv_path)
        except Exception as e:
            print(f"[Dataset] Error reading CSV: {e}")
            raise e

        # Extract columns to lists for faster indexing
        self.img_paths = self.meta_df["edit_image"].tolist()
        self.prompts = self.meta_df["prompt"].tolist()

        self.transform = transform

        print(f"[Dataset] Successfully loaded. Found {len(self.img_paths)} images.")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # 1. Get the image path and prompt directly from the loaded data
        rel_path = self.img_paths[idx]
        prompt = str(self.prompts[idx])  # Ensure it is a string

        # 2. Construct full image path
        img_path = self.root_dir / rel_path

        # TODO: hard code 1024x1024 for now
        image = Image.open(img_path).convert("RGB").resize((1024, 1024))

        image = TF.to_tensor(image)

        if self.transform:
            image = self.transform(image)

        return {"img": image, "prompts": prompt}


def cycle(dl):
    while True:
        for data in dl:
            yield data
