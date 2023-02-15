import ast
from collections import defaultdict
import gc
import os

import cupy as cp
from cuml.metrics import pairwise_distances
from cuml.neighbors import NearestNeighbors
import cv2
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, default_collate, DataLoader
from tqdm import tqdm
from transformers import TrainerCallback

from utils import get_pos_score


class SDITPDataset(Dataset):
    def __init__(
        self,
        pairs_df,
        prompt_df,
        image_df,
        correlation_df,
        image_folder,
        size=(512, 512),
        objective="cosine",  # cosine or constrative
    ):
        self.pairs_df = pairs_df
        self.prompt_df = prompt_df
        self.image_df = image_df
        self.image_folder = image_folder
        self.correlation_df = correlation_df
        self.size = size
        self.objective = objective

        self.image_id_to_path_dict = dict(
            zip(list(self.image_df.id.values), list(self.image_df.path.values))
        )
        self.prompt_id_to_emb_dict = dict(
            zip(list(self.prompt_df.id.values), list(self.prompt_df.emb.values))
        )

        self.image_paths, self.prompt_embs, self.labels = self.preprocess_df()

    def preprocess_df(self):
        # get image path from self.image_id_to_path_dict and pairs
        image_ids = list(self.pairs_df.image_id.values)
        image_paths = [self.image_id_to_path_dict[id] for id in image_ids]
        image_paths = [os.path.join(self.image_folder, name) for name in image_paths]
        labels = list(self.pairs_df.target.values)

        prompt_ids = self.pairs_df.prompt_id
        embs = [self.prompt_id_to_emb_dict[id] for id in prompt_ids]
        return image_paths, embs, labels

    def __len__(self):
        return len(self.pairs_df)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.resize(image, self.size)
        image = image.transpose(2, 0, 1)

        emb = self.prompt_embs[idx]
        emb = ast.literal_eval(
            emb.replace("[ ", "[")
            .replace("  ", ",")
            .replace(" ", ",")
            .replace("\n", "")
        )

        label = self.labels[idx]
        return (
            torch.tensor(image).float(),
            torch.tensor(emb).float(),
            torch.tensor(int(label)).float(),
        )


def collate_fn(batch):
    images, embs, labels = zip(*batch)
    images = torch.stack(images)
    embs = torch.stack(embs)
    labels = torch.stack(labels)
    return {"images": images, "embs": embs, "labels": labels}


class InferenceDataset(Dataset):
    def __init__(self, image_paths, size=(512, 512)):
        self.image_paths = image_paths
        self.size = size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.resize(image, self.size)
        return image


class DatasetUpdateCallback(TrainerCallback):
    """
    Trigger re-computing dataset

    A hack that modifies the train/val dataset, pointed by Trainer's dataloader

    0. Calculate new train/val topic/content embeddings, train KNN, get new top-k
    1. Calculate top-k max positive score, compare to current val best, if greater, continue to step 2, else do nothing
    2. Update supervised_df and update dataset
    """

    def __init__(
        self,
        trainer,
        train_prompt_ids,
        val_prompt_ids,
        prompt_df,
        image_df,
        image_folder,
        correlation_df,
    ):
        self.trainer = trainer
        self.train_prompt_ids = train_prompt_ids
        self.val_prompt_ids = val_prompt_ids
        self.prompt_df = prompt_df
        self.image_df = image_df
        self.image_folder = image_folder
        self.correlation_df = correlation_df

        # create dataloader to calculate embedding for images
        image_paths = self.image_df.path.values
        self.inference_dataset = InferenceDataset(image_paths)
        self.inference_dataloader = DataLoader(
            self.inference_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=32,
            pin_memory=True,
            drop_last=False,
        )

    def update_dataset(self):
        pass

    def on_train_begin(self, args, state, control, **kwargs):
        # update dataset at the beginning of training
        pass

    def on_epoch_end(self, args, state, control, **kwargs):
        # update dataset at the end of each epoch
        pass
