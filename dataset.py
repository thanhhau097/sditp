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

from utils import get_pos_score, f2_score


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
        is_train=True,
    ):
        self.pairs_df = pairs_df
        self.prompt_df = prompt_df
        self.image_df = image_df
        self.image_folder = image_folder
        self.correlation_df = correlation_df
        self.size = size
        self.objective = objective
        self.is_train = is_train

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

        if not self.is_train or self.objective == "cosine": # self.objective == "cosine" and 
            # only get positive labels
            image_paths = [image_paths[i] for i, label in enumerate(labels) if label == 1]
            embs = [embs[i] for i, label in enumerate(labels) if label == 1]
            labels = [label for label in labels if label == 1]

        return image_paths, embs, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.resize(image, self.size)
        image = image.transpose(2, 0, 1) / 255

        emb = self.prompt_embs[idx]
        try:
            emb = ast.literal_eval(emb)
        except:
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
        image = image.transpose(2, 0, 1) / 255
        return torch.tensor(image).float()


class DatasetUpdateCallback(TrainerCallback):
    """
    Trigger re-computing dataset

    A hack that modifies the train/val dataset, pointed by Trainer's dataloader

    0. Calculate new train/val prompt/image embeddings, train KNN, get new top-k
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
        best_score=0,
        top_k=50,
    ):
        self.trainer = trainer
        self.train_prompt_ids = train_prompt_ids
        self.val_prompt_ids = val_prompt_ids
        self.prompt_df = prompt_df
        self.image_df = image_df
        self.image_folder = image_folder
        self.correlation_df = correlation_df
        self.best_score = best_score
        self.top_k = top_k

        # retrieve train/val prompt embeddings from prompt_df
        self.train_prompt_embs = [
            ast.literal_eval(
                x.replace("[ ", "[")
                .replace("  ", ",")
                .replace(" ", ",")
                .replace("\n", "")
            )
            for x in tqdm(self.prompt_df[self.prompt_df.id.isin(self.train_prompt_ids)].emb.values)
        ]

        def convert_emb(emb):
            try:
                emb = ast.literal_eval(emb)
            except:
                emb = ast.literal_eval(
                    emb.replace("[ ", "[")
                    .replace("  ", ",")
                    .replace(" ", ",")
                    .replace("\n", "")
                )
            return emb

        self.val_prompt_embs = [
            convert_emb(x) for x in tqdm(self.prompt_df[self.prompt_df.id.isin(self.val_prompt_ids)].emb.values)
        ]

        # create dataloader to calculate embedding for images
        def inference_collate_fn(batch):
            images = torch.stack(batch)
            return images

        image_paths = [os.path.join(image_folder, path) for path in self.image_df.path.values]
        self.inference_dataset = InferenceDataset(image_paths)
        self.inference_dataloader = DataLoader(
            self.inference_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=32,
            pin_memory=True,
            drop_last=False,
            collate_fn=inference_collate_fn,
        )

    def update_dataset(self):
        print("Calculating new image embeddings")
        self.trainer.model.eval()
        image_embs = []
        for batch in self.inference_dataloader:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            batch = batch.to(device)
            image_embs.extend(
                self.trainer.model(batch).detach().cpu().numpy()
            )

        prompt_embs = cp.array(self.val_prompt_embs)
        image_embs = cp.array(image_embs)

        # Release memory
        torch.cuda.empty_cache()

        # KNN model
        image_idx_to_id = {}
        for i, row in self.image_df.iterrows():
            image_idx_to_id[i] = row.id

        print("Evaluating current score...")
        for selected_k in [1, 2, 5, 10, 20, 50]:
            neighbors_model = NearestNeighbors(n_neighbors=selected_k, metric="cosine")
            neighbors_model.fit(image_embs)

            indices = neighbors_model.kneighbors(prompt_embs, return_distance=False)

            predictions = []
            for k in tqdm(range(len(indices))):
                pred = indices[k]
                p = " ".join([image_idx_to_id[ind] for ind in pred.get()])
                predictions.append(p)

            knn_preds = pd.DataFrame(
                {"prompt_id": self.val_prompt_ids, "image_id": predictions}
            ).sort_values("prompt_id")

            gt = self.correlation_df[
                self.correlation_df.prompt_id.isin(self.val_prompt_ids)
            ].sort_values("prompt_id")
            score = get_pos_score(
                gt["image_id"],
                knn_preds.sort_values("prompt_id")["image_id"],
                selected_k,
            )
            print(
                "Selecting",
                selected_k,
                "nearest images",
                "top-k score =",
                f2_score(
                    gt["image_id"],
                    knn_preds.sort_values("prompt_id")["image_id"],
                ),
                "max positive score =",
                score,
            )

        print("Training KNN model...")
        print("Generating KNN predictions with top_k =", self.top_k)
        neighbors_model = NearestNeighbors(n_neighbors=self.top_k, metric="cosine")
        neighbors_model.fit(image_embs)

        print("Generating embedding for validation topics")
        indices = neighbors_model.kneighbors(prompt_embs, return_distance=False)
        predictions = []
        for k in tqdm(range(len(indices))):
            pred = indices[k]
            p = " ".join([image_idx_to_id[ind] for ind in pred.get()])
            predictions.append(p)

        knn_preds = pd.DataFrame(
            {"prompt_id": self.val_prompt_ids, "image_id": predictions}
        ).sort_values("prompt_id")

        score = get_pos_score(
            gt["image_id"],
            knn_preds.sort_values("prompt_id")["image_id"],
            self.top_k,
        )
        print("Current Score:", score, "Best Score:", self.best_score)

        if score > self.best_score:
            self.best_score = score
            print("saving best model to data/ folder")
            torch.save(
                self.trainer.model.state_dict(), f"data/siamese_model_{score}.pth"
            )

        # Genearate new pairs
        print("Building new validation pair df")
        new_val_pairs_df = build_new_pairs_df(knn_preds, self.correlation_df)[
            ["prompt_id", "image_id", "target"]
        ]
        if score == self.best_score:  # only save for the best checkpoint
            print("saving new_val_pairs_df to data/ folder")
            new_val_pairs_df.to_csv("data/new_val_pairs_df.csv")

        # for training set
        print("Generating embedding for train topics")
        train_prompt_embs_gpu = cp.array(self.train_prompt_embs)

        train_indices = neighbors_model.kneighbors(
            train_prompt_embs_gpu, return_distance=False
        )

        train_predictions = []
        for k in tqdm(range(len(train_indices))):
            pred = train_indices[k]
            p = " ".join([image_idx_to_id[ind] for ind in pred.get()])

            train_predictions.append(p)

        train_knn_preds = pd.DataFrame(
            {
                "prompt_id": self.train_prompt_ids,
                "image_id": train_predictions,
            }
        ).sort_values("prompt_id")

        print("Building new train supervised df")
        new_train_pairs_df = build_new_pairs_df(train_knn_preds, self.correlation_df)

        if score == self.best_score:  # only save for the best checkpoint
            print("saving new_train_supervised_df to data/ folder")
            new_train_pairs_df.to_csv("data/new_train_pairs_df.csv")

        # update train_dataset and val_dataset
        print("preprocess csv for train/validation topics, contents, labels")
        self.trainer.train_dataset.pairs_df = new_train_pairs_df.dropna()
        (
            self.trainer.train_dataset.image_paths,
            self.trainer.train_dataset.prompt_embs,
            self.trainer.train_dataset.labels,
        ) = self.trainer.train_dataset.preprocess_df()

        self.trainer.eval_dataset.pairs_df = new_val_pairs_df.dropna()
        (
            self.trainer.eval_dataset.image_paths,
            self.trainer.eval_dataset.prompt_embs,
            self.trainer.eval_dataset.labels,
        ) = self.trainer.eval_dataset.preprocess_df()

        del (
            train_prompt_embs_gpu,
            train_knn_preds,
            train_indices,
            train_predictions,
        )

        del (
            prompt_embs,
            image_embs,
            knn_preds,
            indices,
            neighbors_model,
            predictions,
        )

        gc.collect()
        torch.cuda.empty_cache()

        # if hasattr(self.trainer.callback_handler.train_dataloader.sampler, "prompt_id"):
        #     prompt_ids, labels = (
        #         self.trainer.train_dataset.supervised_df["prompt_id"].values,
        #         self.trainer.train_dataset.supervised_df["target"].values,
        #     )
        #     self.trainer.callback_handler.train_dataloader.sampler.initialize(prompt_ids, labels)

    def on_train_begin(self, args, state, control, **kwargs):
        self.on_epoch_end(args, state, control, **kwargs)

    def on_epoch_end(self, args, state, control, **kwargs):
        # update dataset at the end of each epoch
        print("Updating dataset")
        self.update_dataset()


def build_new_pairs_df(knn_df, correlations):
    # Create lists for training
    prompt_ids = []
    image_ids = []

    mapping = set()
    # get all class 1 in correlations
    prompt_ids = set(knn_df.prompt_id.values)
    filtered_correlations = correlations[correlations.prompt_id.isin(prompt_ids)]
    for i, row in tqdm(filtered_correlations.iterrows()):
        if str(row["image_id"]) and str(row["image_id"]) != "nan":
            image_ids = str(row["image_id"]).split(" ")
            for image_id in image_ids:
                mapping.add((row["prompt_id"], image_id, 1))

    for i, row in tqdm(knn_df.iterrows()):
        if str(row["image_id"]) and str(row["image_id"]) != "nan":
            image_ids = str(row["image_id"]).split(" ")
            for image_id in image_ids:
                if (
                    row["prompt_id"],
                    image_id,
                    1,
                ) not in mapping:  # because mapping already contains all positive cases
                    mapping.add((row["prompt_id"], image_id, 0))

    # Build training dataset
    mapping = list(mapping)
    new_df = pd.DataFrame(
        {
            "prompt_id": [item[0] for item in mapping if item[1]],
            "image_id": [item[1] for item in mapping if item[1]],
            "target": [item[2] for item in mapping if item[1]],
        }
    )
    # Release memory
    del prompt_ids, image_ids
    gc.collect()
    return new_df
