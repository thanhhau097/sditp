import gc
from typing import Dict

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from torchvision.ops import sigmoid_focal_loss
from transformers import Trainer
from transformers.trainer_pt_utils import nested_detach

from model import Model

from typing import Iterable, Dict
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import CosineEmbeddingLoss
from enum import Enum


class SiameseDistanceMetric(Enum):
    """
    The metric for the contrastive loss
    """

    EUCLIDEAN = lambda x, y: F.pairwise_distance(x, y, p=2)
    MANHATTAN = lambda x, y: F.pairwise_distance(x, y, p=1)
    COSINE_DISTANCE = lambda x, y: 1 - F.cosine_similarity(x, y)


class OnlineContrastiveLoss(nn.Module):
    """
    Online Contrastive loss. Similar to ConstrativeLoss, but it selects hard positive (positives that are far apart)
    and hard negative pairs (negatives that are close) and computes the loss only for these pairs. Often yields
    better performances than  ConstrativeLoss.

    :param distance_metric: Function that returns a distance between two emeddings. The class SiameseDistanceMetric contains pre-defined metrices that can be used
    :param margin: Negative samples (label == 0) should have a distance of at least the margin value.
    :param size_average: Average by the size of the mini-batch.
    """

    def __init__(
        self, distance_metric=SiameseDistanceMetric.COSINE_DISTANCE, margin: float = 0.2
    ):
        super(OnlineContrastiveLoss, self).__init__()
        self.margin = margin
        self.distance_metric = distance_metric

    def forward(self, embeddings, labels, size_average=False):
        distance_matrix = self.distance_metric(embeddings[0], embeddings[1])
        negs = distance_matrix[labels == 0]
        poss = distance_matrix[labels == 1]

        # select hard positive and hard negative pairs
        negative_pairs = negs[negs < (poss.max() if len(poss) > 1 else negs.mean())]
        positive_pairs = poss[poss > (negs.min() if len(negs) > 1 else poss.mean())]

        positive_loss = positive_pairs.pow(2).sum()
        negative_loss = F.relu(self.margin - negative_pairs).pow(2).sum()
        loss = positive_loss + negative_loss
        return loss


def cos_sim(a: Tensor, b: Tensor):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


class MultipleNegativesRankingLoss(nn.Module):
    """
    This loss expects as input a batch consisting of sentence pairs (a_1, p_1), (a_2, p_2)..., (a_n, p_n)
    where we assume that (a_i, p_i) are a positive pair and (a_i, p_j) for i!=j a negative pair.
    For each a_i, it uses all other p_j as negative samples, i.e., for a_i, we have 1 positive example (p_i) and
    n-1 negative examples (p_j). It then minimizes the negative log-likehood for softmax normalized scores.
    This loss function works great to train embeddings for retrieval setups where you have positive pairs (e.g. (query, relevant_doc))
    as it will sample in each batch n-1 negative docs randomly.
    The performance usually increases with increasing batch sizes.
    For more information, see: https://arxiv.org/pdf/1705.00652.pdf
    (Efficient Natural Language Response Suggestion for Smart Reply, Section 4.4)
    You can also provide one or multiple hard negatives per anchor-positive pair by structering the data like this:
    (a_1, p_1, n_1), (a_2, p_2, n_2)
    Here, n_1 is a hard negative for (a_1, p_1). The loss will use for the pair (a_i, p_i) all p_j (j!=i) and all n_j as negatives.
    Example::
        from sentence_transformers import SentenceTransformer, losses, InputExample
        from torch.utils.data import DataLoader
        model = SentenceTransformer('distilbert-base-uncased')
        train_examples = [InputExample(texts=['Anchor 1', 'Positive 1']),
            InputExample(texts=['Anchor 2', 'Positive 2'])]
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)
        train_loss = losses.MultipleNegativesRankingLoss(model=model)
    """

    def __init__(self, scale: float = 20.0, similarity_fct=cos_sim):
        """
        :param scale: Output of similarity function is multiplied by scale value
        :param similarity_fct: similarity function between sentence embeddings. By default, cos_sim. Can also be set to dot product (and then set scale to 1)
        """
        super(MultipleNegativesRankingLoss, self).__init__()
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, reps, labels):
        embeddings_a = reps[0]
        embeddings_b = torch.cat(reps[1:])
        scores = self.similarity_fct(embeddings_a, embeddings_b) * self.scale
        labels = torch.tensor(
            range(len(scores)),
            dtype=torch.long,
            device=scores.device,
        )
        return self.cross_entropy_loss(scores, labels)

    def get_config_dict(self):
        return {"scale": self.scale, "similarity_fct": self.similarity_fct.__name__}


class CustomTrainer(Trainer):
    def __init__(self, pos_neg_ratio=1, **kwargs):
        super().__init__(**kwargs)
        self.pos_neg_ratio = pos_neg_ratio

    def compute_loss(self, model: Model, inputs: Dict, return_outputs=False):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        outputs = model(inputs["images"])

        embs = inputs.get("embs")
        labels = inputs.get("labels")
        # try:
        objective = model.module.objective
        # except:
        #     objective = model.objective
        if objective == "cosine":
            loss_fct = CosineEmbeddingLoss()
            labels = (labels * 2) - 1
            loss = loss_fct(outputs, embs, labels.float())
        elif objective == "mnrl":
            loss_fct = MultipleNegativesRankingLoss(scale=50)
            loss = loss_fct([outputs, embs], None)
        elif objective == "contrastive":
            loss_fct = OnlineContrastiveLoss()
            loss = loss_fct([outputs, embs], labels.float())
        else:
            raise ValueError("objective should be cosine/contrastive")

        if return_outputs:
            return (loss, outputs)
        return loss

    def create_optimizer(self):
        model = self.model
        no_decay = []
        for n, m in model.named_modules():
            if isinstance(
                m,
                (
                    torch.nn.BatchNorm1d,
                    torch.nn.BatchNorm2d,
                    torch.nn.LayerNorm,
                    torch.nn.LayerNorm,
                    torch.nn.GroupNorm,
                ),
            ):
                no_decay.append(n)

        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
            self.args
        )
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer

    def prediction_step(
        self, model, inputs, prediction_loss_only=False, ignore_keys=None
    ):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()

        if prediction_loss_only:
            return (loss, None, None)

        if type(outputs) == tuple:
            outputs = outputs[0]  # return only classification outputs
        outputs = outputs.float()
        outputs = nested_detach(outputs)
        # del inputs["topic_inputs"]
        # del inputs["content_inputs"]
        # del inputs["combined_inputs"]

        gc.collect()
        return loss, outputs, inputs["embs"]


def compute_metrics(eval_preds):
    # calculate cosine similarity between embeddings of predicted and actual labels
    predictions = eval_preds.predictions
    labels = eval_preds.label_ids
    predictions = np.array(predictions)
    labels = np.array(labels)
    consine_sim = np.dot(predictions, labels.T)
    consine_sim = consine_sim / np.linalg.norm(predictions, axis=1)[:, None]
    consine_sim = consine_sim / np.linalg.norm(labels, axis=1)[None, :]
    consine_sim = np.diag(consine_sim)
    consine_sim = np.mean(consine_sim)
    return {"score": consine_sim}
