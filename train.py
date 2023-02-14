import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
import transformers
from joblib import Parallel, delayed
from transformers import HfArgumentParser, TrainingArguments, set_seed
from transformers.trainer_utils import get_last_checkpoint, is_main_process

from data_args import DataArguments
from dataset import DatasetUpdateCallback, SDITPDataset, collate_fn, init_tokenizer
from engine import CustomTrainer, compute_metrics
from model import Model
from model_args import ModelArguments
from utils import get_processed_text_dict


torch.set_float32_matmul_precision("high")
logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((DataArguments, ModelArguments, TrainingArguments))
    data_args, model_args, training_args = parser.parse_args_into_dataclasses()

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this"
                " behavior, change the `--output_dir` or add `--overwrite_output_dir` to train"
                " from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(
        logging.INFO if is_main_process(training_args.local_rank) else logging.WARN
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        # transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()

    # Set seed before initializing model.
    set_seed(training_args.seed)
    fold = data_args.fold

    print("Reading prompt data...")
    prompt_df = pd.read_csv(data_args.prompt_path)
    print("Reading image data...")
    image_df = pd.read_csv(data_args.image_path)
    print("Reading correlation data...")
    correlation_df = pd.read_csv(data_args.correlation_path)

    # split train and val prompt from correlation_df
    train_prompt_ids = correlation_df[correlation_df["fold"] != fold]["prompt_id"].unique()
    val_prompt_ids = correlation_df[correlation_df["fold"] == fold]["prompt_id"].unique()

    train_dataset = SDITPDataset(
        prompt_df=prompt_df,
        image_df=image_df,
        correlation_df=correlation_df,
        image_folder=data_args.image_folder,
        prompt_ids=train_prompt_ids,
    )

    val_dataset = SDITPDataset(
        prompt_df=prompt_df,
        image_df=image_df,
        correlation_df=correlation_df,
        image_folder=data_args.image_folder,
        prompt_ids=val_prompt_ids,
    )

    # Initialize trainer
    print("Initializing model...")
    model = Model(
        model_name=model_args.model_name,
        objective=model_args.objective,
    )
    if last_checkpoint is None and model_args.resume is not None:
        logger.info(f"Loading {model_args.resume} ...")
        checkpoint = torch.load(model_args.resume, "cpu")
        if "state_dict" in checkpoint:
            checkpoint = checkpoint.pop("state_dict")
        checkpoint = {k[6:]: v for k, v in checkpoint.items()}
        model.model.load_state_dict(checkpoint)

        if "fc.weight" in checkpoint:
            model.fc.load_state_dict(
                {"weight": checkpoint["fc.weight"], "bias": checkpoint["fc.bias"]}
            )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    print("Start training...")
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )

    # if model_args.objective == "siamese":
    #     callback = DatasetUpdateCallback(
    #         trainer=trainer,
    #         train_topic_ids=train_topic_ids,
    #         val_topic_ids=val_topic_ids,
    #         topic_df=topic_df,
    #         content_df=content_df,
    #         topic_dict=topic_dict,
    #         content_dict=content_dict,
    #         correlation_df=correlation_df,
    #         tokenizer_name=model_args.tokenizer_name,
    #         max_len=data_args.max_len,
    #         best_score=0,
    #         top_k=data_args.top_k_neighbors,
    #         use_translated=data_args.use_translated,
    #         mix_translated=data_args.mix_translated,
    #     )
    #     trainer.add_callback(callback)

    # Training
    if training_args.do_train:
        checkpoint = last_checkpoint if last_checkpoint else None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

if __name__ == "__main__":
    main()