import os
import random
import uuid

from sentence_transformers import SentenceTransformer, models
from diffusers import StableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler                                                                                                                                                                                             
import torch
from huggingface_hub import login
import pandas as pd
from sklearn.model_selection import KFold
from urllib.request import urlretrieve
import pandas as pd
from tqdm import tqdm


# login()


def generate(args):
    st_model = SentenceTransformer(args.sentence_transformers_weights_path)

    # Download the parquet table
    print("Downloading metadata.parquet...")
    table_url = f"https://huggingface.co/datasets/poloclub/diffusiondb/resolve/main/metadata-large.parquet"
    urlretrieve(table_url, "metadata.parquet")
    metadata_df = pd.read_parquet("metadata.parquet")

    prompts = list(metadata_df.prompt.values)[args.start : args.end]

    print("Encoding prompts...")
    embeddings = []
    for prompt in tqdm(prompts):
        prompt_embeddings = st_model.encode(prompt).flatten()
        embeddings.append(prompt_embeddings)

    class CFG:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        seed = 43
        generator = torch.Generator(device).manual_seed(seed)
        model_id = "stabilityai/stable-diffusion-2"

    cfg = CFG()

    model_pipe = StableDiffusionPipeline.from_pretrained(
        cfg.model_id, torch_dtype=torch.float16, revision="fp16", use_auth_token=True
    )
    model_pipe = model_pipe.to(cfg.device)                                                                                                                                                                                                                                              
    model_pipe.scheduler = DPMSolverMultistepScheduler.from_config(model_pipe.scheduler.config)                                                                                                                                                               

    def generate_image(
        prompt, idx, model, n_images=1, image_folder=args.save_image_dir
    ):
        image_paths = []
        for i in range(n_images):
            image_path = os.path.join(
                image_folder, f"{str(idx).zfill(8)}_{str(i).zfill(2)}.jpg"
            )
            if not os.path.exists(image_path):
                image = model(prompt, num_inference_steps=20).images[0]
                # save image to image_folder
                image.save(image_path)
            image_paths.append(image_path)

        return image_paths

    generated_image_paths = []
    images_idx = []
    for i, prompt in enumerate(prompts):
        idx = i + args.start
        generated_image_paths.append(
            generate_image(prompt, idx, model_pipe, args.n_images_per_prompt)
        )
        images_idx.append(idx)

    prompt_ids = []
    image_ids = []
    image_paths = []
    all_prompt_images = []

    for prompt, images in zip(prompts, generated_image_paths):
        prompt_id = str(uuid.uuid4())
        prompt_ids.append(prompt_id)

        prompt_images = []
        for image in images:
            image_id = os.path.basename(image).split(".")[0]
            image_ids.append(image_id)
            image_paths.append(image)
            prompt_images.append(image_id)

        all_prompt_images.append(prompt_images)

    prompt_df = pd.DataFrame({"id": prompt_ids, "text": prompts, "emb": embeddings})

    image_df = pd.DataFrame({"id": image_ids, "path": image_paths})

    correlation_df = pd.DataFrame(
        {
            "prompt_id": prompt_ids,
            "image_id": [" ".join(images) for images in all_prompt_images],
        }
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    correlation_df["fold"] = -1
    for fold, (train_idx, val_idx) in enumerate(kf.split(correlation_df)):
        correlation_df.loc[val_idx, "fold"] = fold

    # write to csv
    prompt_df.to_csv(
        os.path.join(args.save_dir, f"prompt_{args.start}.csv"), index=False
    )
    image_df.to_csv(os.path.join(args.save_dir, f"image_{args.start}.csv"), index=False)
    correlation_df.to_csv(
        os.path.join(args.save_dir, f"correlation_{args.start}.csv"), index=False
    )

    pairs = []
    for prompt_id in correlation_df.prompt_id.unique():
        prompt_images = (
            correlation_df[correlation_df.prompt_id == prompt_id]
            .image_id.values[0]
            .split(" ")
        )
        for image_id in prompt_images:
            pairs.append((prompt_id, image_id, 1))

        # each value is a string, we need to split it to get the list of image_id
        negative_images = correlation_df[
            correlation_df.prompt_id != prompt_id
        ].image_id.values
        negative_images = [image_id.split(" ") for image_id in negative_images]
        negative_images = [
            image_id for image_ids in negative_images for image_id in image_ids
        ]

        pairs.extend(
            [(prompt_id, random.choice(negative_images), 0) for _ in range(10)]
        )

    pairs_df = pd.DataFrame(pairs, columns=["prompt_id", "image_id", "target"])
    pairs_df.to_csv(os.path.join(args.save_dir, f"pairs_{args.start}.csv"), index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="./generated_data")
    parser.add_argument(
        "--sentence_transformers_weights_path",
        type=str,
        default="/home/thanh/shared_disk/thanh/sditp/data/all-MiniLM-L6-v2",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=10)
    parser.add_argument("--n_images_per_prompt", type=int, default=4)

    args = parser.parse_args()

    save_image_dir = os.path.join(args.save_dir, "images")
    args.save_image_dir = save_image_dir
    if not os.path.exists(save_image_dir):
        os.makedirs(save_image_dir)

    generate(args)

    # download sentence transformers weights here: https://www.kaggle.com/datasets/inversion/sentence-transformers-222
    # kaggle datasets download -d inversion/sentence-transformers-222
    # sample usage
    # python generate_data.py --save_dir ./generated_data --start 0 --end 3000 --n_images_per_prompt 1 --sentence_transformers_weights_path /home/thanh/sditp/data/all-MiniLM-L6-v2
