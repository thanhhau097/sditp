# run generate_data.py with n processes, each with one GPU device, with multiprocessing Pool

import os
import multiprocessing
import subprocess


def run_generate_data(i, sentence_transformers_weights_path):
    num_images_per_process = 250000
    num_images_per_prompt = 1
    subprocess.run(
        f"CUDA_VISIBLE_DEVICES={i} python generate_data.py --start {i * num_images_per_process} --end {(i + 1) * num_images_per_process} --n_images_per_prompt {num_images_per_prompt} --sentence_transformers_weights_path {sentence_transformers_weights_path}",
        shell=True,
    )


if __name__ == "__main__":
    num_processes = 8
    sentence_transformers_weights_path = (
        "/home/phamhoan/sditp/data/all-MiniLM-L6-v2"
    )
    with multiprocessing.Pool(num_processes) as p:
        p.starmap(
            run_generate_data,
            [(i, sentence_transformers_weights_path) for i in range(num_processes)],
        )

# huggingface-cli login
# download sentence transformers weights here: https://www.kaggle.com/datasets/inversion/sentence-transformers-222
