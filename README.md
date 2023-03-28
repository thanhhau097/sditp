# README

## NEPTUNE
```
export NEPTUNE_PROJECT="thanhhau097/sditp"
export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlMTRjM2ExOC1lYTA5LTQwODctODMxNi1jZjEzMjdlMjkxYTgifQ=="
```

Some instructions:
1. Prompts for this challenge were generated using a variety of (non disclosed) methods, and range from fairly simple to fairly complex with multiple objects and modifiers. Images were generated from the prompts using Stable Diffusion 2.0 (768-v-ema.ckpt) and were generated with 50 steps at 768x768 px and then downsized to 512x512 for the competition dataset. (This script was used, with the majority of default parameters unchanged.) References: 
    - model: https://huggingface.co/stabilityai/stable-diffusion-2/blob/main/768-v-ema.ckpt
    - script: https://github.com/CompVis/stable-diffusion/blob/main/scripts/txt2img.py. 
    - src: https://www.kaggle.com/competitions/stable-diffusion-image-to-prompts/data
2. Text embedding model: https://www.kaggle.com/datasets/inversion/sentence-transformers-222. Reference: https://www.kaggle.com/code/inversion/calculating-stable-diffusion-prompt-embeddings
3. Public data: https://huggingface.co/datasets/poloclub/diffusiondb

## Scripts
Cosine:

```
CUDA_VISIBLE_DEVICES=1 python train.py --evaluation_strategy epoch --save_strategy epoch --save_total_limit 5 --logging_strategy steps --logging_steps 50 --fp16 --warmup_ratio 0.01 --lr_scheduler_type cosine --adam_eps 1e-6 --optim adamw_torch --do_train --do_eval --metric_for_best_model eval_loss --model_name resnet50 --fold 0 --dataloader_num_workers 32 --learning_rate 1e-4 --num_train_epochs 30 --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --remove_unused_columns False --overwrite_output_dir --load_best_model_at_end --objective cosine --output_dir ./outputs_resnet50/
```

Contrastive:

```
CUDA_VISIBLE_DEVICES=1 python train.py --evaluation_strategy epoch --save_strategy epoch --save_total_limit 5 --logging_strategy steps --logging_steps 20 --fp16 --warmup_ratio 0.01 --lr_scheduler_type cosine --adam_eps 1e-6 --optim adamw_torch --do_train --do_eval --metric_for_best_model eval_loss --model_name resnet18 --fold 0 --dataloader_num_workers 32 --learning_rate 5e-4  --num_train_epochs 30 --per_device_train_batch_size 128 --per_device_eval_batch_size 128 --remove_unused_columns False --load_best_model_at_end --objective contrastive --output_dir ./outputs_resnet18  --gradient_accumulation_steps 1 --report_to none --pairs_path ./data/generated_data/pairs_2000.csv --prompt_path ./data/generated_data/prompt_2000.csv --correlation_path ./data/generated_data/correlation_2000.csv --image_path ./data generated_data/image_2000.csv
```

## TODO
- [ ] Pretrain with public dataset - multi negative ranking loss
- [ ] Generate images from original model, using text from public dataset
- [ ] Generate embeddings from those texts and create mapping image-text-emb: https://www.kaggle.com/code/inversion/calculating-stable-diffusion-prompt-embeddings
- [ ] Train image to embedding models
- [ ] Use mean and std: OPENAI_CLIP_MEAN

## IDEA:
- [ ] Generate to text then compare two texts as in LECR
- [ ] Use openclip model -> get embeddings -> add more layers to train on this embeddings. => faster


## Note:
1. One prompt can generate multiple images depends on the configs we use. Thoughts: In this case, we can consider `prompt` as topic and generated images as `contents` as in LECR competition.
