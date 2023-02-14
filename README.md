# README

Some instructions:
1. Prompts for this challenge were generated using a variety of (non disclosed) methods, and range from fairly simple to fairly complex with multiple objects and modifiers. Images were generated from the prompts using Stable Diffusion 2.0 (768-v-ema.ckpt) and were generated with 50 steps at 768x768 px and then downsized to 512x512 for the competition dataset. (This script was used, with the majority of default parameters unchanged.) References: 
    - model: https://huggingface.co/stabilityai/stable-diffusion-2/blob/main/768-v-ema.ckpt
    - script: https://github.com/CompVis/stable-diffusion/blob/main/scripts/txt2img.py. 
    - src: https://www.kaggle.com/competitions/stable-diffusion-image-to-prompts/data
2. Text embedding model: https://www.kaggle.com/datasets/inversion/sentence-transformers-222. Reference: https://www.kaggle.com/code/inversion/calculating-stable-diffusion-prompt-embeddings
3. Public data: https://huggingface.co/datasets/poloclub/diffusiondb

## TODO
- [ ] Pretrain with public dataset
- [ ] Generate images from original model, using text from public dataset
- [ ] Generate embeddings from those texts and create mapping image-text-emb: https://www.kaggle.com/code/inversion/calculating-stable-diffusion-prompt-embeddings
- [ ] Train image to embedding models


## Note:
1. One prompt can generate multiple images depends on the configs we use. Thoughts: In this case, we can consider `prompt` as topic and generated images as `contents` as in LECR competition.
