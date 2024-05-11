#!/bin/bash

# Lora dreambooth training with inpainting tuned SD/SDXL model

source ./venv/bin/activate

pwd

accelerate launch diffusers/examples/research_projects/dreambooth_inpaint/train_dreambooth_inpaint_lora_sdxl.py \
    --pretrained_model_name_or_path="./models/sdxl-inpainting-1.0" \
    --instance_data_dir="./dataset/your-subject-images" \
    --output_dir="./lora-weights/ms-sofa-sdxl-from-inpainting" \
    --instance_prompt="a photo of a ms sofa" \
    --mixed_precision="fp16" \
    --resolution=1024 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=4 \
    --learning_rate=1e-4 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --max_train_steps=500 \
    --seed="42" \