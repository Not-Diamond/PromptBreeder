#!/usr/bin/env sh

python -u test.py \
    --comment "test" \
    --device "AWS-A10G-Train" \
    --writerLLM "checkpoints/hashed/tlsh/gpt-3.5-turbo/roberta-base/dpo_model/checkpoint-10000" \
    --LLM "gpt-3.5-turbo" \
    --sample_size 100 \
    --dataset "squadv2"