#!/usr/bin/env sh

python -u main.py \
    --comment "test" \
    --device "t7-fmwk" \
    --LLM "gemini-pro" \
    --dataset "hellaswag" \
    --sample_size 100 \
    --epochs 20 \
    --mut_prompts 2 \
    --think_sty 4 \
