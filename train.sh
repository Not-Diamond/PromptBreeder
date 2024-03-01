#!/usr/bin/env sh

python -u main.py \
    --comment "test" \
    --device "AWS-Eval" \
    --writerLLM "gpt-3.5-turbo" \
    --LLM "gemini-pro" \
    --sample_size 100 \
    --epochs 20 \
    --mut_prompts 2 \
    --think_sty 4 \
    # --dataset "hellaswag" \
