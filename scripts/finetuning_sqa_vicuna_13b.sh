torchrun --nproc_per_node 8 --master_port 12345 --nproc_per_node 8 train.py \
    --llm_model 13B\
    --llama_model_path ../data/weights/ \
    --data_path ../data/alpaca_data.json \
    --max_seq_len 512 \
    --batch_size 4 \
    --accum_iter 1 \
    --epochs 20 \
    --warmup_epochs 2 \
    --blr 9e-3 \
    --weight_decay 0.02 \
    --output_dir ./LaVIN-Vicuna-13B/\
    --adapter_type attn\
    --adapter_dim 8\
    --adapter_scale 1\
    --n_prompt 6 \
    --prompt_format QCM-ALE \
    --temperature 5.\
    --visual_adapter_type router \
    --use_vicuna

torchrun --nproc_per_node 1  eval.py \
    --ckpt_dir ../data/weights/ \
    --llm_model 13B\
    --tokenizer_path ../data/weights/tokenizer.model \
    --data_root ../data \
    --caption_file ../data/captions.json \
    --adapter_path ./LaVIN-Vicuna-13B/checkpoint-19.pth \
    --adapter_type attn \
    --adapter_dim 8 \
    --adapter_scale 1 \
    --prompt_format QCM-ALE \
    --max_batch_size 64\
    --max_seq_len 512 \
    --split test \
    --n_prompt 6 \
    --temperature 5.\
    --visual_adapter_type router \
    --use_vicuna=True