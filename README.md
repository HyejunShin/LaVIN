# Hyperparameter tuning on LaVIN, an Efficient Vision-Language Instruction Tuning for Large Language Models

> CSCI-GA 3033-091 IDLS Fall 2023 Final Project

[LaVIN](https://arxiv.org/pdf/2305.15023.pdf) is a multimodal model derived from LLaMA (Large Language Model) and powered by the novel [Mixture-of-Modality Adaptation](https://arxiv.org/pdf/2302.08106.pdf) (MMA). Developed by [luogen1996](https://github.com/luogen1996) to extend the capabilities of large language models.
Our foal is to extend the capabilities of LLMs for multimodal tasks by efficiently fine-tuning & modifying LaVIN 7B. To achieve this, we proposed, implemented, and evaluated differents modifications of LaVIN.

## Project Milestones
- [x] Train (original) LaVIN 7B and validate accuracy
- [x] Train LaVIN 7B using Cyclical Learning Rate and validate accuracy
- [x] Train LaVIN 7B using Step Decay (Learning Rate) and validate accuracy
- [x] Train LaVIN 7B using Exponential Decay (Learning Rate) and validate accuracy
- [x] Train LaVIN 7B having the dropout probability (in the MMA adapter) increase after each layer
- [x] Train LaVIN 7B having the dropout probability (in the MMA adapter) decrease after each layer
- [x] Elaborate comparative graphs
- [x] Validate each "modification" using a live chatbot (on text questions and images)


## Downloads
- Install LaVIN from [official repo](https://github.com/luogen1996/LaVIN)
- Download ScienceQA dataset from [official repo](https://github.com/lupantech/ScienceQA)
- Download [LLaMA-7B](https://huggingface.co/nyanko7/LLaMA-7B/tree/main) from HuggingFace


## Repository and Code Structure
```bash
LaVIN/
  |-- lavin
  |-- scripts
  |-- train.py
  |-- eval.py
  ......
data/
  |-- problem.json
  |-- pid_splits.json
  |-- captions.json
  |-- images
      |-- train          # ScienceQA train image
      |-- val            # ScienceQA val image
      |-- test           # ScienceQA test image
  |-- weights
      |-- tokenizer.model
      |--7B
          |-- params.json
          |-- consolidated.00.pth
```


## Commands to Execute the Fine-tunning of the Model
To run the the basic (original) finetuning of LaVIN-7B:
```bash
bash ./scripts/finetuning_sqa_7b.sh
```

which contains:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 --master_port 11111 train.py \
    --llm_model 7B\
    --llama_model_path ../data/weights/ \
    --data_path ../data/alpaca_data.json \
    --max_seq_len 512 \
    --batch_size 2 \
    --accum_iter 4 \
    --epochs 20 \
    --warmup_epochs 2 \
    --blr 9e-3 \
    --weight_decay 0.02 \
    --output_dir ./LaVIN-7B/\
    --adapter_type attn\
    --adapter_dim 8\
    --adapter_scale 1\
    --n_prompt 6 \
    --prompt_format QCM-ALE \
    --temperature 10.\
    --visual_adapter_type router

CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node 1 --master_port 11111 eval.py \
    --ckpt_dir ../data/weights/ \
    --llm_model 7B\
    --tokenizer_path ../data/weights/tokenizer.model \
    --data_root ../data \
    --caption_file ../data/captions.json \
    --adapter_path ./LaVIN-7B/checkpoint-19.pth \
    --adapter_type attn \
    --adapter_dim 8 \
    --adapter_scale 1 \
    --prompt_format QCM-ALE \
    --max_batch_size 64\
    --max_seq_len 512 \
    --split test \
    --n_prompt 6 \
    --temperature 10.\
    --visual_adapter_type router
```

### Other commands
We provided with different bash files to finetune LaVIN using the different [modifications]() we proposed, this files are located in `./scripts/1`. To finetune LaVIN-7B using different learning reates schedulers, you can check `./utils/lr_sched.py`. 

## Modifications/Variations
We proposed and implemented the following modifications to LaVIN:

- Learning Rate Scheduler modifications:
  - Cyclical Learning Rate 
  - Step Decay
  - Exponential Decay
- Adapter's modifications:
  - Increasing dropout probability (in the MMA adapter) after each layer
  - Decreasing dropout probability (in the MMA adapter) after each layer
 
### Learning Rate Scheduler modifications
The idea of implementing different learning rates arises from the classic problem associated with neural networks: hyperparameter fine-tuning. By default, LaVIN implements XXX. On our part, we have implemented the following well-known algorithms: Cyclical Learning Rate, Step Decay, and Exponential Decay to experiment and evaluate the results of using these learning rates schedulers instead of the one used by the original implementation.

### Adapter's modifications
We implemented a slight modification to the MMA adapter used in LaVIN. This modification consists in continuously modifying the dropout probability of the adapter’s nodes after every layer, until it reaches a certain limit. To achieve this, we implemented teh following new hyper-parameters:
- dropout_prob : the initial dropout probability (range 0-1)
- dropout_lim : the limit set for the dropout probability. Once this limit is reached, the dropout probability is no longer modified.
- dropout_var : the dropout variability factor. This factor controls how the dropout probability is changed after every layer. Is described as:

When increasing the dropout probability after every layer, dropout_lim represents the "ceiling limit" and dropout_var > 1. More formally:

When decresing the dropout probability after every layer, dropout_lim represents the "floor limit" and dropout_var < 1. More formally:


## Experiment and Results

### Experiment environment
Each experiment was run using 4xV100 GPUs on Greene NYU HPC.

> Note: Fine-tuning LaVIN-7B requires at least 33GB per GPU when using batch-size=4. Since we worked with V100 GPUs, we used batch-size=2

### Results Table
Observation, we ran two experiments for each Increasing Dropout Probability and Decreasing Dropout Probability after every layer; each with different hyperparameters values. For Increasing prob. we used {dropout_lim=0.2, dropout_prob=0.1, dropout_var=1.01} and {dropout_lim=0.2, dropout_prob=0.1, dropout_var=1.03}. The later is refered as V2. For Decreasing prob. we used {dropout_lim=0.1, dropout_prob=0.2, dropout_var=0.99} and dropout_lim=0.1, dropout_prob=0.2, dropout_var=0.97}. The later is refered as V2. 

| Method             |   NAT |   SOC |   LAN |   TXT |   IMG |    NO |   G1-6 |   G7-12 |   Average |
|:-------------------|------:|------:|------:|------:|------:|------:|-------:|--------:|----------:|
| Base               | 88.45 | 94.71 | 84.82 | 87.59 | 86.37 | 87.8  |  90.2  |   86.35 |     88.82 |
| Cyclical LR        | 85.48 | 92.58 | 82.18 | 84.51 | 83.44 | 84.88 |  87.15 |   84.25 |     86.11 |
| Step Decay         | 86.23 | 94.15 | 83.27 | 85.34 | 85.42 | 86.48 |  88.36 |   84.9  |     87.13 |
| Exp. Decay         | 85.52 | 93.59 | 81.45 | 84.51 | 83.99 | 84.88 |  87.67 |   83.45 |     86.16 |
| Inc. Drop. Prob    | 86.5  | 93.03 | 83.82 | 85.78 | 84.18 | 86.69 |  88.62 |   84.57 |     87.17 |
| Dec. Drop. Prob    | 87.21 | 94.6  | 82.82 | 86.22 | 86.02 | 86.27 |  89.35 |   84.51 |     87.62 |
| Inc. Drop. Prob V2 | 86.63 | 94.15 | 83.73 | 85.63 | 85.23 | 87.25 |  88.58 |   85.43 |     87.46 |
| Dec. Drop. Prob V2 | 87.57 | 94.15 | 83.73 | 86.51 | 85.37 | 87.11 |  89.35 |   85.43 |     87.95 |

### Graphs
![](./assets/download.png)
![](./assets/download-1.png)
![](./assets/download-2.png)
![](./assets/download-3.png)
![](./assets/download-4.png)
![](./assets/download-5.png)
![](./assets/download-6.png)
![](./assets/download-7.png)
![](./assets/download-8.png)
![](./assets/download-9.png)


## Demo
[View demo here](https://drive.google.com/file/d/10HibI0_xwqL7hfBdDYaVGuj3yOTuCpPS/view?usp=sharing)

## Observations and Conclusion
- None of the adjustments surpassed the baseline model's performance.
- This suggests that LaVIN 7B is already highly optimized for the multimodal tasks.
- Out of the modified models, Dec. Drop. Prob V2 performed the best. 
- Future work may explore combining these modifications or experimenting with other hyperparameters to potentially exceed the baseline model's performance.


## References
1) https://github.com/luogen1996/LaVIN
2) https://github.com/lupantech/ScienceQA
3) https://huggingface.co/nyanko7/LLaMA-7B/tree/main

