# CSCI-GA 3033-091 IDLS Fall 2023 Project Proposal

> Hyejun Shin, Lucas Martinez

> November 2023

**Project Title:**

Efficient Vision-Language Instruction Tuning for Large Language Models using LaVIN

**Team members:**

Hyejun Shin (hs4543), Lucas Martinez (lom2017)

**Goal/Objective:**

The goal of this project is to extend the capabilities of Large Language Models (LLMs) for multimodal tasks by efficiently fine-tuning them using a novel approach called LaVIN. The objective is to enhance the performance and training efficiency on multimodal tasks, such as science question answering and dialogue systems, without incurring the substantial computational and financial costs associated with traditional pre-training methods.

**Challenges:**

- Integrating vision and language models without increasing parameter count substantially.
- Achieving high performance on multimodal tasks.
- Ensuring quick adaptation to different modalities with limited computational resources.
- Managing the effective transfer of knowledge from text-only to text-image datasets without loss of understanding.

**Approach/Techniques:**

The approach leverages the Mixture-of-Modality Adaptation (MMA) technique to fine-tune LLMs for multimodal understanding. This involves using lightweight adapters for bridging vision and language models and employing a routing scheme for dynamic modality adaptation.
Implementation details

**Hardware:**

- Similar performance as 8 A100 GPUs (on HPC NYU Greene)

**Software:**

- LaVIN 7B and 13B
- LLaMA 7B and 13B
- CLIP-ViT

**Dataset:**

- ScienceQA
- MultiModal ChatBot

**Demo planned**

- A demonstration showcasing the LaVIN model’s ability to answer science-related questions accurately using both text and images.
- A live multimodal chatbot session displaying the model's proficiency in coding, mathematics, and image captioning.

**References**

- Original Paper URL: https://openreview.net/forum?id=t877958UGZ
- LaVIN Code Repository: https://github.com/luogen1996/LaVIN
- Additional references from the literature on LLMs, VL pre-training, and existing multimodal solutions that the paper cites and builds upon.

