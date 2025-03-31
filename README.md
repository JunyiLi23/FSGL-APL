# Adaptive Pseudo-Labeling with Large Language Models for Sparse Label Graphs
## Abstract
Graphs excel in modeling complex topological relationships and find extensive applications in multiple domains. However, graph-oriented models are limited in real scenarios with label sparsity issue. Pseudo-Labeling methods alleviate it through the trained model on labeled nodes, but also constrained by the base model, threshold settings, and data noise. To tackle aforementioned problems, we propose an Adaptive Pseudo-Labeling approach with Large Language Models for Graph Learning (APL-LLM). Leveraging the powerful few-shot learning capabilities of LLMs, APL-LLM adaptive prompts LLMs to generate high quality pseudo-labels, with class prototypes to gain reliable semantic guidance. To avoid the error accumulation issues caused by fixed thresholds, we introduce the label quality screening, incorporating the pre-sampling with active learning and the post-filtering with dynamic threshold. We further design a stagewise training strategy for mitigating the impact of noise and model adaptation bias. Extensive experiments demonstrate that the proposed algorithm is effective and efficient in label sparse scenarios, and outperforms existing advanced methods.

## Environment Setups
```
pip install -r requirements.txt
```
```
pip3 install torch torchvision torchaudio
```
and install torch-geometric, faiss
## About the datasets
1. TAG version
2. put files into `xxx/FSGL-APL/data`
3. Set the corresponding path in `config.yaml`
## How to use this repo and run the code
Run the code `python src/test111.py`

Detailed examples:

Run the following code `python src/main.py --dataset cora --model_name GCN --data_format sbert --main_seed_num 5 --split active --output_intermediate 0 --no_val 1 --strategy pagerank2 --debug 1 --total_budget 140 --filter_strategy consistency --loss_type ce --second_filter conf+entropy --epochs 22 --debug_gt_label 0 --early_stop_start 150 --filter_all_wrong_labels 0 --oracle 1 --ratio 0.29 --alpha 0.33 --beta 0.33` for Cora.

Please run the following code `python src/main.py --dataset pubmed --model_name GCN --data_format sbert --main_seed_num 3 --split active --output_intermediate 0 --no_val 1 --strategy pagerank2 --debug 1 --total_budget 120 --filter_strategy consistency --loss_type ce --second_filter conf+entropy --epochs 30 --debug_gt_label 0 --early_stop_start 150 --filter_all_wrong_labels 0 --oracle 1 --ratio 0.2 --alpha 0.33 --beta 0.33` for PubMed.
## Notes
I'll optimize the code structure when I have more time ⏳.
