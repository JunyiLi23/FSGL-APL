# Adaptive Pseudo-Labeling with Large Language Models for Graph Learning
## Abstract
Graph Neural Networks (GNNs) have excellent processing capabilities for unstructured data and are widely used in multiple fields. However, their performance is limited in real scenarios where obtaining sufficient labeled data is not easy. Pseudo-Labeling methods can alleviate the label sparsity issue, but they are also constrained by the performance of the base model, threshold settings, and the impact of data noise. To tackle aforementioned problems, we propose an Adaptive Pseudo-Labeling approach with Large Language Models for Graph Semi-Supervised Learning (GAPL-LLM). Inspired by the rapidly developing Large Language Models (LLMs) in recent years, which possess stable cross-domain knowledge and powerful few-shot learning capabilities, GAPL-LLM generates pseudo-labels with high quality by LLMs, incorporating phases containing prompting with class prototype and fine-tuning with low-rank adaptation. To avoid the error accumulation issues caused by fixed thresholds, we introduce the label quality screening, incorporating the pre-sampling with active learning and the post-filtering with dynamic threshold. We design a stagewise training strategy with consistency regularization for mitigating the impact of noise and model adaptation bias. Extensive experiments demonstrate that the proposed algorithm is effective and efficient in label sparse scenarios, and outperforms existing advanced methods.

## Environment Setups
```
pip install -r requirements.txt
```
```
pip3 install torch torchvision torchaudio
```

## Datasets

## How to use this repo and run the code

## Notes
I'll optimize the code structure when I have more time ⏳.
