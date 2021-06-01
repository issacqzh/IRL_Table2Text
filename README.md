# How Helpful is Inverse Reinforcement Learning for Table-to-Text Generation?
This repository contains the source code and dataset used in the ACL 2021 paper "How Helpful is Inverse Reinforcement Learning for Table-to-Text Generation?"

## Abstract
Existing approaches for the Table-to-Text task suffer from issues such as missing information, hallucination and repetition. Many approaches to this problem use Reinforcement Learning (RL), which maximizes a single manually defined reward, such as BLEU. In this work, we instead pose the Table-to-Text task as Inverse Reinforcement Learning (IRL) problem. We explore using multiple interpretable unsupervised reward components that are combined linearly to form the composite reward function. The reward function and the description generator are jointly learned. We find that IRL outperforms strong RL baselines marginally. We further study the generalization of learned IRL rewards in scenarios involving domain adaptation. Our experiments reveal significant challenges in using IRL for this task.

## Requirements
**Environment**
```
pip install -r requirements.txt
```

**Dataset**
<br>
The split datasets are included in the [data](data) folder.

## Quick Start
**Training**
```
python main.py --cuda --mode 0
```
**Testing**
```
python main.py --cuda --mode 1
```
