# LPGNN
Code for the paper "Multivariate Time Series Prediction of Complex Systems Based on Graph Neural Networks with Location Embedding Graph Structure Learning"

![LPGNN](figures/model_architecture.jpg "Model Architecture")

This is a Pytorch implementation.
## Requirements
- scipy>=1.7.0
- numpy>=1.21.0
- pandas>=1.2.5
- pyaml
- statsmodels
- pytorch>=1.8.0
- networkx>=2.6.3

## Data Preparation
Our traffic flow dataset has been placed in the dataset folder of the code. You need to unzip the dataset to this folder first. 
METR-LA and PEMS-BAY source and original paper of [DCRNN](https://github.com/liyaguang/DCRNN). 
PEMSD4 and PEMSD8 come from the paper [ASTGCN](https://github.com/Davidham3/ASTGCN)

## Model Training

You need to first specify the dataset name in main.py and then directly to run
```bash
python main.py
```

## Citation
