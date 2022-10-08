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
The traffic flow dataset has been placed in the dataset folder of the code. You need to unzip the dataset to this folder first. 
METR-LA and PEMS-BAY source and original paper of [DCRNN](https://github.com/liyaguang/DCRNN). 
PEMSD4 and PEMSD8 come from the paper [ASTGCN](https://github.com/Davidham3/ASTGCN).

## Model Training

You need to first specify the dataset name in main.py and then directly to run
```bash
python main.py
```
Please wait patiently for the program to finish running.

## Baseline models hyper-parameter setting
Explanation of hyper-parameter settings for some baseline models:
1)	[GraphWaveNet](https://github.com/SGT-LIM/GraphWavenet): GraphWaveNet is the first to use the combination of Temporal Convolutional Networks (TCN) and Graph Convolutional Networks (GCN) for spatio-temporal modeling. The model has several more important parameters. First of all, GraphWaveNet is not a model designed in the absence of graph structure information. However, we found in their original code that the model could be trained using only the learned adaptive adjacency matrix via hyper-parameter settings. Therefore, we set the model to use only the adaptive adjacency matrix. In addition, after parameter adjustment, we set the number of channels of the dilated convolution to 64. The batch-size is set to 64. Other parameters, such as learning rate, dropout, and weight decay, remain at their default values.
2)	[StemGNN](https://github.com/microsoft/StemGNN): StemGNN performs feature extraction in the spectral domain of the graph signal. Although the capability of feature extraction can theoretically be increased, it will increase the computational complexity of the model. We have performed a lot of hyper-parameter tuning on this model, but the results are far behind other models. The model parameters after our adjustment are as follows: StemGNN block is set to 3, the dropout rate is set to 0.5, the Leaky-Relu rate is set to 0.2 and the batch-size is set to 64.
3)	[AGCRN](https://github.com/LeiBAI/AGCRN): AGCRN is a spatiotemporal modeling method that combines recurrent neural network (RNN) and GCN. The model takes into account the unique dynamic patterns of each node, assigning each node an independent parameter matrix. The model achieved good prediction results on our dataset. The main hyper-parameter settings are as follows: The embedding dimension of the recurrent network is 64, the recurrent unit is 64, the number of network layers is 3, and the Chebyshev order is 2. The batch-size is set to 64. Other parameters remain at their default values.
4)	[MTGNN](https://github.com/nnzhan/MTGNN): MTGNN is modeled using a combination of TCN and GCN. The model has many hyper-parameter settings. We set the GCN convolutional layers for each spatiotemporal module to 2 and the subgraph size to 10. The number of convolution channels and residual connection channels is set to 32, the skip connection channel is 64, and the output channel is 128. The default value of the number of spatiotemporal feature extraction modules is optimal, so keep it at 4. The batch-size is set to 64. No other hyper-parameters were adjusted.
5)	[GTS](https://github.com/chaoshangcs/GTS): GTS is mainly aimed at the learning problem of graph structure. The model combines neural relational reasoning (NRI) with diffuse graph convolution method (DCRNN) to improve the prediction accuracy on the task of traffic flow prediction. However, GTS may have training stability issues on our dataset, so we preserve the optimal results during training. The hyper-parameters are also adjusted to ensure that the results are optimal., The hyper-parameters are configured as follows: The maximum diffusion step size of graph diffusion convolution is set to 2, the number of RNN layers is set to 2, and the number of RNN units is set to 64. We use curriculum learning for training. For the predefined K-nearest neighbor (KNN) graph, we set the number of node neighbors to 3. Since the model of learning graph structure in GTS occupies more GPU memory, in order to minimize the modification of the original code, we do not use the multi-GPU training method, so set the batch-size to 12.


## Result
Note: The Model is not designed for traffic flow prediction, but performs well on the traffic flow dataset.
Baseline code: [STGCN](https://github.com/VeritasYin/STGCN_IJCAI-18),
[DCRNN](https://github.com/liyaguang/DCRNN),
[MTGNN](https://github.com/nnzhan/MTGNN),
[GMAN](https://github.com/zhengchuanpan/GMAN),
[STGNN](https://github.com/LMissher/STGNN),
[GTS](https://github.com/chaoshangcs/GTS),
[Ada-STNet](https://github.com/LiuZH-19/Ada-STNet),
[STFGNN](https://github.com/MengzhangLI/STFGNN),
[GraphWaveNet](https://github.com/SGT-LIM/GraphWavenet),
[ASTGCN](https://github.com/Davidham3/ASTGCN),
[STSGCN](https://github.com/Davidham3/STSGCN),
[AGCRN](https://github.com/LeiBAI/AGCRN),
[Z-GCNETs](https://github.com/Z-GCNETs/Z-GCNETs),
[DSTAGNN](https://github.com/SYLan2019/DSTAGNN),
[STG-NCDE](https://github.com/jeongwhanchoi/STG-NCDE).

| **Dataset**        | **METR-LA** |       |       | **PEMS-BAY** |       |       |
|--------------------|-------------|-------|-------|--------------|-------|-------|
| **Baseline**       | MAE         |MAPE(%)| RMSE  | MAE          |MAPE(%)| RMSE  |
| **STGCN**          | 4.45        | 11.8  | 8.41  | 2.49         | 5.69  | 5.79  |
| **DCRNN**          | 3.6         | 10.5  | 7.59  | 2.07         | 4.74  | 4.9   |
| **MTGNN**          | 3.49        | 9.87  | 7.23  | 1.94         | 4.53  | 4.49  |
| **GMAN**           | 3.48        | 10.1  | 7.3   | 1.86         | 4.32  | 4.31  |
| **STGNN**          | 3.49        | 9.69  | 6.94  | 1.83         | 4.15  | 4.2   |
| **GTS**            | 3.41        | 9.9   | 6.74  | 1.91         | 4.4   | 3.97  |
| **Ada-STNet**      | 3.47        | 9.8   | 7.18  | 1.89         | 4.5   | 4.36  |
| **STFGNN(SOTA)**   | 3.18        | 8.81  | 6.4   | 1.66         | 3.77  | 3.74  |
| **ours**           | 3.16        | 8.83  | 6.38  | 1.64         | 3.68  | 3.72  |
|                    |             |       |       |              |       |       |
| **Dataset**        | **PEMSD4**  |       |       | **PEMSD8**   |       |       |
| **Baseline**       | MAE         |MAPE(%)| RMSE  | MAE          |MAPE(%)| RMSE  |
| **STGCN**          | 21.16       | 13.83 | 35.69 | 17.5         | 11.29 | 27.09 |
| **DCRNN**          | 21.22       | 14.17 | 37.23 | 16.82        | 10.92 | 26.36 |
| **GraphWaveNet**   | 28.15       | 18.52 | 39.88 | 20.3         | 13.84 | 30.82 |
| **ASTGCN**         | 22.93       | 16.56 | 34.33 | 18.25        | 11.64 | 28.06 |
| **MSTGCN**         | 23.96       | 14.33 | 37.21 | 19           | 12.38 | 29.15 |
| **STSGCN**         | 21.19       | 13.9  | 33.69 | 17.13        | 10.96 | 26.86 |
| **STFGNN**         | 19.83       | 13.02 | 31.88 | 16.64        | 10.6  | 26.22 |
| **AGCRN**          | 19.83       | 12.97 | 32.3  | 15.95        | 10.09 | 25.22 |
| **Z-GCNETs**       | 19.5        | 12.78 | 31.61 | 15.76        | 10.01 | 25.11 |
| **DSTAGNN**        | 19.3        | 12.7  | 31.46 | 15.67        | 9.94  | 24.77 |
| **STG-NCDE(SOTA)** | 19.21       | 12.76 | 31.09 | 15.45        | 9.92  | 24.81 |
| **ours**           | 19.15       | 12.46 | 31.15 | 15.44        | 9.54  | 24.56 |




## Citation
Please do not repurpose our code until our paper is accepted.
