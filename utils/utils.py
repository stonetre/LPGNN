import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import community
import torch
import numpy as np

def graphshow(graph, epoch, result_file, edgethreshold):
    plt.figure(num=1, figsize=(8, 8))
    G = nx.Graph()
    for i in range(len(graph)):
        for j in range(len(graph)):
            if (i != j) and (graph[i][j] > edgethreshold):
                G.add_edge(i, j)
    if len(G.edges()) > 2:
        fig, ax = plt.subplots(figsize=(10, 10))
        partition = community.best_partition(G)
        pos = nx.layout.spring_layout(G) 
        NodeId = list(G.nodes())
        node_size = [G.degree(i)**1.02 * 10 for i in NodeId]
        options = {
            'node_size': node_size,
            'edge_color': 'grey',
            'linewidths': 0.1,
            'width': 0.4,
            'style': 'solid',
            'nodelist': NodeId,
            'node_color': list(partition.values()),
            'font_color': 'b',
            'font_size': 10
        }
        nx.draw(G, pos=pos, ax=ax, with_labels=True, **options)
        plt.savefig(result_file + '_epo' + str(epoch) + '_thred' + str(edgethreshold) + "_Graph.png", bbox_inches='tight')
        print(f'Saved a graph !')
    plt.close('all')

def position_show(position_embd, result_file, epoch):
    dim=position_embd.shape[1]
    print(position_embd[0,:])
    print(position_embd[38,:])
    f = plt.figure(num=2,figsize=(16,16), dpi =200)
    f.clear()
    x_ = position_embd[:,0].cpu()
    y_ = torch.zeros(x_.shape[0])
    n=np.arange(len(x_))
    if dim==2 or dim==1:
        if dim==2:
            y_ = position_embd[:,1].cpu()
        plt.scatter(x_, y_, c=np.arange(0, 500*len(x_), 500))
        plt.grid(True)
        for i,txt in enumerate(n):
            plt.annotate(txt,(x_[i],y_[i]))
    if dim==3:
        y_ = position_embd[:,1].cpu()
        z_ = position_embd[:,2].cpu()
        ax = Axes3D(f)
        ax.scatter(x_, y_, z_, s=150, alpha=0.5, linewidths=1, c=np.arange(0, 500*len(x_), 500))
        ax.view_init(30, 20)
        ax.set(xlim=[min(x_), max(x_)], ylim=[min(y_), max(y_)], zlim=[min(z_), max(z_)])
    plt.savefig(result_file + '_epo' + str(epoch) + '_' + str(dim) + "_d_position_Graph.png", bbox_inches='tight')
    plt.close('all')