# GrAMME
GrAMME: Semi-Supervised Learning using Multi-layered Graph Attention Models

In this work, we perform semi-supervised learning on multi-layer graphs using attention models.
We have proposed two attention models for effective feature learning, GrAMME-SG and GrAMME-Fusion, that exploit the inter-layer
dependencies for building multi-layered graph embeddings. Both are architectures are depicted in the figures below:
## GrAMME-Fusion
![GrAMME-Fusion](https://github.com/udayshankars/GrAMME/blob/master/gramme-fusion-1.jpg){height=60% width=60%}

## GrAMME-SupraGraph
![GrAMME-SupraGraph](https://github.com/udayshankars/GrAMME/blob/master/gramme-sg-1.jpg)


## Usage
-----

**Example Usage**
``python run_gramme_fusion.py --multiplex_edges_filename multilayer.edges
--multiplex_labels_filename multilayer.labels``

*multilayer.edges* file should be an edgelist along with layer information as shown below:

``layer_ID node_ID node_ID``

layer_ID starts from 0 and node_ID starts from 0

**--multiplex_edges_filename**: *multilayer.edges*, eg::
```  
0 0 1
0 1 0
0 2 5
0 5 2
...
1 0 3
1 3 0
1 4 9
1 9 4
...
2 0 7
2 7 0
2 1 3
2 3 1
...
```
**Note**: For undirected edges between node u and v, edgelist should contain both the entries u-> v and v-> u
```
0 u v
0 v u
```

*multilayer.labels* should only contain the true labels of the nodes (explicit node ids should not be there), class labels start from 0 to C-1 where C is the total number of classes

**multiplex_labels_filename**: *multilayer.labels*, eg::
```
0
0
2
1
9
0
3
...
```


## Requirements
* python 3.6
* numpy >= 1.11.0
* tensorflow >= 1.5.0
* scikit-learn >= 0.18
* matplotlib >= 2.1.0

## Citations

If you find GrAMME useful in your research, please cite the following paper:
```
@article{shanthamallu2018attention,
  title={GrAMME: Semi-Supervised Learning using Multi-layered Graph Attention Models},
  author={Shanthamallu, Uday Shankar and Thiagarajan, Jayaraman J and Song, Huan and Spanias, Andreas},
  journal={arXiv preprint arXiv:1810.01405},
  year={2018}
}
```
