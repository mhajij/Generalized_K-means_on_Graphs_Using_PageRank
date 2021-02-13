# Generalized k-means on Graphs
Generalized K-means on graph is an algorithm that utilizes centrality measures such as PageRank, harmonic centrality, etc to obtain a k-means-like clustering algorithm to directed and undirected graphs.

The details of the algorithm are described in the paper :

https://diglib.eg.org/xmlui/bitstream/handle/10.2312/cgvc20201152/063-066.pdf?sequence=1

The script demonstrated here can be applied to obtain cluster graphs as well as point clouds.

## Package Requirement

* NetworkX >= 2.0 (Based Graph library)
* scikit-learn >= 0.23.2
* NumPy 

## Simple Example


```ruby
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from graphkmeans import GraphKMeans

# (1) define the graph
G = nx.random_geometric_graph(350, 0.115)

# (2) Run the clustering algorithm
graph_k_means=GraphKMeans(G,k=4,n_iter=10,centrality_method="page_rank")               
d=graph_k_means.fit() # returns a dictionary with keys are nodes of the graph, values are id of the clusters

# (3) plot to inspect the clustering results
draw=graph_parition_plot(G,d)    
draw.draw_graph_partition()

```


### Using the script for graph clustering 

The algorithm can be utilized to obtain cluster on a given graph. Two main arguments are assumed : the input graph G and the number of cluster k. The following is an example.

```ruby
python main_graphkmeans.py -G graph_example.graph -k 4
```

### Using the script for point clouds clustering

The algorithm can be utlized to obtain a clustering algorithm for point cloud. For instance the following computes the algorithm on 2 centric circles dataset with k=2 and default centrality measure set to PageRank.  

```ruby
python main_pointcloud.py -pc circles.npy -k 2 -nbrs 5 
```

Utilizing PageRank as the centralitiy measure subroutine is fast but it can lead to sub-optimal clustering results. If qualtiy of clusters are disrable then other centraltiy measures such as harmonic centrality are recoemnded. The script support many centrality measure. For instance one can specify harmonic centrality  

```ruby
main_pointcloud.py -pc moon.npy -k 2 -nbrs 5 -c harmonic_centrality
```


## Cite
```ruby
@article{hajij2020generalized,
  title={Generalized K-means for Metric Space Clustering Using PageRank},
  author={Hajij, Mustafa and Said, Eyad and Todd, Robert},
  year={2020},
  publisher={The Eurographics Association}
}
```
