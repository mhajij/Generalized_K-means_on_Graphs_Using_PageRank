# Generalized_K-means_on_Graphs
Generalized K-means on graph is an algorithm that utilizes centrality measures such as PageRank, harmonic centrality, etc to obtain k-means clustering algorithm to directed and undirected graphs. 

The details of the algorithm are described in the paper :

https://diglib.eg.org/xmlui/bitstream/handle/10.2312/cgvc20201152/063-066.pdf?sequence=1

The script demonstrated here can be applied to obtain cluster graphs as well as point clouds.

## Downloading the repo


## Using the script on graphs 


## Using the script on point clouds

The algorithm can be utlized to obtain a clustering algorithm for point cloud. For instance the following computes the algorithm on 2 centric circles dataset with k=2 and default centrality measure set to PageRank.  

```ruby
python main_pc.py -pc circles.npy -k 2 -nbrs 5 
```

Utilizing PageRank as the centralitiy measure subroutine is fast but it can lead to sub-optimal clustering results. If qualtiy of clusters are disrable then other centraltiy measures such as harmonic centrality are recoemnded. The script support many centrality measure. For instance one can specify harmonic centrality  

```ruby
main_pc.py -pc moon.npy -k 2 -nbrs 5 -c harmonic_centrality
```


## Citation

@article{hajij2020generalized,
  title={Generalized K-means for Metric Space Clustering Using PageRank},
  author={Hajij, Mustafa and Said, Eyad and Todd, Robert},
  year={2020},
  publisher={The Eurographics Association}
}
