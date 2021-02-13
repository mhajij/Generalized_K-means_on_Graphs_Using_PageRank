# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 04:52:16 2021

@author: Mustafa Hajij
"""

import numpy as np
import networkx as nx
import argparse

import graphkmeans as gkm 
import plt_util as myplot

 
 
data_folder='data/'    

    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # required arguments
    parser.add_argument('-G', '--graph',type=str,required=True,help='Specify the path of the graph you want to compute the k means on.')   
    parser.add_argument('-k', '--k',type=int,required=True,help='number of clusters')

    # optional arguments
    parser.add_argument('-it', '--n_iter',default=10, type=int,required=False,help='number of iterations')    
    parser.add_argument('-c', '--centrality_method',default="page_rank",required=False, help='centrality method, default is pagerank (very fast), other options : betweenness_centrality, information_centrality,local_reaching_centrality,voterank,closeness_centrality,harmonic_centrality, hits, and katz_centrality')
    parser.add_argument('-m', '--metric',default="dijkstra",required=False, help='metric calculations on the graph. Default is dijkstra. Other options are : communicability (slow)')

    # saving arguments
    parser.add_argument('-p', '--plot',default=True,required=False, help='save graph plot')
    parser.add_argument('-t', '--target_folder',default=data_folder,required=False,help='target folder where you want to save the output clusters')


    
    args = parser.parse_args()    

    
    G=nx.read_gml(args.graph)
    kmeans=gkm.GraphKMeans(G,args.k,args.n_iter,centrality_method=args.centrality_method,metric=args.metric)                   
    d=kmeans.fit()  
    
    if args.plot:
        print("saving plot to main repo folder")        
        draw=myplot.graph_parition_plot(G,d)    
        draw.draw_graph_partition() # currently supports 14 colors (14 clusters)

    if args.target_folder!=None:        
        np.save(args.target_folder+'graph_cluster_dictionary.npy', d)  # load using np.load('file_name.npy',allow_pickle=True).item()
