# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 04:52:16 2021

@author: Mustafa Hajij
"""

import numpy as np
import argparse


import graphkmeans as gkm 
import plt_util as myplot

 
 
data_folder='data/'    

    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # required arguments
    parser.add_argument('-pc', '--pointcloud',type=str,required=True,help='Specify the path of the graph you want to compute the k means on.')   
    parser.add_argument('-k', '--k',type=int,required=True,help='number of clusters')


    # optional arguments
    parser.add_argument('-nbrs', '--neighbors',default=5,type=int,required=False,help='number of clusters')
    parser.add_argument('-it', '--n_iter',default=10, type=int,required=False,help='number of iterations')    
    parser.add_argument('-c', '--centrality_method',default="page_rank",required=False, help='centrality method, default is pagerank (very fast), other options : betweenness_centrality, information_centrality,local_reaching_centrality,voterank,closeness_centrality,harmonic_centrality, hits, and katz_centrality')
    parser.add_argument('-m', '--metric',default="dijkstra",required=False, help='metric calculations on the graph. Default is dijkstra. Other options are : communicability (slow)')

    # saving arguments
    parser.add_argument('-p', '--plot',default=True,required=False, help='save graph plot')
    parser.add_argument('-t', '--target_folder',default=data_folder,required=False,help='target folder where you want to save the output clusters')


    args = parser.parse_args() 

    X=np.load(args.pointcloud)
    
    kmeans_pc=gkm.GraphKMeansOnPointCloud(pointcloud=X,number_of_clusters=args.k,neighbors=args.neighbors,n_iter=args.n_iter,centrality_method=args.centrality_method,metric=args.metric)                   
    d=kmeans_pc.fit()  
    
    if args.plot:
        myplot.plot_point_cloud(X,d)

    if args.target_folder!=None:        
        np.save(args.target_folder+'pointcloud_cluster_dictionary.npy', d)  # load using np.load('file_name.npy',allow_pickle=True).item()
