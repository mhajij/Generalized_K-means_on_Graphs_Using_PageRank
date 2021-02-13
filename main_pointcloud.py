# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 04:52:16 2021

@author: mustafa
"""

import collections
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import argparse

from mpl_toolkits.mplot3d import Axes3D

from sklearn.datasets import make_swiss_roll
from sklearn import cluster, datasets

import graphkmeans as gkm 

 



def plot_point_cloud(X,d,to_hd=True):
    """
    X : numpy array of shape (n,2) or (n,3).
    d : a dictionary with values represent the partition of X.   
    """
    
    od = collections.OrderedDict(sorted(d.items()))
    
    label=np.array(list(od.values()))
    labelss=set(od.values())
        
    fig = plt.figure()

    if len(X[0])==3:
        ax = Axes3D(fig)
        for l in labelss:
            ax.scatter(X[label == l, 0], X[label == l, 1],X[label == l, 2],
                       color=plt.cm.jet(float(l) / np.max(label + 1)),
                       s=20, edgecolor='k')     
        plt.show()  
    
    
    elif len(X[0])==2:
        for l in labelss:
            plt.scatter(X[label == l, 0], X[label == l, 1],
                       color=plt.cm.jet(float(l) / np.max(label + 1)),
                       s=20, edgecolor='k')        
    else:
        raise ValueError("the set X has " + str(len(X[0])) +" dimension and it cannot be plotted." )
        
    if to_hd:
        plt.show(block=False)
        plt.savefig("pointcloud_test.png", format="PNG")

    
 
 
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
        plot_point_cloud(X,d)

    if args.target_folder!=None:        
        np.save(args.target_folder+'pointcloud_cluster_dictionary.npy', d)  # load using np.load('file_name.npy',allow_pickle=True).item()
