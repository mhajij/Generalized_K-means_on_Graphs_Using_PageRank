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

import graphkmeans as gkm 

class graph_parition_plot():
    def __init__(self,G,dic):

        self.G=G
        self.final_dic=dic        
        
        self.pos = nx.kamada_kawai_layout(self.G,weight='weight')
        
        self.colors=['red','green','blue','black','gold','grey','carol','cyan','m','firebrick','purple','khaki','orange','darkgreen']


    
    
    def draw_graph_partition(self,node_size=40,width=1,edge_color='grey',set_weight=None,constant_value=None,to_hd=True):
        '''
        draw the graph self.G
        '''
        if nx.is_connected(self.G)==False:
            
            raise ValueError("The graph is diconnected, this function will terminate.")
            return


        # to do : support more color options
        vals=np.unique(list(self.final_dic.values()))
        
        color_d={}
        i=0
        
        for v in vals:
            color_d[v]=self.colors[i]
            
            i=i+1
            
        color_map = []
        if constant_value != None:
            
            color_map=[0]*len(self.G.node())
            
            
        else:
                
            for i in range(0,len(self.G.node())):
                
                v=self.final_dic[i]
                color_map.append(color_d[v])
            
        
        #nx.draw_kamada_kawai(self.G)
        nx.draw(self.G,pos=self.pos,node_size=node_size,edge_color=edge_color,width=width,node_color = color_map)  
        if to_hd:
            plt.show(block=False)
            plt.savefig("Graph_test.png", format="PNG")
        return 1
        
    
 



    
 
 
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
    #R = nx.random_geometric_graph(200, 0.15)
    #nx.write_gml(R, "sample_graph.graph")     
    #G=nx.read_gml("sample_graph.graph")  
    
    G=nx.read_gml(args.graph)
    kmeans=gkm.GraphKMeans(G,args.k,args.n_iter,centrality_method=args.centrality_method,metric=args.metric)                   
    d=kmeans.fit()  
    
    if args.plot:
        print("saving plot to main repo folder")        
        draw=graph_parition_plot(G,d)    
        draw.draw_graph_partition() # currently supports 14 colors (14 clusters)

    if args.target_folder!=None:        
        np.save(args.target_folder+'graph_cluster_dictionary.npy', d)  # load using np.load('file_name.npy',allow_pickle=True).item()
