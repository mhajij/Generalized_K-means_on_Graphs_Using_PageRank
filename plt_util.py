# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 01:29:47 2021

@author: Mustafa Hajij
"""

import numpy as np
import matplotlib.pyplot as plt
import collections
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx


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
