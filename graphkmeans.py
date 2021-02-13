# -*- coding: utf-8 -*-
"""
Created on Fri May  3 16:31:38 2019

@author: Mustafa Hajij
"""


import random
import numpy as np
import scipy

import networkx as nx
from sklearn.neighbors import kneighbors_graph


class GraphKMeans():
    def __init__(self,graph,k,n_iter=50,centrality_method="page_rank",edge_weight_method=None,metric='dijkstra',relabel=True,fitting_method="explicit"):
        
        """"
        
        input:
        _____    
            graph : networkx graph
            k: number of clusters
            n_iter : number of iterations in the k means
            centrality_method : method used to find centers in graphs.
            edge_weight_method: impose weights on the edges of the graph
            metric : metric used in the computation of the k means on graph. Default is dijkstra.
            fitting_method : these are two methods available to fit : explicit and "voronoi_cells_method"
    
        """
        
        
        
        # graph clustering init
        if relabel==True:
            
            self.G=nx.convert_node_labels_to_integers(graph)
            print("The graph nodes have been relabeled to integers.")
        else:
            self.G=graph
        self.k=k # number of clusters
        
        if k>len(graph.nodes):
            raise ValueError("number of clusters cannot exceed number of nodes in the input graph.")
        
        self.iters=n_iter   
        
        self.final_dic={}
        
        # compute the weights of the edges in the graph
        
            
        self.set_edge_weight(edge_weight_method)
        
        self.method=centrality_method
        self.metric=metric
        self.distance=[]
        self.fitting_method=fitting_method
        
     

    def set_edge_weight(self,edge_weight_method='weight'):
        
        if edge_weight_method=='weight':
            return
        
        # Centrality based methods
        
        elif edge_weight_method=='edge_betweenness_centrality':
            print("comptuing edge_betweenness_centrality..")
            C=nx.edge_betweenness_centrality(self.G,weight='weight')
            print("done!")

        elif edge_weight_method=='edge_betweenness_centrality_subset':
            print("comptuing edge_betweenness_centrality_subset..")
            C=nx.edge_current_flow_betweenness_centrality(self.G,weight='weight')
            print('done')


        elif edge_weight_method=='edge_current_flow_betweenness_centrality_subset':
            print("comptuing edge_current_flow_betweenness_centrality_subset..")
            C=nx.edge_current_flow_betweenness_centrality_subset(self.G,weight='weight')
            print('done')
            
            
        elif edge_weight_method=='edge_load_centrality':
            print("comptuing edge_load_centrality..")
            C=nx.edge_load_centrality(self.G)  
            print('done!')
            
        # Link Prediction based methods
     
        elif edge_weight_method=='adamic_adar_index':
            print("comptuing adamic_adar_index ..")
            preds=nx.adamic_adar_index(self.G,self.G.edges())  
            C={}
            for u, v, p in preds:
                C[(u,v)]=p


        elif edge_weight_method=='ra_index_soundarajan_hopcroft':
            print("comptuing ra_index_soundarajan_hopcroft ..")
            preds=nx.ra_index_soundarajan_hopcroft(self.G,self.G.edges())  
            C={}
            for u, v, p in preds:
                C[(u,v)]=p 


        elif edge_weight_method=='preferential_attachment':
            print("comptuing preferential_attachment ..")
            preds=nx.preferential_attachment(self.G,self.G.edges())  
            C={}
            for u, v, p in preds:
                C[(u,v)]=p 

        #elif edge_weight_method=='cn_soundarajan_hopcroft':
        #    print("comptuing cn_soundarajan_hopcroft ..")
        #    preds=nx.cn_soundarajan_hopcroft(self.G,self.G.edges())  
        #    C={}
        #    for u, v, p in preds:
        #        C[(u,v)]=p 

        elif edge_weight_method=='within_inter_cluster':
            print("comptuing within_inter_cluster ..")
            preds=nx.within_inter_cluster(self.G,self.G.edges())  
            C={}
            for u, v, p in preds:
                C[(u,v)]=p                


        elif edge_weight_method=='resource_allocation_index':
            print("comptuing resource allocation index ..")
            preds=nx.resource_allocation_index(self.G,self.G.edges())  
            C={}
            for u, v, p in preds:
                C[(u,v)]=p
                

            
        elif edge_weight_method=='jaccard_coefficient':
            print("comptuing jaccard_coefficient..")
            preds=nx.jaccard_coefficient(self.G,self.G.edges())  
            C={}
            for u, v, p in preds:
                C[(u,v)]=p
                
            
            print('done!')            

            
        
        for u,v,d in self.G.edges(data=True):
            if edge_weight_method==None:
                d['weight']=1
            else:

                d['weight']=C[(u,v)]
              
        return 1

    # for f:V->values and x in values find f^{-1}(x)
    def getKeysByValue(self,dictOfElements, valueToFind):
        '''
        input :
        ______    
            dictOfElements: the function f
            
            valueToFind: a value in the codomain of f
              
        output: 
        _______    
            f^{-1}(x)      
        '''
        listOfKeys = list()
        listOfItems = dictOfElements.items()
        for item  in listOfItems:
            if item[1] == valueToFind:
                listOfKeys.append(item[0])
        return  listOfKeys

    def compute_metric(self,metric='dijkstra'):
        
        if metric=='dijkstra':
            print("computing the distance matrix on the graph")
            self.distance=dict(nx.all_pairs_dijkstra_path_length(self.G))
        else:
            print("computing communicability..")
            self.distance=nx.communicability(self.G)
            print("done!")
        
        return 1


    def compute_subgraph_center(self,subgraph):
        
                if self.method=='betweenness_centrality':
                    d=nx.betweenness_centrality(subgraph,weight='weight')
                    center=max(d, key=d.get)

                elif self.method=='betweenness_centrality_subset':
                    d=nx.betweenness_centrality_subset(subgraph,weight='weight')
                    center=max(d, key=d.get)

                elif self.method=='information_centrality':
                    d=nx.information_centrality(subgraph,weight='weight')
                    center=max(d, key=d.get)     

                elif self.method=='local_reaching_centrality':
  
                    d={}
                    for n in self.G.nodes():
                        d[n]=nx.local_reaching_centrality(self.G,n,weight='weight')
 
                    center=max(d, key=d.get)   

                elif self.method=='voterank':
                    d=nx.voterank(subgraph)
                    center=max(d, key=d.get)  

                elif self.method=='percolation_centrality':
                    d=nx.percolation_centrality(subgraph,weight='weight')
                    center=max(d, key=d.get)  
                    
                elif self.method=='subgraph_centrality':
                    d=nx.subgraph_centrality(subgraph)
                    center=max(d, key=d.get)     

                elif self.method=='subgraph_centrality_exp':
                    d=nx.subgraph_centrality_exp(subgraph)
                    center=max(d, key=d.get)  

                elif self.method=='estrada_index':
                    d=nx.estrada_index(subgraph)
                    center=max(d, key=d.get)  

                elif self.method=='second_order_centrality':
                    d=nx.second_order_centrality(subgraph)
                    center=max(d, key=d.get)  
                    
                elif self.method=='eigenvector_centrality':
          
                    d=nx.eigenvector_centrality(subgraph,weight='weight')
                    center=max(d, key=d.get)
                elif self.method=='load_centrality':
          
                    d=nx.load_centrality(subgraph,weight='weight')
                    center=max(d, key=d.get)                    
                    
                elif self.method=='closeness_centrality':
                    d=nx.closeness_centrality(subgraph)
                    center=max(d, key=d.get)
                    
                elif self.method=='current_flow_closeness_centrality':
                    d=nx.current_flow_closeness_centrality(subgraph,weight='weight')
                    center=max(d, key=d.get)

                elif self.method=='current_flow_betweenness_centrality':
                    d=nx.current_flow_betweenness_centrality(subgraph,weight='weight')
                    center=max(d, key=d.get)                    

                elif self.method=='current_flow_betweenness_centrality_subset':
                    d=nx.current_flow_betweenness_centrality_subset(subgraph,weight='weight')
                    center=max(d, key=d.get)  

                elif self.method=='approximate_current_flow_betweenness_centrality':
                    d=nx.approximate_current_flow_betweenness_centrality(subgraph,weight='weight')
                    center=max(d, key=d.get)
                    
                elif self.method=='harmonic_centrality':
                    d=nx.harmonic_centrality(subgraph)
                    center=max(d, key=d.get)
                    
                    
                elif self.method=='page_rank':
                    
                    d=nx.pagerank(subgraph,weight='weight')
                    center=max(d, key=d.get)

                elif self.method=='hits':
                    
                    d=nx.hits(subgraph)
                    center=max(d, key=d.get) 
                    
                elif self.method=='katz_centrality':
                    d=nx.katz_centrality(subgraph,weight='weight')
                    center=max(d, key=d.get)
                    
                else:
                    new_centers=nx.center(subgraph)
                    
                    # new_centers gives a list of centers and here we just pick one randomly --not good for stability
                    # to do : find a better way to choose the center--make it stable
                    
                    index=random.randint(0,len(new_centers)-1)
       
                    center=new_centers[index]   
                return center    
    
    
    def fit_voronoi_cells_method(self):
        #not robust to graph with multiple connected components
        '''
        Purpose :
        _________    
            compute the kmeans of the graph G using the method specified in the constructor
        
        Ouput :
        _______    
            a dictionary final_d :Graph.Nodes()-> Set of all centers
                every node is assocaited to its closest center 
                
        Remarks:
        ________
            Usually faster than the explicit method, however, the input graph must be connected for the voronoi cells to be utilized.
            If the graph is disconnected, then use the other fitting method, fit_explicit_method, for computation.
        '''
        
        if nx.is_connected(self.G)==False:
            raise ValueError("the input graph is disconnected. Use the method fit_explicit_method to fit the data instead.")
            
        # pick randomly k numbers from (0,len(G))    
        nodes_k=random.sample(self.G.nodes(), self.k)

        
        current_centers=nodes_k
        
        # the output is a function that maps every node to the center index
        final_d={}
        stable=0
        cells=[]
        for i in range(0,self.iters):
            print("Doing iteration "+str(i))
            # stage 1 : assigning each node to its closest center
            
            cells=nx.voronoi_cells(self.G, current_centers, weight='weight')
            '''
            For more inforamtion, check the refs: 
            
                
            (i) https://iopscience.iop.org/article/10.1088/1367-2630/16/6/063007/pdf?fbclid=IwAR0uCHUYqzdKhqp9DIOwZEMcUSxZs3h6ctT2WAEIYGw6p4SaNx-BqHXYOzA    
                
            (ii) Erwig, Martin. (2000), “The graph Voronoi diagram with applications.” Networks, 36: 156–163. <dx.doi.org/10.1002/1097-0037(200010)36:3<156::AID-NET2>3.0.CO;2-L>
            
            (iii) https://web.engr.oregonstate.edu/~erwig/papers/GraphVoronoi_Networks00.pdf?fbclid=IwAR0i1R4pfnfqwggBMlIfTc48WNl2ttFEbN2hkHim3F8E3RPUAVbbgm4MTLs      
            '''

               
            newcenters=[]
            # stage 2 : updating the centers based on the previous step.
      

            for cell in cells.values():
                
                #1 define the graph with the previous nodes :
                
                subgraph=self.G.subgraph(cell)
                
                #2 compute the center of the subgraph after choosing the center update method
                #_____________________
                
                center=self.compute_subgraph_center(subgraph)
                

                newcenters.append(center)
            #stage 3 check for convergence  
            if sorted(current_centers)==sorted(newcenters): # centers are not changing for a certain number of iterations
                stable=stable+1
                if stable==3:
                    print("Algorithm converges with "+str(i)+" steps.")
                    break
                
            #stage 4 update the centers if we did not converge and get ready for the next loop
            current_centers=newcenters  
        if stable!=3: # TODO : better implementation needed
            print("Algorithm did not converge")
            

        
        for key,value in zip(cells.keys() ,cells.values()):
            for v in value:
                final_d[v]=key

        self.final_dic=final_d 
        
        print("Done!")
        return final_d  

    
    def fit_explicit_method(self):
        '''
        Purpose:
        ________
        
            compute the kmeans of the graph G using the method specified in the constructor
        
        Ouput : 
            a dictionary final_d :Graph.Nodes()-> Set of all centers
                every node is assocaited to its closest center        
        '''
        
        # compute the distance matrix.
        self.compute_metric(self.metric) # this is a slow part of the algorithm, can be improved and a better implementation needed.
        
        # pick randomly k numbers from (0,len(G))  
        nodes_k=random.sample(self.G.nodes(), self.k) # TODO : find better method.
        
        # this is a collection of functions f_v:G->R one for each node v. Each one of them correponds to a node v and it is 
        # the distance from f_v(w)= distance betwen v and w
        
        
        #length=list(nx.all_pairs_dijkstra_path_length(self.G))
        
        
        # all_pairs_dijkstra_path(G, weight='weight')
        current_centers=nodes_k
        
        # the output is a function that maps every node to the center index.
        
        # computing final_d is equivalent to computing voronoi_cells.
        final_d={}
        stable=0
        
        for i in range(0,self.iters):
            print("doing iteration "+str(i))
            # stage 1 : assigning each node to its closest center

              
            for node in self.G.nodes():
               
                
                #1 this is the distance function from the node with index n_index to every other node              
                distance2node=self.distance[node]
                
                #2 for the current node find the distances to the current centers
                dict_you_want={}
                
                for c in current_centers:
                    if c in distance2node:
                        dict_you_want[c]=distance2node[c]
                    else:# dealing with disconnected graphs, this is needed in particular when working with point clouds 
                        dict_you_want[c]=np.Infinity

                #dict_you_want = { your_key: distance2node[your_key] for your_key in current_centers }
                
                #3 find the center that gives the min distance
                mincenter=min(dict_you_want, key=dict_you_want.get)
                
                #4 assign that center to the node and store this in the function final_d
                final_d[node]=mincenter
            # here we are assuming the center set is going to be updated in the next loop 
            newcenters=[] 
            # stage 2 : updating the centers based on the previous step.
      

            for center in current_centers:

                #1 get all nodes whose center is #center#
                subgraph_nodes=self.getKeysByValue(final_d,center)
                
                #2 define the graph with the previous nodes :
                subgraph=self.G.subgraph(subgraph_nodes)
                
                #3 compute the center of the subgraph after choosing the center update method
                #_____________________
                
                center=self.compute_subgraph_center(subgraph)
                

                newcenters.append(center)
            #stage 3 check for convergence  
            if sorted(current_centers)==sorted(newcenters):
                stable=stable+1
                if stable==3:
                    print("Algorithm converges with "+str(i)+" steps.")
                    break
                
            #stage 4 update the centers if we did not converge and get ready for the next loop
            current_centers=newcenters  
        if stable!=3:
            print("Algorithm did not converge")
    
        self.final_dic=final_d 
        
        print("done!")
        return final_d  

    def fit(self):
        if self.fitting_method=="voronoi_cells_method":
            return self.fit_voronoi_cells_method()
        else:
            return self.fit_explicit_method()



class GraphKMeansOnPointCloud(GraphKMeans):
    
    """
    Purpose: 
    ________    
        Clustering point cloud using k means on graphs. Advantages on Euclidian k means, can find natural clusters that are non-convex in the input domain.   
    """
    
    def __init__(self,pointcloud,number_of_clusters,neighbors=5,n_iter=50,centrality_method="page_rank",edge_weight_method='weight',metric='dijkstra'):

        """
        input:
        _____    
            
            pointcloud : numpy array
            k: number of clusters
            neighbors : number of neighbors in the k nearest neighborhood graph
            n_iter: number of iterations in the algorithm
            centrality_method : method used to find centers on the KNN graph (constructed on the top of the input point cloud).
            edge_weight_method: impose weights on the edges of the graph
            metric : metric used in the computation of the k means on graph. Default is dijkstra.  
        """
        
        self.PC=pointcloud
        A=kneighbors_graph(pointcloud,neighbors,mode='distance')
        
        self.G=nx.Graph()
        cx = scipy.sparse.coo_matrix(A)

        for i,j,v in zip(cx.row, cx.col, cx.data):
            self.G.add_edge(i,j,weight=v)
            

        GraphKMeans.__init__(self,graph=self.G,k=number_of_clusters,n_iter=n_iter,centrality_method=centrality_method,edge_weight_method=edge_weight_method,metric=metric,relabel=False,fitting_method="explicit")


