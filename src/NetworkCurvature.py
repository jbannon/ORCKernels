from typing import Dict, Tuple
import matplotlib.pyplot as plt
import sys
import numpy as np
import networkx as nx
import ot
import tqdm 
import time
from GraphRicciCurvature.OllivierRicci import OllivierRicci


def _make_all_pairs_shortest_path_matrix(
	G:nx.Graph,
	weight:str = 'weight'
	) -> np.ndarray:
	
	
	N = len(G.nodes)
	D = np.zeros((N,N))


	path_lengths = dict(nx.all_pairs_dijkstra_path_length(G,weight=weight))
	for node1 in path_lengths.keys():
		node1Lengths = path_lengths[node1]
		for node2 in node1Lengths.keys():
			D[node1,node2] = np.round(node1Lengths[node2],5)
	e = time.time()
	# print("apsp took {t}".format(t=e-s))
	if not (D==D.T).all():
		issues = np.where(D!=D.T)
		print("symmetry issue")
		sys.exit(1)

	return D



def _assign_single_node_density(
	G:nx.Graph,
	node:int,
	weight:str = 'weight',
	alpha:float = 0.5,
	measure_name:str = 'density',
	min_degree_value:float = 1E-5
	) -> None:
	
	
	
	neighbors_and_weights = []
	for neighbor in G.neighbors(node):
		edge_weight = G[node][neighbor][weight]
		neighbors_and_weights.append((neighbor,edge_weight))
	
	node_degree = sum([x[1] for x in neighbors_and_weights])
	nbrs = [x[0] for x in neighbors_and_weights]

	if node_degree>min_degree_value:
		
		pmf = [(1.0-alpha)*w/node_degree for _,w in neighbors_and_weights]
		labeled_pmf = [(nbr,(1.0-alpha)*w/node_degree) for nbr, w in neighbors_and_weights]
	else:
		# assign equal weight to all neighbors
		pmf = [(1.0-alpha)/len(neighbors_and_weights)]*len(neighbors_and_weights)
		labeled_pmf = [(nbr,(1.0-alpha)/len(neighbors_and_weights)) for nbr, w in neighbors_and_weights]
	
	nx.set_node_attributes(G,{node:{x:y for x,y in labeled_pmf}},measure_name)
	
	
	return pmf + [alpha], nbrs + [node]
	


def _compute_single_edge_curvature(
	G:nx.Graph,
	node1:int, 
	node2:int, 
	weight:str = "weight",
	alpha:float = 0.5,
	measure_name:str = 'density',
	APSP_Matrix:np.array = None,
	min_degree_value:float = 1E-5,
	min_distance:float = 1E-7,
	path_method:str = "all_pairs",
	)-> Tuple[Tuple[int,int],float]:
	


	pmf_x, nbrs_x = _assign_single_node_density(G,node1,weight, alpha, measure_name, min_degree_value)
	pmf_y, nbrs_y = _assign_single_node_density(G,node2, weight, alpha, measure_name, min_degree_value)
	
	

	D = APSP_Matrix[np.ix_(nbrs_x, nbrs_y)]

	
	node1_to_node2_distance, _ = nx.bidirectional_dijkstra(G, node1,node2)


	
	OT_distance = ot.emd2(pmf_x, pmf_y,D)
	if node1_to_node2_distance < min_distance:
		kappa = 0
	else:
		kappa = 1 - OT_distance/node1_to_node2_distance
	

	return ((node1,node2),kappa)

	


	



class OllivierRicciCurvature:
	def __init__(
		self,
		G:nx.Graph,
		alpha:float = 0.5,
		weight_field:str = "weight",
		curvature_field:str = "ricci_curvature",
		measure_name:str = 'density',
		min_distance:float = 1E-5,
		min_degree:float = 1E-5,
		verbose: bool = False
		) ->None:
		
		self.G = G.copy()
		self.alpha = alpha
		self.weight_field = weight_field
		self.curvature_field = curvature_field
		self.measure_name = measure_name
		self.min_distance = min_distance
		self.min_degree = min_degree
		

		if not nx.get_edge_attributes(self.G, self.weight_field):
			for (v1, v2) in self.G.edges():
				self.G[v1][v2][self.weight_field] = 1.0


		self.verbose = verbose
		self.APSP_Matrix = _make_all_pairs_shortest_path_matrix(self.G,self.weight_field).copy()


	def compute_edge_curvatures(
		self
		) -> None:
		
		curvatures = {}
		for edge in (tqdm.tqdm(self.G.edges(),leave=False) if self.verbose else self.G.edges()):
			node1, node2 = edge[0],edge[1]
			
			curv_tuple = _compute_single_edge_curvature(
				G = self.G,
				node1 = node1,
				node2 = node2,
				weight = self.weight_field,
				alpha = self.alpha,
				measure_name = self.measure_name,
				min_degree_value = self.min_degree,
				min_distance = self.min_distance,
				APSP_Matrix = self.APSP_Matrix
				)

			curvatures[curv_tuple[0]] = curv_tuple[1]
		nx.set_edge_attributes(self.G, curvatures, self.curvature_field)
		self.edge_curvatures_computed = True

	
	




