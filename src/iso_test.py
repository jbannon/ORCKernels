# import networkx.algorithms.isomorphism as iso
from networkx.algorithms import bipartite
import networkx as nx
import matplotlib.pyplot as plt 
import seaborn as sns 
import NetworkCurvature as nc 
import sys
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np 
import os

from grakel.kernels import ShortestPath
from grakel.utils import graph_from_networkx

def plot_curvature_hist(graph_label, curvatures, make_title = True):
	plt.hist(curvatures,bins=np.linspace(-0.5,1,20))
	plt.xlabel('Ricci curvature')
	plt.ylabel('Count')
	if make_title:
		plt.title("{g} Graph Edge Curvature Histogram".format(g=graph_label))
	os.makedirs("../figs/examples/",exist_ok = True)
	plt.savefig("../figs/examples/{g}_hist.png".format(g=graph_label))
	plt.close()
	# plt.show()
	


def plot_graph(
	G,
	graph_label,
	curvatures
	):
	colors=curvatures
	cmap=plt.cm.viridis
	vmin = min(curvatures)
	vmax = max(curvatures)
	if graph_label == 'Grid':
		pos = {(x,y):(y,-x) for x,y in G.nodes()}
	# elif graph_label == 'Tree':
	# 	pos = nx.nx_agraph.graphviz_layout(G)
	elif graph_label == 'G2':
		# pos=nx.fruchterman_reingold_layout(G)
		pos=nx.spectral_layout(G)
	else:
		pos=nx.circular_layout(G)
		# pos=nx.spectral_layout(G)
		pos=nx.fruchterman_reingold_layout(G)
		# pos = nx.spring_layout(G,k=5,pos=pos,seed=12345)
	fig, ax = plt.subplots()
	label_toggle = True if graph_label in ['G1','G2'] else False
	nx.draw(G,pos, node_color='black',edge_color=curvatures, width=4, edge_cmap=cmap,with_labels=label_toggle, vmin=vmin, vmax=vmax, font_color = 'white')
	sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin = vmin, vmax=vmax))
	cbar = plt.colorbar(sm)#, fraction=0.5, pad=0)
	cbar.ax.set_ylabel('Edge Curvature',labelpad = 20,rotation = 270,fontsize=16)
	plt.savefig("../figs/examples/{g}_graph.png".format(g=graph_label))
	# plt.show()
	plt.close()
	

rng = np.random.seed(1234)
N = 13
G1 = nx.fast_gnp_random_graph(N,0.3,rng)
while not nx.is_connected(G1):
	G1 = nx.fast_gnp_random_graph(N,0.3,rng)

G2 = nx.fast_gnp_random_graph(N,0.7,rng)
c = 0

while nx.is_isomorphic(G1,G2):
	c = c+1
	G2 = nx.fast_gnp_random_graph(N,0.3,rng)
	print(c)


print("generation done")
# nx.draw(G1,with_labels = True)


# nx.draw(G2,with_labels = True)
for graph_label, G in zip(['Random Graph 1','Random Graph 2'],[G1,G2]):
	orc = nc.OllivierRicciCurvature(G)
	orc.compute_edge_curvatures()
	G = orc.G.copy()
	edges, curvatures = zip(*nx.get_edge_attributes(G,"ricci_curvature").items())
	plot_curvature_hist(graph_label, curvatures)
	plot_graph(G,graph_label,curvatures)




G1 = nx.Graph()
G1.add_edges_from([(1,4),(1,2),(1,6),(2,6),(4,5),(4,3),(5,3)])
G2 = nx.cycle_graph(range(1,7))
G2.add_edges_from([(1,4)])


for graph_label, G in zip(['G1','G2'],[G1,G2]):
	print(graph_label)
	orc = nc.OllivierRicciCurvature(G)
	orc.compute_edge_curvatures()
	G = orc.G.copy()
	edges, curvatures = zip(*nx.get_edge_attributes(G,"ricci_curvature").items())
	for e in G.edges(data=True):
		print(e)
	plot_curvature_hist(graph_label, curvatures,False)
	plot_graph(G,graph_label,curvatures)

G =  graph_from_networkx([G1,G2])
# print(G)
G = [g for g in G]
# print(G)
sp_kernel = ShortestPath(with_labels = False)
sp_kernel.fit(G)
print(sp_kernel.parse_input(G))
