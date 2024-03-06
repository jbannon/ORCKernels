import math
from sklearn import preprocessing, metrics
import pandas as pd
import networkx as nx
import numpy as np
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt
from GraphRicciCurvature.util import cut_graph_by_cutoff
from GraphRicciCurvature.OllivierRicci import OllivierRicci




gfile = "../data/genesets/LINCS.txt"

with open(gfile, 'r') as istream:
	lines = istream.readlines()
	geneset_genes = [x.rstrip() for x in lines]

exp = "../data/expression/cri/Pembro/SKCM/expression.csv"
exp = pd.read_csv(exp)
exp = exp[['Run_ID']+[x for x in geneset_genes if x in exp.columns]]

resp = "../data/expression/cri/Pembro/SKCM/response.csv"
resp = pd.read_csv(resp)

targs = resp['Response'].values
tmap = {0:'NR',1:'R'}
labels = {i:tmap[targs[i]] for i in range(len(targs))}
X = np.log2(exp[exp.columns[1:]].values+1)
A = kneighbors_graph(X, 15, mode='distance', include_self=False).toarray()
# for i in range(A.shape[0]):
#     for j in range(A.shape[1]):
#         if A[i,j]>0:
#             w = np.exp(-np.linalg.norm(X[i,:]-X[j,:])**2/2)
#             A[i,j] = w
G = nx.from_numpy_array(A)
nx.set_node_attributes(G,labels, 'response')

colors = []
for node in G.nodes(data = True):
    if node[1]['response']=='NR':
        colors.append("blue")
    else:
        colors.append("red")

nx.draw_spring(G,node_color = colors,node_size=20)
plt.savefig("Pembro_SKCM_raw.png")


print(G)
print(len(colors))


orf = OllivierRicci(G, alpha=0.5,proc=4)
orf.compute_ricci_flow(iterations = 50)
G = orf.G.copy()
print(np.array([e[2]['weight'] for e in G.edges(data=True)]))
cutoff = np.quantile(np.array([e[2]['weight'] for e in G.edges(data=True)]), 0.80)
print(cutoff)
# cc = orf.ricci_community()
# print(cc)
G = cut_graph_by_cutoff(orf.G,cutoff=cutoff)



# print(G)


nx.draw_spring(G,node_color = colors,node_size=20)
plt.savefig("Pembro_SKCM_clusters.png")
