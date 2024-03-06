import numpy as np
from scipy.stats import kurtosis,mode, skew, beta
import sys
from typing import List, Dict, Union,Tuple
import networkx as nx
import pandas as pd
from collections import defaultdict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import tqdm
import NetworkCurvature as nc
from sklearn.decomposition import PCA



DRUG_TISSUE_MAP = {"Atezo":["KIRC","BLCA"],"Pembro":["STAD"],
    "Nivo":["KIRC","SKCM"], "Ipi":["SKCM"], "Ipi+Pembro":["SKCM"],
    "erlotinib":['LUAD'],"crizotinib":['LUAD'],'sorafenib':["LUAD"],'sunitinib':["LUAD"]}


DRUG_DATASET_MAP = {
    'sorafenib':'ccle',
    'erlotinib':'ccle',
    'crizotinib':'ccle',
    'sunitinib':'ccle',
    'Nivo':'cri',
    'Ipi':'cri',
    'Pembro':'cri',
    'Atezo':'cri',
    'Ipi+Pembro':'cri'
}


DRUG_TARGET_MAP = {'Atezo':'PD-L1','Pembro':'PD1','Nivo':'PD1','Ipi':'CTLA4'}


TARGET_GENE_MAP = {'PD-L1':'CD274', 'PD1':'PDCD1', 'CTLA4':'CTLA4'}



def parse_transform(
    transform:str
    ) -> Tuple[str,bool, bool]:
    if transform in ['curvature',"curvature-hist",'curvature-pca']:
        feature_map = 'edge_curvature'
    elif transform in ['weights','weights-pca']:
        feature_map = 'edge_weight'
    else:
        feature_map = 'log2p1'

    if transform in ['curvature-pca','weights-pca','expression-pca']:
        do_pca = True
    else:
        do_pca = False

    if transform == 'curvature-hist':
        do_histogram = True
    else:
        do_histogram = False

    return feature_map, do_pca, do_histogram

def featurize_data(
    data:pd.DataFrame,
    PPI_Graph:nx.Graph,
    feature_map:str,
    idx_to_gene
    ):
    
    y = data['Response'].values
    data_subset = data.drop(['Run_ID','Response'],axis=1)
    

    if feature_map == 'log2p1':
        X = np.log2(data_subset.values+1)
    else:
        X = np.empty((data_subset.shape[0],len(PPI_Graph.edges())))
        for idx, row in tqdm.tqdm(data_subset.iterrows(), total = data_subset.shape[0]):
            G = PPI_Graph.copy()
            for edge in G.edges():
                node1, node2 = edge[0], edge[1]
                gene1, gene2 = idx_to_gene[node1], idx_to_gene[node2]
                weight = np.round(np.log2(row[gene1]+1)*np.log2(row[gene2]+1),5)
                G[node1][node2]['weight'] = weight
            if feature_map == 'edge_weight':
                 X[idx,:] =[e[2]['weight'] for e in G.edges(data=True)]
            elif feature_map == 'edge_curvature':
                orc = nc.OllivierRicciCurvature(G)
                orc.compute_edge_curvatures()
                X[idx,:] =[e[2]['ricci_curvature'] for e in orc.G.edges(data=True)]
    return X,y

def fetch_geneset(
    geneset_dir:str,
    geneset:str,
    ) -> List[str]:
    

    geneset = geneset.upper()
    assert geneset in ['LINCS','COSMIC', "LINCS+COSMIC"], "invalid geneset"
    
    if geneset in ['LINCS','COSMIC']:
        fname = geneset_dir + geneset + ".txt"
        with open(fname, 'r') as istream:
            lines = istream.readlines()
            lines = [x.rstrip() for x in lines]
            gene_list = lines
    elif geneset == "LINCS+COSMIC":
        gene_list = []
        for gs in ["LINCS","COSMIC"]:
            fname = geneset_dir + gs+ ".txt"
            with open(fname, 'r') as istream:
                lines = istream.readlines()
                lines = [x.rstrip() for x in lines]
            genes = lines
            gene_list.extend(genes)
    gene_list = list(pd.unique(gene_list))
    return gene_list



def unpack_parameters(
    D:Dict
    ):
    if len(D.values())>1:
        return tuple(D.values())
    else:
        return tuple(D.values())[0]

def make_file_path(
    base_dir:str, 
    path_names:List[str], 
    fname:str, 
    ext:str
    )->str:
    
    path_names.append("")
    ext = ext if ext[0]=="." else "." + ext

    base_dir = base_dir[:-1] if base_dir[-1]=="/" else base_dir
    path = [base_dir]
    path.extend(path_names)
    path ="/".join([x for x in path])
    
    file_path = "".join([path,fname,ext])
 
    

    return file_path

def harmonize_graph_and_geneset(
    G:nx.Graph,
    gene_set:List[str]
    ) -> nx.Graph:
    

    
    common_genes = [x for x in list(G.nodes) if x in gene_set]
    
    G.remove_nodes_from([n for n in G.nodes if n not in common_genes])

    LCC_genes = sorted(list(max(nx.connected_components(G), key=len)))
    G.remove_nodes_from([n for n in G.nodes if n not in LCC_genes])
    
    
    
    return G


def rename_nodes(
    G:nx.Graph,
    new_field_name:str = 'Gene'
    ):
    gene_to_idx  = {} 
    idx_to_gene = {}
    for idx, gene in enumerate(G.nodes):
        gene_to_idx[gene] = idx
        idx_to_gene[idx] = gene
    G = nx.relabel_nodes(G,gene_to_idx)
    nx.set_node_attributes(G,idx_to_gene, new_field_name)
    return G, gene_to_idx, idx_to_gene


def histogram_transform(
    X:np.array,
    num_bins:int,
    lower_bound:int
    )->np.array:
    

    X_ = np.empty((X.shape[0],num_bins))
    bins = np.concatenate(([-np.inf], np.linspace(lower_bound,1,num = num_bins)))
    
    for row in range(X.shape[0]):
        X_[row,:] = np.histogram(X[row,:],bins = bins)[0]
    return X_
    
def make_model_and_param_grid(
    model_name:str,
    reg_min:float,
    reg_max:float,
    reg_step:float,
    model_max_iters:int,
    do_pca:bool = False,
    pca_dim: int = 20
    ):
    
  
    preproc = [('preproc',StandardScaler())]
    if do_pca:
        preproc = preproc + [('pca',PCA())]
    
    if model_name == 'LogisticRegression':
        classifier = ('clf',LogisticRegression(class_weight = 'balanced',max_iter = model_max_iters))
        param_grid = {'clf__C':np.arange(reg_min,reg_max,reg_step)} 
    elif model_name == 'LinearSVM':
        classifier = ('clf',LinearSVC(class_weight = 'balanced',max_iter = model_max_iters))
        param_grid = {'clf__C':np.arange(reg_min,reg_max,reg_step)} 
    

    
    if do_pca:
        param_grid['pca__n_components'] = [pca_dim]                          
    
    model = preproc+[classifier]
    

    return Pipeline(model), param_grid

