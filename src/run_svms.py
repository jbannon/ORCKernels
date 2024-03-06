import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import confusion_matrix, roc_auc_score, auc, precision_recall_curve, accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
import time 
from collections import defaultdict
import matplotlib.pyplot as plt
import argparse
from typing import Dict, Union, List
import sys
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import numpy as np 
import networkx as nx
import ot
import os
import yaml 
import utils
import pickle
import NetworkCurvature as nc
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import tqdm


"""


Simple Experiment to tell if curvature histogram is better than:
	-> Weights w/ PCA to same number of dims 
	-> Expression w/ PCA to same number of dims
	-> Try Normalizing weights

"""


def main(
	Config:Dict
	) -> None:
	
	

	drug, tissues, geneset, rng_seed, transformations, alpha, min_TPM, num_iters, \
	train_pct,num_bins = \
		utils.unpack_parameters(config['EXPERIMENT_PARAMS']) 
	
	data_dir, geneset_dir, network_dir,result_dir = \
		utils.unpack_parameters(config['DIRECTORIES'])

	topology, weighting, min_distance  = \
		utils.unpack_parameters(config['NETWORK_PARAMS'])
	

	weight_field, rename_field, density_field, edge_curvature_field = \
		utils.unpack_parameters(config['FIELD_NAMES'])


	# model_name, min_C, max_C, C_step, max_iter = \
	# 	utils.unpack_parameters(config["MODEL_PARAMS"])
	model_info = config['MODEL_PARAMS']
	
	rng = np.random.RandomState(rng_seed)
	

	for tissue in tissues:
		expression_file = utils.make_file_path(data_dir,[drug, tissue],'expression','.csv')
		response_file = utils.make_file_path(data_dir,[drug,tissue],'response','.csv')
		
		response = pd.read_csv(response_file)		
		expression = pd.read_csv(expression_file)
		
		trimmed_expression = expression[expression.columns[1:]]
		trimmed_expression = (trimmed_expression>min_TPM).all(axis=0)
		
		keep_genes = list(trimmed_expression[trimmed_expression].index)
		

		gene_universe = utils.fetch_geneset(geneset_dir, geneset)
		genes = [x for x in keep_genes if x in gene_universe]
		

	
		res_path = "".join([result_dir,"/".join(["classification",drug,tissue,geneset]),"/"])
		os.makedirs(res_path,exist_ok = True)
		res_name = "{p}svm_classification .csv".format(p=res_path)
		
		network_file = utils.make_file_path(network_dir,[topology],weighting,".pickle")
		with open(network_file,"rb") as istream:
			PPI_Graph = pickle.load(istream)
		

		
		PPI_Graph = utils.harmonize_graph_and_geneset(PPI_Graph,genes)
		
		keep_cols = [g for g in PPI_Graph.nodes()]


		expression = expression[['Run_ID']+ keep_cols]
		expression = expression.merge(response, on = 'Run_ID')
		
		PPI_Graph, gene_to_idx, idx_to_gene = utils.rename_nodes(PPI_Graph,rename_field)

		
		

		transformed_data_cache = {}
		for fmap in ['edge_curvature','edge_weight','log2p1']:
			X_temp,y = utils.featurize_data(expression,PPI_Graph,fmap,idx_to_gene)
			transformed_data_cache[fmap] = (X_temp.copy(), y.copy())

		results = defaultdict(list)


		for model_name in model_info.keys():
			for transform in transformations:
				print("working on: {t}".format(t=transform))
				
				feature_map, do_pca, do_histogram = utils.parse_transform(transform)
				
				model_info[model_name].update({'do_pca':do_pca,'pca_dim':num_bins})
				model, param_grid = utils.make_model_and_param_grid(model_name,**model_info[model_name])
				

				X,y = transformed_data_cache[feature_map]
			
				for i in tqdm.tqdm(range(num_iters)):
					X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = train_pct, 
						random_state = rng,shuffle = True, stratify = y)
					

					if transform in ['curvature',"curvature-norm"]:
						
						X_train, lb = utils.histogram_transform(X_train,num_bins)
						X_test, _ = utils.histogram_transform(X_test, num_bins, lb)
						
					clf = GridSearchCV(model, param_grid)
					clf.fit(X_train,y_train)
					
					train_preds_bin = clf.predict(X_train)
					test_preds_bin = clf.predict(X_test)
					
					train_acc = accuracy_score(y_train, train_preds_bin)
					test_acc = accuracy_score(y_test, test_preds_bin) 
					test_bal_acc = balanced_accuracy_score(y_test, test_preds_bin)

					tn, fp, fn, tp = confusion_matrix(y_test, test_preds_bin,labels = [0,1]).ravel()
								
					results['drug'].append(drug)
					results['tissue'].append(tissue)
					results['feature'].append(transform)
					results['iter'].append(i)
					results['model'].append(model_name)
					results['Train Accuracy'].append(train_acc)
					results['Test Accuracy'].append(test_acc)
					results['Test Balanced Accuracy'].append(test_bal_acc)
					results['Test TN'].append(tn)
					results['Test FP'].append(fp)
					results['Test FN'].append(fn)
					results['Test TP'].append(tp)

		results = pd.DataFrame(results)
		results.to_csv(res_name, index = False)




		
	
		
					


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-config",help="The config file for these experiments")
	args = parser.parse_args()
	
	with open(args.config) as file:
		config = yaml.safe_load(file)

	main(config)
	
	