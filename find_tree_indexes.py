#############################################################
# Module: find_tree_indexes.py
#
# Searches for start-end indexes of trees of an input 
# catalog.
#
#    Input file: --> injection_<catalog>_<mass_model>_<density_model>.csv
#    Output file:--> tree_indexes_<catalog>.csv
#
#############################################################

import numpy as np
import pandas as pd

import time
from tqdm import tqdm

import tree


def find_indexes(catalog, mass_model, density_model):

	lbs = ['galaxyId', 'lastProgenitorId', 'snapnum', 'descendantId', 'P1_Id', 'P2_Id', 'D_z',
		   'D_mass', 'D_bulge', 'sfr', 'sfr_bulge', 'D_BH', 'P1_z', 'P2_z', 'M1', 'M2', 
		   'P1_bulge', 'P2_bulge', 'P1_stars', 'P2_stars', 'M_cold', 'M_hot', 'V_vir', 
		   'P1_M_cold', 'P1_M_hot', 'P1_V_vir', 'P2_M_cold', 'P2_M_hot', 'P2_V_vir',
		   'bh_mass', 'P1_BH_mass', 'P2_BH_mass', 'q', 'mass1', 'mass2', 'r_eff_P1',
		   'r_inf_P1', 'sigma_P1', 'r_eff_P2', 'r_inf_P2', 'sigma_P2', 'host_r_eff', 'host_sigma', 
		   'satellite_sigma', 'satellite_BH', 'host_BH', 'r_eff', 'r_inf', 'sigma_inf', 'rho_inf',
		   'm_dot', 'hardening_type']

	data = pd.read_csv('Data/InputData/injection_%s_%s_%s.csv' %(str(catalog), str(mass_model), str(density_model)),
						    names = lbs, skiprows = 1, delimiter = ',')

	tree_start = []
	tree_end = []

	galaxyId = data['galaxyId'].copy()

	i = 0
	while i < len(galaxyId):
		j = i 
		tree_start.append(j)
		while (galaxyId[j] != -1): # still the same tree
			j = j + 1
			if (j >= (len(galaxyId))): # in case the data set has ended
				break

		tree_index = j
		tree_end.append(tree_index)
		tree_length = tree_index - i


		i = i + tree_length + 1

		if (i >= len(galaxyId)):
			break

	tree_index_data = {}
	tree_index_data['start'] = tree_start
	tree_index_data['end'] = tree_end

	df = pd.DataFrame(data = tree_index_data)

	df.to_csv('Data/InputData/tree_indexes_%s.csv' %str(catalog), index=False) 

