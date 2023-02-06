#############################################################
# MAIN program
#
# Launches the analisys of merger trees of a given catalog
#
#############################################################

import numpy as np
import pandas as pd

import os

import time
from tqdm import tqdm

import tree
import model
import find_tree_indexes

import yaml

#################################################################
#################################################################
# Select catalog and model
#catalogs = ['de_lucia', 'bertone', 'guo2010', 'guo2013', 'horizon']
catalog = 'guo2013'
mass_model = 'KH' # KH or millennium
density_model = 'isothermal' # isothermal or dehnen

print('Analysing %s catalog with masses modeled as %s and %s density profile' 
		%(str(catalog),str(mass_model),str(density_model)))
#################################################################
#################################################################

with open('settings.yaml') as f:
    doc = yaml.load(f, Loader=yaml.FullLoader)

np.random.seed(0)

data_folder = 'Data/InputData'

lbs = ['galaxyId', 'lastProgenitorId', 'snapnum', 'descendantId', 'P1_Id', 'P2_Id', 'D_z',
	   'D_mass', 'D_bulge', 'sfr', 'sfr_bulge', 'D_BH', 'P1_z', 'P2_z', 'M1', 'M2', 
	   'P1_bulge', 'P2_bulge', 'P1_stars', 'P2_stars', 'M_cold', 'M_hot', 'V_vir', 
	   'P1_M_cold', 'P1_M_hot', 'P1_V_vir', 'P2_M_cold', 'P2_M_hot', 'P2_V_vir',
	   'bh_mass', 'P1_BH_mass', 'P2_BH_mass', 'q', 'mass1', 'mass2', 'r_eff_P1',
	   'r_inf_P1', 'sigma_P1', 'r_eff_P2', 'r_inf_P2', 'sigma_P2', 'host_r_eff', 'host_sigma', 
	   'satellite_sigma', 'satellite_BH', 'host_BH', 'r_eff', 'r_inf', 'sigma_inf', 'rho_inf',
	   'm_dot', 'hardening_type']

catalog_properties = doc[catalog]
h = eval(str(catalog_properties['h']))
omega_matter = eval(str(catalog_properties['omega_matter']))
omega_lambda = eval(str(catalog_properties['omega_lambda']))

# Injection files paths
path_data = '%s/injection_%s_%s_%s.csv' %(str(data_folder), str(catalog), str(mass_model), str(density_model))
path_index = '%s/tree_indexes_%s.csv' %(str(data_folder), str(catalog))

# Check whether the specified
# path exists or not
if os.path.exists(path_data):
	data = pd.read_csv(path_data, names = lbs, skiprows = 1, delimiter = ',')
	print('Opening data file')
else:
	print('Data file does not exist, generating file')
	model.generate_input(catalog, mass_model, density_model, h)
	data = pd.read_csv(path_data, names = lbs, skiprows = 1, delimiter = ',')
	print('Opening data file')

if os.path.exists(path_index):
	index_data = pd.read_csv(path_index, names = ['start', 'end'], skiprows = 1, delimiter = ',')
	print('Opening index file')
else:
	print('Index file does not exist, generating file')
	find_tree_indexes.find_indexes(catalog, mass_model, density_model)
	index_data = pd.read_csv(path_index, names = ['start', 'end'], skiprows = 1, delimiter = ',')
	print('Opening index file')

tree_start = index_data['start'].copy()
tree_end = index_data['end'].copy()

print('Launching tree analysis')
tree.tree(catalog, density_model, mass_model, omega_matter, omega_lambda, data, tree_start, tree_end)
