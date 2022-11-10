import numpy as np
import random

import pandas as pd

import time
from tqdm import tqdm

import tree

#################################################################
#################################################################
# Select catalogs and cosmology
#catalog = ['de_lucia', 'bertone', 'guo2010',' guo2013']
catalog = 'bertone'
mass_model = 'KH'
density_model = 'isothermal'
omega_matter = 0.272
omega_lambda = 0.728
h = 0.704
#################################################################
#################################################################

data_folder = 'Data/InputData'

lbs = ['galaxyId', 'lastProgenitorId', 'snapnum', 'descendantId', 'P1_Id', 'P2_Id', 'D_z',
	   'D_mass', 'D_bulge', 'sfr', 'sfr_bulge', 'D_BH', 'P1_z', 'P2_z', 'M1', 'M2', 
	   'P1_bulge', 'P2_bulge', 'P1_stars', 'P2_stars', 'M_cold', 'M_hot', 'V_vir', 
	   'P1_M_cold', 'P1_M_hot', 'P1_V_vir', 'P2_M_cold', 'P2_M_hot', 'P2_V_vir',
	   'bh_mass', 'P1_BH_mass', 'P2_BH_mass', 'q', 'mass1', 'mass2', 'r_eff_P1',
	   'r_inf_P1', 'sigma_P1', 'r_eff_P2', 'r_inf_P2', 'sigma_P2', 'host_r_eff', 'host_sigma', 
	   'satellite_sigma', 'satellite_BH', 'host_BH', 'r_eff', 'r_inf', 'sigma_inf', 'rho_inf',
	   'm_dot', 'hardening_type']

data = pd.read_csv('%s/injection_%s_%s_%s.csv' %(str(data_folder), str(catalog), str(mass_model), str(density_model)),
					    names = lbs, skiprows = 1, delimiter = ',')

tree.tree(catalog, omega_matter, omega_lambda, data)