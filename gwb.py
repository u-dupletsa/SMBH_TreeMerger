import math
import numpy as np
import random

import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

#############################################################
#############################################################
catalogue = 'bertone'
omega_matter = 0.25
omega_lambda = 0.75
h = 0.73
#############################################################
#############################################################


def chirp(mass1,mass2):
	"""
	Function that computes the chirp mass

	input parameters:
		mass1,mass2 -> component masses of the binary

	return:
		the value (float) of the chirp mass
	"""
	chirp_mass = (mass1*mass2)**(3/5)/(mass1 + mass2)**(1/5)

	return chirp_mass


# useful constants
M_sun = 1.99*10**(30)
pi = math.pi
Gyr = 3.15*10**16
c = 3.0*10**8
pc = 3.086*10**(16)
G = 6.67*10**(-11)
freq_1yr = 3.17098*10**(-8)
t_1yr = 3.15*10**7
TH0 = 13.4*Gyr
H0 = 100*h*10**3/(10**6*pc)
constant = 1/((500.0/h)*(10**6*pc))**3*c**3/(H0**2)*4*pi


data = pd.read_csv('Data/OutputData/TreeAnalysis/output_%s.csv' %catalogue)
mass1 = data.iloc[:,0].values
mass2 = data.iloc[:,1].values
binary = data.iloc[:,2].values
triplet = data.iloc[:,3].values
ejection = data.iloc[:,4].values
quadruplets = data.iloc[:,5].values
failed = data.iloc[:,6].values
failed_prompt = data.iloc[:,7].values
failed_ejection = data.iloc[:,8].values
failed_failed = data.iloc[:,9].values
redshift = data.iloc[:,10].values
redshift_merger = data.iloc[:,11]

mass1_vec = []
mass2_vec = []
binary_vec = []
triplet_vec = []
ejection_vec = []
quadruplets_vec = []
failed_vec = []
failed_prompt_vec = []
failed_ejection_vec = []
failed_failed_vec = []
redshift_vec = []
redshift_merger_vec = []

for i in range(len(mass1)):
	if(mass1[i] != -1):
		mass1_vec.append(mass1[i])
		mass2_vec.append(mass2[i])
		binary_vec.append(binary[i])
		triplet_vec.append(triplet[i])
		ejection_vec.append(ejection[i])
		quadruplets_vec.append(quadruplets[i])
		failed_vec.append(failed[i])
		failed_prompt_vec.append(failed_prompt[i])
		failed_failed_vec.append(failed_ejection[i])
		failed_failed_vec.append(failed_failed[i])
		redshift_vec.append(redshift[i])
		redshift_merger_vec.append(redshift_merger[i])




dn = 1/((500.0/h)*(pc*10**6))**3

h_c_delay = 0
#For cycle over mergers to compute the strain
for i in range(len(mass1_vec)):
	if(binary_vec[i] == 1 or triplet_vec[i] == 1 or ejection_vec[i] == 1):
		chirp_mass = chirp(mass1_vec[i]*M_sun, mass2_vec[i]*M_sun)
		redshift_bin = redshift_merger_vec[i]
		h_c_delay = chirp_mass**(5/3)/(1+redshift_bin)**(1/3) + h_c_delay

constant_h_c = 4*G**(5/3)/(3*pi**(1/3)*freq_1yr**(4/3)*c**2)*dn
strain_delay = h_c_delay*constant_h_c
strain_delay = strain_delay**(1/2)

print('gwb delay %s' %catalogue, strain_delay)


h_c_nodelay = 0
#For cycle over mergers to compute the strain
for i in range(len(mass1_vec)):
	chirp_mass = chirp(mass1_vec[i]*M_sun, mass2_vec[i]*M_sun)
	redshift_gal = redshift_vec[i]
	h_c_nodelay = chirp_mass**(5/3)/(1+redshift_gal)**(1/3) + h_c_nodelay

strain_nodelay = h_c_nodelay*constant_h_c
strain_nodelay = strain_nodelay**(1/2)

print('gwb no-delay %s' %catalogue, strain_nodelay)


print('total number of mergers', len(mass1_vec))
print('total binaries', sum(binary_vec))
print('total triplet', sum(triplet_vec))
print('total ejections', sum(ejection_vec))
print('total quadruplets', sum(quadruplets_vec))
print('total failed', sum(failed_vec))
print('total failed_prompt', sum(failed_prompt_vec))
print('total failed_ejection', sum(failed_ejection_vec))
print('total failed_failed', sum(failed_failed_vec))










