#############################################################
# Module: lookback.py										#
#															#
# Contains functions to find descendants, the time elapsed  #
# between galaxy merger and binary merger and the redshift  #
# at binary merger											#
#															#
#	--> lookback_function(z)								#
#	--> time_between_mergers(z1,z2)							#
#	--> find_descendant(k, tree_index)						#
#	--> find_redshift(z1,time_to_merge)						#
# 															#
# !Further information is provided below each function      #
#############################################################

import math
import numpy as np

import pandas as pd

catalogue = 'guo2013'

# Open data file with panda
data = pd.read_csv('Data/InputData/sel_starting_ordered_%s.csv' %catalogue)

# Store relevant data in individual arrays
galaxyId = data.iloc[:,0].values
DlastProgenitor = data.iloc[:,1].values
snapnum = data.iloc[:,2].values
Ddescendant = data.iloc[:,3].values
P1_galaxyId = data.iloc[:,4].values
P2_galaxyId = data.iloc[:,5].values
redshift = data.iloc[:,6].values

#############################################################
def lookback_function(z,omega_matter,omega_lambda):

	"""
	lookback_function evaluates the integrand at a given z

	input parameters:
		z -> the redshift value at which to evaluate the integrand
		omega_matter -> matter energy density
		omega_lambda -> dark energy density

	return:
		value (float) of the integrand, f
	"""

	f = 1/((1+z)*np.sqrt(omega_matter*(1+z)**3+omega_lambda))
	return f

def time_between_mergers(z1,z2,omega_matter,omega_lambda):

	"""
	time_between_mergers function computes the time
	elapsed between two subsequent galaxy mergers

	input parameters:
		z1 -> redshift of the subsequent galaxy merger
		z2 -> redshift of the previous galaxy merger
		
	return:
		value (float) of the time elapsed between 
		two subsequent galaxy mergers, in Gyr
	"""

	# Retrieve useful constants
	TH0 = 13.4 #(in Gyr)

	# Establish number of points to integrate and create
	# a uniformly spaced vector of z values
	N = 100
	dz_vector = np.linspace(z1,z2,num=N)
	sum = 0
	# Numerically integrate the lookback integral
	for i in range(len(dz_vector)):
		sum = sum + lookback_function(dz_vector[i],omega_matter,omega_lambda)
	sum = (z2-z1)*sum/N
	value = TH0*(sum)

	return value

#############################################################
def find_descendant(k,tree_index):

	"""
	find_descendant searches in the same catalogue of data
	as the main program (tree.py) to find the descendant 
	of a merger in a tree

	input parameters:
		k -> index in the catalogue at which the merger
			occurs
		tree_index -> index at which the corresponding tree
					 ends

	return:
		vector containing, respectively, the index (int)
		of the descendant, whether the descendant
		is the first progenitor (P1=1 and P2=0) or
		the second progenitor (P1=0 and P2=1) and
		the redshift of the descendant (-1 if there
		is no descendant, i.e. last merger of a tree)

		--> [descendant_index, P1, P2, z]
	"""


	# Use the same set of data analyzed in the main program, 
	# tree.py, to find descendants

	# Set to zero the descendant type (progenitor 1 or 2)
	P1 = 0
	P2 = 0
	descendant_index = -1 # deafult value if there is no descendant
	for l in range(k+1, tree_index):
		if((snapnum[l]-snapnum[k])>0):
			if ((snapnum[l]-snapnum[k])==(1+galaxyId[k]-P1_galaxyId[l])):
				P1 = 1
				z = redshift[l]
				descendant_index = l
				break 
			if ((snapnum[l]-snapnum[k])==(1+galaxyId[k]-P2_galaxyId[l])):
				P2 = 1
				z = redshift[l]
				descendant_index = l
				break
	if (descendant_index==-1):
		z = 0
	output = np.array([descendant_index,P1,P2,z])
	
	return output

#############################################################
def find_redshift(z,time_to_merge,omega_matter,omega_lambda):
	"""
	find_redshift searches for the redshift at which the binary
	merges, given the redshift of the galaxy merger and the 
	binary merger time

	input parameters:
		z -> redshift of the galaxy merger
		time_to_merge -> time delay between galaxy and binary
						 merger

	return:
		the value (float) of the redshift, z, at which the 
		binary merger occurs
	"""

	# Select number of points to integrate
	N = 100
	dz = np.linspace(z,0.0,num=N)
	i = 1
	time = time_between_mergers(dz[i],z,omega_matter,omega_lambda)
	# Search for the nearest redshift that gives the elapsed
	# time
	while (time<time_to_merge and i<(N-1)):
		i = i+1
		time = time_between_mergers(dz[i],z,omega_matter,omega_lambda)

	z = dz[i]
	
	return z	


def integrate_rate(z,omega_matter,omega_lambda):
	"""
	Function to calculate merger rates, it integrates
	dz/E(z) from z=0 to z

	input parameters:
		z -> redshift until to integrate
		omega_matter,omega_lambda -> cosmological parameters

	return:
		the value (float) of the integral
	"""

	N = 100
	dz_vector = np.linspace(0,z,num=N)
	sum = 0
	for i in range(len(dz_vector)):
		sum = sum+lookback_function(dz_vector[i],omega_matter,omega_lambda)
	sum = z*sum/N

	return sum

