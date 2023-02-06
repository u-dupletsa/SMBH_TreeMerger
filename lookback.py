#############################################################
# Module: lookback.py
#
# Contains functions to find descendants, the time elapsed
# between galaxy merger and binary merger and the redshift
# at binary merger
#
#    --> lookback_function(z,omega_matter,omega_lambda)
#    --> time_between_mergers(z1,z2,omega_matter,omega_lambda)
#    --> find_descendant(k,tree_index,snapnum,galaxyId,P1_galaxyId,
#		P2_galaxyId,redshift)
#    --> find_redshift(z,time_to_merge,omega_matter,omega_lambda)
#    --> integrate_rate(z,omega_matter,omega_lambda)
#
# !Further information is provided below each function
#############################################################

import time
import math
import numpy as np
import scipy
from scipy.integrate import quad

#############################################################
def lookback_function(z, omega_matter, omega_lambda):

	"""
	lookback_function evaluates the integrand at a given z

	input parameters:
		z -> the redshift value at which to evaluate the integrand
		omega_matter -> matter energy density
		omega_lambda -> dark energy density

	return:
		value (float) of the integrand, f
	"""

	return 1 / ((1 + z) * np.sqrt(omega_matter * (1 + z)**3 + omega_lambda))

def rate_function(z, omega_matter, omega_lambda):

	"""
	lookback_function evaluates the integrand at a given z

	input parameters:
		z -> the redshift value at which to evaluate the integrand
		omega_matter -> matter energy density
		omega_lambda -> dark energy density

	return:
		value (float) of the integrand, f
	"""

	return 1 / (np.sqrt(omega_matter * (1 + z)**3 + omega_lambda))

def time_between_mergers(z1, z2, omega_matter, omega_lambda):

	"""
	time_between_mergers function computes the time
	elapsed between two subsequent galaxy mergers

	input parameters:
		z1 -> redshift of the subsequent galaxy merger
		z2 -> redshift of the previous galaxy merger
		omega_matter -> matter energy density
		omega_lambda -> dark energy density
		
	return:
		value (float) of the time elapsed between 
		two subsequent galaxy mergers, in Gyr
	"""

	integral, precision = quad(lookback_function, z1, z2, args=(omega_matter, omega_lambda))

	return integral

#############################################################
def find_descendant(k, tree_index, snapnum, galaxyId, P1_galaxyId, P2_galaxyId, redshift):

	"""
	find_descendant searches in the same catalogue of data
	as the main program (tree.py) to find the descendant 
	of a merger in a tree

	input parameters:
		k -> index in the catalogue at which the merger
			occurs
		tree_index -> index at which the corresponding tree
					 ends
		snapnum -> snapnum corresponding to the merger
		galaxyId -> Id of the remnant galaxy
		P1_galaxyId -> Id of progenitor 1
		P2_galaxyId -> Id of progenitor 2
		redshift -> redshift of merger

	return:
		vector containing, respectively, the index (int)
		of the descendant, whether the descendant
		is the first progenitor (P1=1 and P2=0) or
		the second progenitor (P1=0 and P2=1) and
		the redshift of the descendant (-1 if there
		is no descendant, i.e. last merger of a tree)

		--> [descendant_index, P1, P2, z]
	"""

	# Set to zero the descendant type (progenitor 1 and 2)
	P1 = 0
	P2 = 0
	descendant_index = -1 # deafult value if there is no descendant
	for l in range(k + 1, tree_index):
		if((snapnum[l] - snapnum[k]) > 0):
			if ((snapnum[l] - snapnum[k]) == (1 + galaxyId[k] - P1_galaxyId[l])):
				P1 = 1
				z = redshift[l]
				descendant_index = l
				break 
			if ((snapnum[l] - snapnum[k]) == (1 + galaxyId[k] - P2_galaxyId[l])):
				P2 = 1
				z = redshift[l]
				descendant_index = l
				break
	if (descendant_index == -1):
		z = 0
	
	return int(descendant_index), int(P1), int(P2), z

#############################################################
def find_redshift(z, time_to_merge, omega_matter, omega_lambda):
	"""
	find_redshift searches for the redshift at which the binary
	merges, given the redshift of the galaxy merger and the 
	binary merger time

	input parameters:
		z -> redshift of the galaxy merger
		time_to_merge -> time delay between galaxy and binary
						 merger
		omega_matter -> matter energy density
		omega_lambda -> dark energy density

	return:
		the value (float) of the redshift, z, at which the 
		binary merger occurs
	"""

	# Select number of points to integrate
	spacing = 0.005
	N = int(z / spacing)
	if (N == 1 or N == 0):
		N = 5
		
	dz = np.linspace(z, 0., num = N)
	i = 1
	time = time_between_mergers(dz[i], z, omega_matter, omega_lambda)
	# Search for the nearest redshift that gives the elapsed
	# time
	while ((time_to_merge - time) / time_to_merge > 0.01 and i < (N - 1)):
		i = i+1
		time = time_between_mergers(dz[i], z, omega_matter, omega_lambda)

	return dz[i]


def integrate_rate(z, omega_matter, omega_lambda):
	"""
	Function to calculate merger rates, it integrates
	dz/E(z) from z=0 to z

	input parameters:
		z -> redshift until to integrate
		omega_matter -> matter energy density
		omega_lambda -> dark energy density

	return:
		the value (float) of the integral
	"""

	integral, precision = quad(rate_function, 0., z, args=(omega_matter, omega_lambda))

	return integral

