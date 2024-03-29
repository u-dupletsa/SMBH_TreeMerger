#############################################################
# Module: density_profile.py
#
# Contains functions to find quantities describing the
# galaxy, depending on the density profile model
#
#    --> effective_radius(bulge_mass,stellar_mass,redshift)
#    --> influence_radius(r_eff,stellar_mass,binary_mass)
#    --> rho_inf(stellar_mass,r_inf,r_eff)
#    --> sigma(stellar_mass,r_eff)
#    --> sigma_inf(binary_mass,r_inf)
#
# !Further information is provided below each function
#############################################################

import numpy as np 
import constants as cst
import math

import scipy
from scipy.integrate import quad


#############################################################
#############################################################
#############################################################
# Set values to variables, as refering to Van der Wel paper
# Paper reference Arxiv:1404.2844

z = ([0.0, 0.25, 0.75, 1.25, 1.75, 2.25, 10.0])

log_A_ellipticals = ([0.60, 0.42, 0.22, 0.09, -0.05, -0.06])
log_A_spirals = ([0.86, 0.78, 0.70, 0.65, 0.55, 0.51])

alpha_ellipticals = ([0.75, 0.71, 0.76, 0.76, 0.76, 0.79])
alpha_spirals = ([0.25, 0.22, 0.22, 0.23, 0.22, 0.18])

A_ellipticals = np.exp(log_A_ellipticals)
A_spirals = np.exp(log_A_spirals)

#############################################################
#############################################################
#############################################################


def effective_radius(bulge_mass, stellar_mass, redshift):
	"""
	Computes the effective radius of a galaxy

	input parameters:
		bulge_mass -> bulge mass of a galaxy (in solar masses)
		stellar_mass -> total mass of a galaxy (in solar masses)
		redshift -> redshift of the galaxy

	return:
		value (float) of the effective radius (in pc)
	"""
	if (bulge_mass / stellar_mass >= 0.7): # elliptical!
		for i in range(len(z) - 1):
			if(redshift < z[i+1] and redshift >= z[i]):
				r_eff = (A_ellipticals[i] * (stellar_mass / (5 * 10**(10.)))**alpha_ellipticals[i])
				r_eff = r_eff * 10**3 # in pc
	else: # spiral!
		for i in range(len(z) - 1):
			if(redshift < z[i+1] and redshift >= z[i]):
				r_eff = (A_spirals[i] * (stellar_mass / (5 * 10**(10.)))**alpha_spirals[i])
				r_eff = r_eff * 10**3 # in pc
		
	return r_eff # in pc



def scale_radius(r_eff):
	"""
	Computes the scale radius of the model

	input parameters:
		r_eff -> effective radius of the galaxy, in pc
		gamma -> parameter for Dehnen profile

	return:
		scale_radius in pc
	"""

	return 4/3 * (2**(1/(3 - cst.gamma)) - 1) * r_eff



def influence_radius(density_model, r_eff, stellar_mass, binary_mass):
	"""
	Function that computes the influence radius of a binary

	input parameters:
		density_model -> isothermal or dehnen
		r_eff -> effective radius of the galaxy, in pc
		stellar_mass -> total mass of the galaxy (in solar masses)
		binary_mass -> total mass of the binary (in solar masses)

	return:
		value (float) of the influence radius (in pc)
	"""

	if (density_model == 'isothermal'):
		r_inf = r_eff * (4 * binary_mass / stellar_mass)

	elif (density_model == 'dehnen'):
		r_scale = scale_radius(r_eff)
		r_inf = r_scale / ((stellar_mass / (2 * binary_mass))**(1/(3 - cst.gamma)) - 1)


	return r_inf # in pc

def rho(density_model, stellar_mass, r_eff, r):
	"""
	Function that computes the density as function of radius

	input parameters:
		density_model -> isothermal or dehnen
		stellar_mass -> total mass of the galaxy (in solar masses)
		r_inf -> influence radius of the binary (in pc)
		r -> radius at which compute rho (in pc)

	return:
		value (float) of the density in solar masses per pc^3 at r
	"""

	if (density_model == 'isothermal'):
		rho = stellar_mass / (8 * np.pi * r**2 * r_eff)

	elif (density_model == 'dehnen'):
		r_scale = scale_radius(r_eff)
		rho = (3 - cst.gamma) / (4 * np.pi) * stellar_mass * r_scale / (r**cst.gamma * (r + r_scale)**(4 - cst.gamma))
	

	return rho # in M_sun/(pc^3)



def rho_inf(density_model, stellar_mass, r_eff, r_inf):
	"""
	Function that computes the density within the influence radius

	input parameters:
		density_model -> isothermal or dehnen
		stellar_mass -> total mass of the galaxy (in solar masses)
		r_inf -> influence radius of the binary (in pc)
		r_eff -> effective radius of the galaxy (in pc)

	return:
		value (float) of the density in solar masses per pc^3
	"""

	if (density_model == 'isothermal'):
		rho = stellar_mass / (8 * np.pi * r_inf**2 * r_eff)

	elif (density_model == 'dehnen'):
		r_scale = scale_radius(r_eff)
		rho = (3 - cst.gamma) / (4 * np.pi) * stellar_mass * r_scale / (r_inf**cst.gamma * (r_inf + r_scale)**(4 - cst.gamma))
	

	return rho # in M_sun/(pc^3)


def integrand_hernquist(r, r_scale, stellar_mass): # with r=r_inf gives sigma_inf!
	"""
	Computes the velocity dispersion
	"""
	integrand = cst.G_new * stellar_mass / (12 * r_scale) * (12 * r * (r + r_scale)**3 /\
				(r_scale)**4 * np.log((r + r_scale) / r) - r / (r + r_scale) * (25 + 52 * r / r_scale + \
				42 * (r / r_scale)**2 + 12 * (r / r_scale)**3))

	return integrand # in pc^2/(s^2)


def sigma(density_model, stellar_mass, r_eff, r_inf):
	"""
	Function that computes the velocity dispersion at the effective 
	radius

	input parameters:
		density_model -> isothermal or dehnen
		stellar_mass -> total mass of the galaxy (in solar masses)
		r_eff -> effective radius of the galaxy (in pc)

	return:
		value (float) of the velocity dispersion in km/s
	"""

	if (density_model == 'isothermal'):
		sigma = (cst.G_new * stellar_mass / (4 * r_eff))**(1/2) * cst.pc / 10**3

	elif (density_model == 'dehnen'):
		r_scale = scale_radius(r_eff)
		sigma, precision = quad(integrand_hernquist, r_inf, r_eff, args=(r_scale, stellar_mass))
		sigma = (sigma / (r_eff - r_inf))**(1/2) * cst.pc/10**3	


	return sigma # in km/s


def sigma_inf(density_model, binary_mass, stellar_mass, r_eff, r_inf):
	"""
	Function that computes the velocity dispersion at the influence 
	radius

	input parameters:
		density_model -> isothermal or dehnen
		stellar_mass -> total mass of the galaxy (in solar masses)
		r_inf -> influence radius of the binary (in pc)

	return:
		value (float) of the velocity dispersion in km/s
	"""

	if (density_model == 'isothermal'):
		sigma_inf_value = (cst.G_new * binary_mass / (4 * r_inf))**(1/2) * cst.pc/10**3

	elif (density_model == 'dehnen'):
		gamma = cst.gamma
		r_scale = scale_radius(r_eff)
		sigma_inf_value = integrand_hernquist(r_inf, r_scale, stellar_mass)
		sigma_inf_value = (sigma_inf_value)**(1/2) * cst.pc/10**3


	return sigma_inf_value # in km/s


