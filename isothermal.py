#############################################################
# Module: isothermal.py										#
#															#
# Contains functions to find quantities describing the 		#
# galaxy, modeled as an isothermal singular sphere			#
#															#
#	--> effective_radius(bulge_mass,stellar_mass,redshift)  #
#	--> influence_radius(r_eff,stellar_mass,binary_mass)	#
#	--> rho_inf(stellar_mass,r_inf,r_eff)					#
#	--> sigma(stellar_mass,r_eff)							#
#	--> sigma_inf(binary_mass,r_inf)						#						
# 															#
# !Further information is provided below each function      #
#############################################################

import numpy as np 
import math


# constants
M_sun = 1.99*10**(30)
G = 6.67*10**(-11)
G_new = G*1.99/(3.086)**3*10**(-18) # in pc^3/(s^2*M_sun)
pc = 3.086*10**(16)

#############################################################
# Set values to variables, as refering to Van der Wel paper
# Paper reference Arxiv:1404.2844

z=([0.0,0.25,0.75,1.25,1.75,2.25,10.0])

log_A_ellipticals=([0.60,0.42,0.22,0.09,-0.05,-0.06])
log_A_spirals=([0.86,0.78,0.70,0.65,0.55,0.51])

alpha_ellipticals=([0.75,0.71,0.76,0.76,0.76,0.79])
alpha_spirals=([0.25,0.22,0.22,0.23,0.22,0.18])

A_ellipticals=np.zeros(len(log_A_ellipticals))
A_spirals=np.zeros(len(log_A_spirals))

for i in range(len(A_spirals)):
	A_ellipticals[i]=math.exp(log_A_ellipticals[i])
	A_spirals[i]=math.exp(log_A_spirals[i])

#############################################################

def effective_radius(bulge_mass,stellar_mass,redshift):
	"""
	Computes the effective radius of a galaxy

	input parameters:
		bulge_mass -> bulge mass of a galaxy (in solar masses)
		stellar_mass -> total mass of a galaxy (in solar masses)
		redshift -> redshift of the galaxy

	return:
		value (float) of the effective radius (in pc)
	"""

	if (bulge_mass/stellar_mass >= 0.7): # elliptical!
		for i in range(len(z)-1):
			if(redshift<z[i+1] and redshift>=z[i]):
				r_eff = (A_ellipticals[i]*(stellar_mass/(5*10**(10)))**alpha_ellipticals[i])
				r_eff = r_eff*10**3 # in pc
	else: # spiral!
		for i in range(len(z)-1):
			if(redshift<z[i+1] and redshift>=z[i]):
				r_eff = (A_spirals[i]*(stellar_mass/(5*10**(10)))**alpha_spirals[i])
				r_eff = r_eff*10**3 # in pc
		
	return r_eff # in pc


def influence_radius(r_eff,stellar_mass,binary_mass):
	"""
	Function that computes the influence radius of a binary

	input parameters:
		r_eff -> effective radius of the galaxy, in pc
		stellar_mass -> total mass of the galaxy (in solar masses)
		binary_mass -> total mass of the binary (in solar masses)

	return:
		value (float) of the influence radius (in pc)
	"""
	r_inf=r_eff*(4*binary_mass/stellar_mass)

	return r_inf # in pc


def rho_inf(stellar_mass,r_inf,r_eff):
	"""
	Function that computes the density within the influence radius

	input parameters:
		stellar_mass -> total mass of the galaxy (in solar masses)
		r_inf -> influence radius of the binary (in pc)
		r_eff -> effective radius of the galaxy (in pc)

	return:
		value (float) of the density in solar masses per pc^3
	"""
	rho=stellar_mass/(8*math.pi*r_inf**2*r_eff)
	
	return rho # in M_sun/(pc^3)

def sigma(stellar_mass,r_eff):
	"""
	Function that computes the velocity dispersion at the effective 
	radius

	input parameters:
		stellar_mass -> total mass of the galaxy (in solar masses)
		r_eff -> effective radius of the galaxy (in pc)

	return:
		value (float) of the velocity dispersion in km/s
	"""
	sigma_value=(G_new*stellar_mass/(4*r_eff))**(1/2)*pc/10**3

	return sigma_value # in km/s

def sigma_inf(binary_mass,r_inf):
	"""
	Function that computes the velocity dispersion at the influence 
	radius

	input parameters:
		stellar_mass -> total mass of the galaxy (in solar masses)
		r_inf -> influence radius of the binary (in pc)

	return:
		value (float) of the velocity dispersion in km/s
	"""
	sigma_inf_value=(G_new*binary_mass/(4*r_inf))**(1/2)*pc/10**3

	return sigma_inf_value # in km/s
