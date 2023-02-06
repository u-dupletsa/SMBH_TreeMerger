#############################################################
# Module: delay.py
#
# Contains functions to calculate the contrubutions to the
# time delay between galaxy and binary merger
#
#   --> coulomb_logarithm(sigma1, sigma2)
#   --> eccentricity(e)
#   --> time_df_phase1(r_eff,stellar_mass,mass1,mass2)
#   --> time_df_phase2(r_eff,sigma1,sigma2,mass2)
#   --> time_star(sigma_inf,rho_inf,r_inf,mass1,
#                         mass2,mass_binary,e)
#   --> time_gas(mass1,mass2,mass_binary,m_dot,
#                         r_inf,e)
#   --> time_gw(a_in,mass1,mass2,mass_binary)
#   --> tot_delay_function(host_r_eff,host_sigma,
#                          satellite_sigma,satellite_BH,
#                          sigma_inf,rho_inf,r_inf,mass1,
#                          mass2,e,m_dot,stellar_mass,
#                          r_eff,hardening_type)
#   --> tot_delay_no_df(sigma_inf,rho_inf,r_inf,mass1,
#                       mass2,e,m_dot,hardening_type)
#
# !Further information is provided below each function
#############################################################

import numpy as np 
import constants as cst


#############################################################

def coulomb_logarithm(sigma1, sigma2):
	"""
	Function to calculate the coulomb logarithm used
	in the dynamical friction expression

	input parameters:
		sigma1 -> velocity dispersion at r_eff of the main
				progenitor galaxy (any units)
		sigma2 -> velocity dispersion at r_eff of the second
				progenitor galaxy (same units as sigma1)

	return:
		value (float) of the coulomb logarithm function
	"""

	if(sigma1 > sigma2):
		value = np.log(2**(3/2) * sigma1 / sigma2) # np.log is the natural logarithm
	else:
		value = np.log(2**(3/2) * sigma2 / sigma1)

	return value


def time_df_binney(stellar_mass, mass1, mass2):
	"""
	Function that evaluates the dynamical friction time
	delay, following the prescription by Binney&Tremaine
	This is used for df time in Horizon-AGN catalog

	input parameters:
		r_eff -> effectve radius (in pc) of the remnant 
				 galaxy
		stellar_mass -> remnant galaxy mass (in solar
						masses)
		mass1 -> first black hole mass (in solar masses)
		mass2 -> second black hole mass (in solar masses)

	return:
		dynamical friction time delay (in Gyr)
	"""
	r_eff = 4000
	coulomb_log1 = np.log10(1 + stellar_mass/mass1)
	coulomb_log2 = np.log10(1 + stellar_mass/mass2)
	sigma = ((0.25 * cst.G_new * stellar_mass/r_eff)**(1/2)) * cst.pc/10**3 # in km/s

	time_dyn_1 = 0.67 * (sigma/100) * (10**8/mass1)/coulomb_log1
	time_dyn_2 = 0.67 * (sigma/100) * (10**8/mass2)/coulomb_log2

	return np.max([time_dyn_1, time_dyn_2])


def time_df_phase1(r_eff, sigma1, sigma2, mass2):
	"""
	Function that evaluates the first phase of dynamical 
	friction contribution to time delay following the 
	prescription by Desopoulou and Antonini

	input parameters:
		r_eff -> effective radius of the main progenitor 
				 galaxy (in pc)
		sigma1 -> velocity dispersion of the main progenitor
			      galaxy within r_eff (in km/s)
		sigma2 -> velocity dispersion of the satellite progenitor
			      galaxy within r_eff (in km/s)
		mass2 -> mass of the satellite black hole (in solar
				 masses)

	return:
		dynamical friction time delay (in Gyr)
	"""
	coulomb_log = coulomb_logarithm(sigma1, sigma2)
	time_dyn_1 = 0.06 * 2/coulomb_log * (r_eff/(10**4))**2. * (sigma1/(300.)) * ((10**8.)/mass2)
	time_dyn_2 = 0.15 * 2/coulomb_log * (r_eff/(10**4)) * (sigma1/(300.))**2 * (100./sigma2)**3.

	return np.max([time_dyn_1, time_dyn_2])


def time_df_phase2(density_model, r_eff, r_inf, sigma1, sigma2, mass1, mass2):
	""" 
	Function that calculates the second phase of dynamical
	friction following Desopoulou and Antonini

	input parameters:
		r_eff -> effective radius of the main progenitor 
				 galaxy (in pc)
		r_inf -> influence radius of the binary (in pc)
		sigma1 -> velocity dispersion of the main progenitor
			      galaxy within r_eff (in km/s)
		sigma2 -> velocity dispersion of the satellite progenitor
			      galaxy within r_eff (in km/s)
		mass1 -> mass of the primary black hole (in solar masses)
		mass2 -> mass of the satellite black hole (in solar
				 masses)

	return:
		dynamical friction time delay for the second phase (in Gyr)

	"""

	coulomb_log = coulomb_logarithm(sigma1, sigma2)
	coulomb_log_primed = np.log(mass1/mass2)

	if(density_model == 'isothermal'):
		b = 0.5
		alpha = 0.5
		beta = 1.37
		delta = -0.85
		gamma = 2
	elif(density_model == 'dehnen'):
		b = 2.5
		alpha = 0.84
		beta = 0.54
		delta = -0.29
		gamma = 4

	chi = (mass2 / (2 * mass1))**(1 / (3 - gamma))

	time_bare = 0.015 * (coulomb_log_primed * alpha + beta + delta)**(-1) / ((1.5 - gamma) * (3 - gamma)) * (chi**(gamma - 1.5) - 1) * \
				(mass1 / (3 * 10**9))**(1/2) * (mass2 / (10**8))**(-1) * (r_inf / 300)**(3/2)
	time_gal = 0.012 * (coulomb_log * alpha + beta + delta)**(-1) / (3 - gamma)**2 * (chi**(gamma - 3) - 1) * (mass1/(3 * 10**9)) * (100/sigma2)**3

	return np.min([time_bare, time_gal])


#############################################################

def eccentricity(e):
	"""
	Function that evaluates the Peter-Matheus expression

	input parameter:
		e -> eccentricity of the orbit (0 <= e < 1)

	return:
		the value (float) of the function
	"""

	return 1/(1 - e**2)**(7/2) * (1 + 73/24 * e**2 + 37/96 * e**4)



#############################################################

def time_star(sigma_inf, rho_inf, r_inf, mass1, mass2, mass_binary):
	"""
	Function to calculate the time delay due to stellar hardening
 
 	input parameters:
		sigma_inf -> the velocity dispersion at the influence radius
      				 (in km/s)
		rho_inf -> the mass density at the influence radius (in solar
      			   masses per pc^3)
		mass1, mass2 -> the masses of the two black holes (in solar
      					masses)
		mass_binary -> the total mass of the binary (in solar masses)
		e -> the eccentricity of the binary, which is by default zero

	return:
		vector (float) containing the separation at which gravitational
		wave emission would overtake the process (in pc) and the stellar 
		hardening time delay (in Gyr)
			-> ([a_hard_gw, time_star])
	"""
	sigma_inf = sigma_inf * 10**3 / cst.pc # convert from km/s to pc/s
	a_hard_gw = (64 * cst.G_new**2 * sigma_inf * mass1 * mass2 * mass_binary * \
				eccentricity(cst.e) / (5 * cst.c_new**5 * cst.H * rho_inf))**(1/5)
	time_star = sigma_inf/(cst.G_new * cst.H * rho_inf) * (1 / a_hard_gw - 1 / r_inf) / cst.Gyr

	return ([a_hard_gw, time_star])



#############################################################

def time_gas(mass1, mass2, mass_binary, m_dot, r_inf):
	"""
	Function to calculate the time delay due to gaseous hardening,
	in case stellar hardening is not efficient enough

	input parameters:
		mass1, mass2 -> the masses of the two black holes (in solar
      					masses)
		mass_binary -> the total mass of the binary (in solar masses)
		m_dot -> accretion rate (in solar masses per second)
		r_inf -> influence radius of the remnant galaxy (in pc)
		e -> the eccentricity of the binary, which is by default zero

	return:
		vector (float) containing the separation at which gravitational
		wave emission would overtake the process (in pc) and the gaseous 
		hardening time delay (in Gyr)
			-> ([a_gas_gw, time_gas])
	"""
	constant = 16 * 2**(1/2) / 5 * cst.G_new**3 / cst.c_new**5 * eccentricity(cst.e)
	a_gas_gw = (constant * mass1**2 * mass2**2  / (m_dot))**(1/4)
	mu = mass1 * mass2 / mass_binary
	time_gas = (2**0.5 / 4 * mu / m_dot * np.log(r_inf / a_gas_gw)) / cst.Gyr

	return ([a_gas_gw, time_gas])


#############################################################

def time_gw(a_in, mass1, mass2, mass_binary):
	"""
	Function that evaluates the gravitational wave emission
	time delay

	input parameters:
		a_in -> initial orbital separation (in pc)
		mass1, mass2 -> the masses of the two black holes (in solar
      					masses)
		mass_binary -> the total mass of the binary (in solar masses)

	return:
		gravitational wave time delay (in Gyr)
	"""
	constant = 5. / 256. * cst.c_new**5 / cst.G_new**3 / eccentricity(cst.e)
	time_gw = (constant * a_in**4 / (mass1 * mass2 * mass_binary)) / cst.Gyr

	return time_gw


#############################################################

def tot_delay_function(density_model, host_r_eff, host_sigma, satellite_sigma, satellite_BH,
						sigma_inf, rho_inf, r_inf, mass1, mass2, m_dot, 
						stellar_mass, r_eff,hardening_type):
	"""
	Function that calculates the total delay time between galaxy and
	binary merger

	input parameters:
		host_r_eff -> effective radius of the host progenitor galaxy (in pc)
		host_sigma -> velocity dispersion at r_eff of the host progenitor
					  galaxy (in km/s)
		sigma_satellite -> velocity dispersion at r_eff of the satellite
						   galaxy (in km/s)
		satellite_BH -> satellite black hole mass (in solar masses)
		sigma_inf -> velocity dispersion at r_inf of the remnant galaxy
		             (in km/s)
		rho_inf -> density at r_inf of the remnant galaxy (in solar masses
				   pc^3)
		r_inf -> influence radius of the remnant galaxy (in pc)
		mass1, mass2 -> the masses of the two black holes (in solar
      					masses)
      	e -> the eccentricity of the binary, which is by default zero	
      	m_dot -> accretion rate (in solar masses per second)
		stellar_mass -> remnant galaxy mass (in solar masses)
		r_eff -> effective radius of the remnant galaxy (in pc)
		hardening_type -> integer value that indicates whether the 
						  hardening process could be both stellar and 
						  gaseous (hardening_type == 0), or stellar
						  only (hardening_type == 1)

	return:
		vector (float) containig the total delay time and the single
		contributions, in Gyr
			-> ([delay_time, df_phase1, df_phase2, stars, gas, gws])
	"""

	mass_binary = mass1 + mass2

	df_phase1 = time_df_phase1(host_r_eff, host_sigma, satellite_sigma, satellite_BH)
	df_phase2 = time_df_phase2(density_model, host_r_eff, r_inf, host_sigma, satellite_sigma, mass1, mass2)

	if(hardening_type == 0): # Both stellar and gaseous hardening
		a_stars_gw, stars = time_star(sigma_inf, rho_inf, r_inf, mass1, mass2, mass_binary)
		a_gas_gw, gas = time_gas(mass1, mass2, mass_binary, m_dot, r_inf)

		if(stars < gas):
			gws = time_gw(a_stars_gw, mass1, mass2, mass_binary)
			delay_time = df_phase1 + df_phase2 + stars + gws
		else:
			gws = time_gw(a_gas_gw, mass1, mass2, mass_binary)
			delay_time = df_phase1 + df_phase2 + gas + gws

	else:
		a_stars_gw, stars = time_star(sigma_inf, rho_inf, r_inf, mass1, mass2, mass_binary)
		gas = 0.
		gws = time_gw(a_stars_gw, mass1, mass2, mass_binary)
		
		delay_time = df_phase1 + df_phase2 + stars + gws

	return ([delay_time, df_phase1, df_phase2, stars, gas, gws])
	

def tot_delay_no_df(sigma_inf, rho_inf, r_inf, mass1, mass2, m_dot, hardening_type):
	"""
	Function that evaluates the delay time due to the hardening and gw
	coalescence processes (no dynamical friction)

	input parameters:
		sigma_inf -> velocity dispersion at r_inf of the remnant galaxy
		             (in km/s)
		rho_inf -> density at r_inf of the remnant galaxy (in solar masses
				   pc^3)
		r_inf -> influence radius of the remnant galaxy (in pc)
		mass1, mass2 -> the masses of the two black holes (in solar
      					masses)
      	e -> the eccentricity of the binary, which is by default zero	
      	m_dot -> accretion rate (in solar masses per second)
      	hardening_type -> integer value that indicates whether the 
						  hardening process could be both stellar and 
						  gaseous (hardening_type == 0), or stellar
						  only (hardening_type == 1)

		return:
			vector (float) containing the total delay time (hardening+gw)
			and the single contributions (in Gyr)
				-> ([delay_time,t_star,t_gas,t_gw])
	"""

	mass_binary = mass1 + mass2

	if(hardening_type == 0):

		a_stars_gw, stars = time_star(sigma_inf, rho_inf, r_inf, mass1, mass2, mass_binary)
		a_gas_gw, gas = time_gas(mass1, mass2, mass_binary, m_dot, r_inf)

		if(stars < gas):
			gws = time_gw(a_stars_gw, mass1, mass2, mass_binary)
			delay_time = stars + gws
		else:
			gws = time_gw(a_gas_gw, mass1, mass2, mass_binary)
			delay_time = gas + gws

	else:

		a_stars_gw, stars = time_star(sigma_inf, rho_inf, r_inf, mass1, mass2, mass_binary)
		gas = 0.
		gws = time_gw(a_stars_gw, mass1, mass2, mass_binary)
		
		delay_time = stars + gws

	return ([delay_time, stars, gas, gws])

	