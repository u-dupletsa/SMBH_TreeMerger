#############################################################
# Module: delay.py                                          #
#                                                           #
# Contains functions to calculate the contrubutions to the  #
# time delay between galaxy and binary merger               #
#                                                           #
#   --> coulomb_logarithm(sigma1, sigma2)                   #
#   --> eccentricity(e)                                     #
#   --> time_df1(r_eff,stellar_mass,mass1,mass2)            #
#   --> time_df2(r_eff,sigma1,sigma2,mass2)                 #
#   --> time_star(sigma_inf,rho_inf,r_inf,mass1,            #
#                         mass2,mass_binary,e)              #
#   --> time_gas(mass1,mass2,mass_binary,m_dot,             #
#                         r_inf,e)                          #
#   --> time_gw(a_in,mass1,mass2,mass_binary)               #
#   --> tot_delay_function(host_r_eff,host_sigma,           #
#                          satellite_sigma,satellite_BH,    #
#                          sigma_inf,rho_inf,r_inf,mass1,   #
#                          mass2,e,m_dot,stellar_mass,      #
#                          r_eff,hardening_type)            #
#   --> tot_delay_no_df(sigma_inf,rho_inf,r_inf,mass1,      #
#                       mass2,e,m_dot,hardening_type)       #
#                                                           #
# !Further information is provided below each function      #
#############################################################

import numpy as np 
import math

# constants (in kms)
M_sun = 1.99*10**(30)
G = 6.67*10**(-11)
pc = 3.086*10**(16)
c = 3.0*10**8
freq_1yr = 3.17098*10**(-8)
h = 0.73
H = 15
t_1yr = 3.15*10**7
Gyr = 3.15*10**16
acc_rate = 0.1
sigma_thompson = 6.65*10**(-29)
m_protone = 1.67*10**(-27)
gamma = 1
e = 0
p = np.pi 

# To convert from kms in solar masses and parsec
length_conv = 0.324*10**(-16)
mass_conv = 0.5025*10**(-30)
# [G] = [L]^3 [M]^(-1) [T]^(-2)
# [c] = [L] [T]
G_new = G*(length_conv)**3/mass_conv
c_new = c*length_conv


#############################################################

def coulomb_logarithm(sigma1, sigma2):
	"""
	Function to calculate the coulomb logarithm used
	in the dynamical friction expression

	input paramters:
		sigma1 -> velocity dispersion at r_eff of the main
				progenitor galaxy
		sigma2 -> velocity dispersion at r_eff of the second
				progenitor galaxy

	return:
		value (float) of the coulomb logarithm function
	"""

	if(sigma1 > sigma2):
		value = np.log(2**(2/3)*sigma1/sigma2)
	else:
		value = np.log(2**(2/3)*sigma2/sigma1)

	return value


def time_df1(stellar_mass,mass1,mass2):
	"""
	Function that evaluates the dynamical friction time
	delay, following the prescription by Binney&Tremaine

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
	coulomb_log1 = math.log10(1+stellar_mass/mass1)
	coulomb_log2 = math.log10(1+stellar_mass/mass2)
	sigma = ((0.25*G_new*stellar_mass/r_eff)**(1/2))*pc/10**3 # in km/s


	time_dyn_1 = 0.67*(sigma/100)*(10**8/mass1)/coulomb_log1
	time_dyn_2 = 0.67*(sigma/100)*(10**8/mass2)/coulomb_log2


	if (time_dyn_1 > time_dyn_2):
		time_dyn = time_dyn_1
	else:
		time_dyn = time_dyn_2

	return time_dyn


def time_df2(r_eff,sigma1,sigma2,mass2):
	"""
	Function that evaluates the dynamical friction
	contribution to time delay following the prescription
	by Desopoulou and Antonini

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
	time_dyn_1 = 0.06*2/coulomb_log*(r_eff/(10**4))**(2)*\
(sigma1/(3*10**2))*((10**8)/mass2)
	time_dyn_2 = 0.15*2/coulomb_log*(r_eff/(10**4))*\
(sigma1/(3*10**2))**2*(10**2/sigma2)**3
	if (time_dyn_1 > time_dyn_2):
		time_dyn = time_dyn_1
	else:
		time_dyn = time_dyn_2

	return time_dyn  


def time_df_phase2(r_eff,r_inf,sigma1,sigma2,mass1,mass2):
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
	#b_max = r_eff
	#b_min = (G*mass2*M_sun/(sigma2*10**3)**2)/pc
	#coulomb_log_primed = np.log(b_max/b_min)
	coulomb_log_primed = np.log(mass1/mass2)

	alpha = 0.5
	beta = 1.37
	delta = -0.85

	time_bare = (0.015)*(coulomb_log_primed*alpha + beta + delta)**(-1)*2*\
(1-(mass2/(2*mass1))**(1/2))*(mass1/(3*10**9))**(1/2)*(mass2/(10**8))**(-1)*\
(r_inf/300)**(3/2)
	time_gal = (0.012)*(coulomb_log*alpha + beta + delta)**(-1)*(2*mass1/mass2-1)*\
(mass1/(3*10**9))*(100/sigma2)**3

	if(time_bare < time_gal):
		t_df_ph2 = time_bare
	else:
		t_df_ph2 = time_gal

	#print(t_df_ph2)

	return t_df_ph2








#############################################################

def eccentricity(e):
	"""
	Function that evaluates the Peter-Matheus expression

	input parameter:
		e -> eccentricity of the orbit (0 <= e < 1)

	return:
		the value (float) of the function
	"""

	value = 1/(1-e**2)**(7/2)*(1+73/24*e**2+37/96*e**4)

	return value



#############################################################

def time_star(sigma_inf,rho_inf,r_inf,mass1,mass2,mass_binary,e):
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
			-> ([a_hard_gw,time_star_hard])
	"""
	sigma_inf = sigma_inf*10**3/pc
	a_hard_gw = (64*G_new**2*sigma_inf*mass1*mass2*mass_binary*\
eccentricity(e)/(5*c_new**5*H*rho_inf))**(1/5)
	time_star_hard = (sigma_inf)/(G_new*H*rho_inf*a_hard_gw)/Gyr

	return ([a_hard_gw,time_star_hard])



#############################################################

def time_gas(mass1,mass2,mass_binary,m_dot,r_inf,e):
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
			-> ([a_gas_gw,time_gas])
	"""
	constant = 16*2**(1/2)/5*G_new**3/c_new**5
	a_gas_gw = (constant*mass1**2*mass2**2*eccentricity(e)/(m_dot))**(1/4)
	time_gas = (1/(2*2**(1/2))*mass1*mass2/mass_binary/(m_dot)*\
math.log(r_inf/a_gas_gw))/Gyr

	return ([a_gas_gw,time_gas])


#############################################################

def time_gw(a_in,mass1,mass2,mass_binary):
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
	constant = 5/256*c_new**5/G_new**3
	time_gw = (constant*a_in**4/(mass1*mass2*mass_binary))/Gyr

	return time_gw


#############################################################

def tot_delay_function(host_r_eff,host_sigma,satellite_sigma,satellite_BH,
sigma_inf,rho_inf,r_inf,mass1,mass2,e,m_dot,stellar_mass,r_eff,hardening_type):
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
			-> ([[delay_time,t_df,t_star,t_gas,t_gw]])
	"""

	mass_binary=mass1+mass2

	if(hardening_type==0):

		T1=time_df2(host_r_eff,host_sigma,satellite_sigma,satellite_BH)
		T1_1=time_df_phase2(host_r_eff,r_inf,host_sigma,satellite_sigma,mass1,mass2)
		T2=(time_star(sigma_inf,rho_inf,r_inf,mass1,mass2,mass_binary,e))[1]
		T3=(time_gas(mass1,mass2,mass_binary,m_dot,r_inf,e))[1]
		if(T2<T3):
			a_in=(time_star(sigma_inf,rho_inf,r_inf,mass1,mass2,mass_binary,e))[0]
			T4=time_gw(a_in,mass1,mass2,mass_binary)
			delay_time=T1+T1_1+T2+T4
		else:
			a_in=(time_gas(mass1,mass2,mass_binary,m_dot,r_inf,e))[0]
			T4=time_gw(a_in,mass1,mass2,mass_binary)
			delay_time=T1+T1_1+T3+T4

	else:
		T1=time_df2(host_r_eff,host_sigma,satellite_sigma,satellite_BH)
		T1_1=time_df_phase2(host_r_eff,r_inf,host_sigma,satellite_sigma,mass1,mass2)
		T2=(time_star(sigma_inf,rho_inf,r_inf,mass1,mass2,mass_binary,e))[1]
		T3=0.0
		a_in=(time_star(sigma_inf,rho_inf,r_inf,mass1,mass2,mass_binary,e))[0]
		T4=time_gw(a_in,mass1,mass2,mass_binary)

	delay_time=T1+T1_1+T2+T4

	return ([delay_time,T1,T2,T3,T4,T1_1])

def tot_delay_no_df(sigma_inf,rho_inf,r_inf,mass1,mass2,e,m_dot,hardening_type):
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

	mass_binary=mass1+mass2

	if(hardening_type==0):

		T2=(time_star(sigma_inf,rho_inf,r_inf,mass1,mass2,mass_binary,e))[1]
		T3=(time_gas(mass1,mass2,mass_binary,m_dot,r_inf,e))[1]
		if(T2<T3):
			a_in=(time_star(sigma_inf,rho_inf,r_inf,mass1,mass2,mass_binary,e))[0]
			T4=time_gw(a_in,mass1,mass2,mass_binary)
			delay_time=T2+T4
		else:
			a_in=(time_gas(mass1,mass2,mass_binary,m_dot,r_inf,e))[0]
			T4=time_gw(a_in,mass1,mass2,mass_binary)
			delay_time=T3+T4
	else:
		T2=(time_star(sigma_inf,rho_inf,r_inf,mass1,mass2,mass_binary,e))[1]
		T3=0
		a_in=(time_star(sigma_inf,rho_inf,r_inf,mass1,mass2,mass_binary,e))[0]
		T4=time_gw(a_in,mass1,mass2,mass_binary)

	delay_time=T2+T4

	return ([delay_time,T2,T3,T4])

	