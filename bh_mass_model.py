#############################################################
# Module: bh_mass.py
#
# Contains functions to find the black hole mass and
# the accreted masses
#
#    --> bh_mass_function(bulge_mass)
#    --> radio(time,hot_mass,vvir,bh_mass,stellar_mass)
#    --> quasar(stellar_mass_P1,stellar_mass_P2,cold_mass,
#				vvir)
#
# !Further information is provided below each function
#############################################################

import numpy as np 
import constants as cst

#############################################################

def bh_mass_function(model, bulge_mass):
	"""
	Function to calculate the central black hole mass
	according to an empirical scaling relation
	(Kormendy & Ho)

	input parameters:
		relation_type -> type of the scaling relatiion
						 relation_type = 1 --> Kormendy&Ho
		bulge_mass -> mass of the galaxy bulge in solar 
					  masses

	return:
		value (float) of the black hole mass in solar
		masses
	"""
	if (model == 'KH'):
		mean = 8.69 + 1.17 * math.log10(bulge_mass / 10**(11))
		sigma = 0.28
		bh_mass = np.random.normal(mean, sigma)
		bh_mass = 10**(bh_mass)

	return bh_mass

#############################################################

def radio(time, hot_mass, vvir, bh_mass, stellar_mass):
	"""
	Function that computes the mass accreted in between 
	two subsequent galaxy mergers (accretion during
	dynamical inactivity)

	input parameters:
		time -> time elapsed during two subsequent mergers,
				expressed in yr
		hot_mass -> hot gas mass of the first remnant
					remnant galaxy, in solar masses
		vvir -> virial velocity in km/s
		bh_mass -> black hole mass of the remnant galaxy,
				   in solar masses
		stellar_mass -> remnant galaxy mass in solar masses

	return:
		the value (float) of the accreted mass in solar
		masses
	"""

	time = time * cst.Gyr / cst.t_1yr
	dark_mass = 0.5 * 10**2 * stellar_mass
	delta_mass_radio = cst.kappa * (hot_mass / dark_mass / 0.1) * (vvir / 200)**3*\
					   (bh_mass * h/10**8)*(time)

	return delta_mass_radio # it's in solar masses

#############################################################

def quasar(stellar_mass_P1, stellar_mass_P2, cold_mass, vvir):
	"""
	Function to compute the cold gas accreted during the 
	merger

	input parameters:
		stellar_mass_P1 -> first progenitor galaxy mass,
						   in solar masses
		stellar_mass_P2 -> second progenitor galaxy mass,
						   in solar masses
		cold_mass -> mass of the cold gas in remnant 
					 galaxy, in solar masses
		vvir -> virial velocity of the remnant galaxy,
			    in km/s

	return:
		the value (float) of the cold mass accreted
		during the merger, in solar masses
	"""

	if(stellar_mass_P1 > stellar_mass_P2):
		stellar_mass_host = stellar_mass_P1
		stellar_mass_satellite = stellar_mass_P2
	else:
		stellar_mass_host = stellar_mass_P2
		stellar_mass_satellite = stellar_mass_P1
		
	delta_mass_quasar = cst.f * (stellar_mass_satellite / stellar_mass_host) * \
						(cold_mass / (1 + 280 / vvir))

	return delta_mass_quasar






