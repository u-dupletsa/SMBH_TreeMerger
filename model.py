#############################################################
#	Module: model.py										
#															
#	Models masses of black holes and different types of radii
#	for both the remnant and the progenitor galaxies. The new 
#	columns are added to the injection data files.
#
#
#	Input file: --> sel_starting_ordered_<catalog>.hdf5 
#	Output file:--> injection_<catalog>.hdf5  									
# 															
#	!Further information is provided below each function      
#############################################################

import numpy as np
import pandas as pd

import constants as cst
import bh_mass_model
import density_profile

from tqdm import tqdm


catalogs = ['bertone', 'de_lucia', 'guo2010', 'guo2013']
data_folder = 'Data/InputData'
mass_model = 'KH'
density_model = 'isothermal'



lbs = ['galaxyId', 'lastProgenitorId', 'snapnum', 'descendantId', 'P1_Id', 'P2_Id', 'D_z',
	   'D_mass', 'D_bulge', 'sfr', 'sfr_bulge', 'D_BH', 'P1_z', 'P2_z', 'M1', 'M2', 
	   'P1_bulge', 'P2_bulge', 'P1_stars', 'P2_stars', 'M_cold', 'M_hot', 'V_vir', 
	   'P1_M_cold', 'P1_M_hot', 'P1_V_vir', 'P2_M_cold', 'P2_M_hot', 'P2_V_vir']

for i in tqdm(range(len(catalogs))):

	data = pd.read_csv('%s/sel_starting_ordered_%s.csv' %(str(data_folder), str(catalogs[i])),
					    names = lbs, skiprows = 1, delimiter = ',')

	ns = len(data['galaxyId'])

	galaxyId = data['galaxyId'].copy()
	lastProgenitorId = data['lastProgenitorId'].copy()
	snapnum = data['snapnum'].copy()
	descendantId = data['descendantId'].copy()
	P1_Id = data['P1_Id'].copy()
	P2_Id = data['P2_Id'].copy()
	D_z = data['D_z'].copy()
	D_mass = data['D_mass'].copy()
	D_bulge = data['D_bulge'].copy()
	sfr = data['sfr'].copy()
	sfr_bulge = data['sfr_bulge'].copy()
	D_BH = data['D_BH'].copy()
	P1_z = data['P1_z'].copy()
	P2_z = data['P2_z'].copy()
	M1 = data['M1'].copy()
	M2 = data['M2'].copy()
	P1_bulge = data['P1_bulge'].copy()
	P2_bulge = data['P2_bulge'].copy()
	P1_stars = data['P1_stars'].copy()
	P2_stars = data['P2_stars'].copy()
	M_cold = data['M_cold'].copy()
	M_hot = data['M_hot'].copy()
	V_vir = data['V_vir'].copy()
	P1_M_cold = data['P1_M_cold'].copy()
	P1_M_hot = data['P1_M_hot'].copy()
	P1_V_vir = data['P1_V_vir'].copy()
	P2_M_cold = data['P2_M_cold'].copy()
	P2_M_hot = data['P2_M_hot'].copy()
	P2_V_vir = data['P2_V_vir'].copy()


	###########################################################################################
	# bh_mass_model is the vector to store information on the mass of the remnant black hole
	# as a result of application of the empirical relation between black hole mass 
	# and host galaxy mass
	# The default one is the Kormendy&Ho empirical relation
	###########################################################################################

	bh_mass = -1. * np.ones(ns)
	P1_BH_mass = -1. * np.ones(ns)
	P2_BH_mass = -1. * np.ones(ns)
	q = -1. * np.ones(ns)

	###########################################################################################
	# The mass of the remnant black hole, bh_mass_model, is split between the binary components
	# in a mass proportional way to the progenitor black hole masses
	# 		q=P1_BH_mass_model[k]/P2_BH_mass_model[k]
	#		mass1[k]=q/(1+q)*bh_mass_model
	#		mass2[k]=1/(1+q)*bh_mass_model
	###########################################################################################

	mass1 =  -1. * np.ones(ns)
	mass2 =  -1. * np.ones(ns)

	###########################################################################################
	# Radii data
	# Progenitor1
	r_eff_P1 =  -1. * np.ones(ns)
	r_inf_P1 =  -1. * np.ones(ns)
	sigma_P1 =  -1. * np.ones(ns)


	# Progenitor2
	r_eff_P2 =  -1. * np.ones(ns)
	r_inf_P2 =  -1. * np.ones(ns)
	sigma_P2 =  -1. * np.ones(ns)
	
	# determine host and satellite progenitors
	host_r_eff =  -1. * np.ones(ns)
	host_sigma =  -1. * np.ones(ns)
	satellite_sigma =  -1. * np.ones(ns)
	satellite_BH =  -1. * np.ones(ns)
	host_BH =  -1. * np.ones(ns)

	# remnant
	r_eff =  -1. * np.ones(ns)
	r_inf =  -1. * np.ones(ns)
	
	# Sigma and rho of the remnant at r_inf
	sigma_inf =  -1. * np.ones(ns)
	rho_inf =  -1. * np.ones(ns)
	



	###########################################################################################
	# accretion rate vector
	m_dot =  -1. * np.ones(ns)
	# If sfr=0, then the hardening is stellar only and hardening_type=1,
	# otherwise hardening_type=0 and it could be either stellar or gaseous
	# depending on which is more efficient
	hardening_type =  -1. * np.ones(ns)

	for k in tqdm(range(ns)):

		if(galaxyId[k] != -1):

			# convert all mass variables in solar masses
			D_mass[k] = D_mass[k] * cst.mass_conv
			D_bulge[k] = D_bulge[k] * cst.mass_conv
			D_BH[k] = D_BH[k] * cst.mass_conv 
			M1[k] = M1[k] * cst.mass_conv 
			M2[k] = M2[k] * cst.mass_conv
			P1_bulge[k] = P1_bulge[k] * cst.mass_conv
			P2_bulge[k] = P2_bulge[k] * cst.mass_conv
			P1_stars[k] = P1_stars[k] * cst.mass_conv
			P2_stars[k] = P2_stars[k] * cst.mass_conv
			M_cold[k] = M_cold[k] * cst.mass_conv
			M_hot[k] = M_hot[k] * cst.mass_conv
			P1_M_cold[k] = P1_M_cold[k] * cst.mass_conv
			P2_M_cold[k] = P2_M_cold[k] * cst.mass_conv
			P1_M_hot[k] = P1_M_hot[k] * cst.mass_conv
			P2_M_hot[k] = P2_M_hot[k] * cst.mass_conv
			# convert sfr variables in Msol per year
			sfr[k] = sfr[k] / cst.t_1yr 
			sfr_bulge[k] = sfr_bulge[k] / cst.t_1yr

			if (mass_model != 'millennium'):

				if (D_bulge[k] > 0.):
					bh_mass[k] = bh_mass_model.bh_mass_function(mass_model, D_bulge[k]) 
				else:
					bh_mass[k] = bh_mass_model.bh_mass_function(mass_model, D_mass[k])

				if (P1_bulge[k] > 0.):
					P1_BH_mass[k] = bh_mass_model.bh_mass_function(mass_model, P1_bulge[k]) 
				else:
					P1_BH_mass[k] = bh_mass_model.bh_mass_function(mass_model, P1_stars[k])

				if (P2_bulge[k] > 0.):
					P2_BH_mass[k] = bh_mass_model.bh_mass_function(mass_model, P2_bulge[k])
				else:
					P2_BH_mass[k] = bh_mass_model.bh_mass_function(mass_model, P2_stars[k])

			else:
				bh_mass[k] = D_BH[k]
				P1_BH_mass[k] = M1[k]
				P2_BH_mass[k] = M2[k]



			# Split the remnant black hole mass in the binary component masses
					# in mass proportional way to the progenitor masses
			q[k] = P1_BH_mass[k]/P2_BH_mass[k]
			mass1[k] = q[k]/(1. + q[k]) * bh_mass[k]
			mass2[k] = 1./(1. + q[k]) * bh_mass[k]
			
			''' NOT SURE IF THIS IS IMPORTANT 
			# Rearrange the masses so that mass1>mass2
			if (mass1[k] < mass2[k]):
				m_aux = mass2[k]
				mass2[k] = mass1[k]
				mass1[k] = m_aux
			'''

			

			r_eff_P1[k] = density_profile.effective_radius(P1_bulge[k], P1_stars[k], P1_z[k])
			r_eff_P2[k] = density_profile.effective_radius(P2_bulge[k], P2_stars[k], P2_z[k])
			r_inf_P1[k] = density_profile.influence_radius(density_model, r_eff_P1[k], P1_stars[k], P1_BH_mass[k])
			r_inf_P2[k] = density_profile.influence_radius(density_model, r_eff_P2[k], P1_stars[k], P2_BH_mass[k])

			sigma_P1[k] = density_profile.sigma(density_model, P1_stars[k], r_eff_P1[k], r_inf_P1[k])
			sigma_P2[k] = density_profile.sigma(density_model, P2_stars[k], r_eff_P2[k], r_inf_P2[k])




			if (P1_stars[k] >= P2_stars[k]):
				host_r_eff[k] = r_eff_P1[k]
				host_sigma[k] = sigma_P1[k]
				host_BH[k] = P1_BH_mass[k]
				satellite_sigma[k] = sigma_P2[k]
				satellite_BH[k] = P2_BH_mass[k]
			else:
				host_r_eff[k] = r_eff_P2[k]
				host_sigma[k] = sigma_P2[k]
				host_BH[k] = P2_BH_mass[k]
				satellite_sigma[k] = sigma_P1[k]
				satellite_BH[k] = P1_BH_mass[k]


			# Calculate the quantities at the influence radius for the remnant galaxy
			# necessary for the hardening timescales
			r_eff[k] = density_profile.effective_radius(D_bulge[k], D_mass[k], D_z[k])
			r_inf[k] = density_profile.influence_radius(density_model, r_eff[k], D_mass[k], bh_mass[k])
			sigma_inf[k] = density_profile.sigma_inf(density_model, bh_mass[k], D_mass[k], r_inf[k])
			rho_inf[k] = density_profile.rho_inf(density_model, D_mass[k], r_inf[k], r_eff[k])



			if (sfr[k] == 0.):
				hardening_type[k] = 1 
				m_dot[k] = 0. # we will employ stellar hardening only!
			else:
				hardening_type[k] = 0.
				m_dot[k] = (1.16 * 10**(13) * (sfr[k])**(0.93)) / (0.1 * cst.c**2)


				


	data['galaxyId'] = 	galaxyId
	data['lastProgenitorId'] = lastProgenitorId
	data['snapnum'] = snapnum
	data['descendantId'] = descendantId
	data['P1_Id'] = P1_Id
	data['P2_Id'] = P2_Id
	data['D_z'] = D_z
	data['D_mass'] = D_mass
	data['D_bulge'] = D_bulge
	data['sfr'] = sfr
	data['sfr_bulge'] = sfr_bulge
	data['D_BH'] = D_BH
	data['P1_z'] = P1_z
	data['P2_z'] = P2_z
	data['M1'] = M1
	data['M2'] = M2
	data['P1_bulge'] = P1_bulge
	data['P2_bulge'] = P2_bulge
	data['P1_stars'] = P1_stars
	data['P2_stars'] = P2_stars
	data['M_cold'] = M_cold
	data['M_hot'] = M_hot
	data['V_vir'] = V_vir
	data['P1_M_cold'] = P1_M_cold
	data['P1_M_hot'] = P1_M_hot
	data['P1_V_vir'] = P1_V_vir
	data['P2_M_cold'] = P2_M_cold
	data['P2_M_hot'] = P2_M_hot
	data['P2_V_vir'] = P2_V_vir
	data['bh_mass'] = bh_mass
	data['P1_BH_mass'] = P1_BH_mass
	data['P2_BH_mass'] = P2_BH_mass
	data['q'] = q
	data['mass1'] = mass1
	data['mass2'] = mass2
	data['r_eff_P1'] = r_eff_P1
	data['r_inf_P1'] = r_inf_P1
	data['sigma_P1'] = sigma_P1
	data['r_eff_P2'] = r_eff_P2
	data['r_inf_P2'] = r_inf_P2
	data['sigma_P2'] = sigma_P2
	data['host_r_eff'] = host_r_eff
	data['host_sigma'] = host_sigma
	data['satellite_sigma'] = satellite_sigma
	data['satellite_BH'] = satellite_BH
	data['host_BH'] = host_BH
	data['r_eff'] = r_eff
	data['r_inf'] = r_inf
	data['sigma_inf'] = sigma_inf
	data['rho_inf'] = rho_inf
	data['m_dot'] = m_dot
	data['hardening_type'] = hardening_type

	data.to_csv('%s/injection_%s_%s_%s.csv' %(str(data_folder), str(catalogs[i]), str(mass_model), str(density_model)), index=False)  
	#delim = ','
	#header = delim.join(data.keys())
	#np.savetxt('%s/injection_%s.csv' %(str(data_folder), str(catalogs[i])), data, fmt = ','.join(['%i'] * 6 + ['%1.4f'] * 45 + ['%i']), 
	#			header=header)


	








