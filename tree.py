#################################################################
# Module: tree.py
#################################################################
# A catalog (name and data) and cosmology are passed to the main
# function:
#
#    --> def tree(catalog, density_model, mass_model, omega_matter, 
#		   omega_lambda, data)
#
# The program returns an output file containing all the information
# of the injection file as well as the results binary/triplet/
# quadruplet interactions and all the phase of dynamical evolution:
#
#    --> output_<catalog>_<mass_model>_<density_model>.csv
#
#################################################################

import math
import numpy as np
import random

import pandas as pd

import time
from tqdm import tqdm

import bonetti
import delay_time
import lookback
import triplets
import bh_mass_model
import constants as cst


#################################################################

def tree(catalog, density_model, mass_model, omega_matter, omega_lambda, data, tree_start, tree_end):

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
	bh_mass = data['bh_mass'].copy()
	P1_BH_mass = data['P1_BH_mass'].copy()
	P2_BH_mass = data['P2_BH_mass'].copy()
	q = data['q'].copy()
	mass1 = data['mass1'].copy()
	mass2 = data['mass2'].copy()
	r_eff_P1 = data['r_eff_P1'].copy()
	r_inf_P1 = data['r_inf_P2'].copy()
	sigma_P1 = data['sigma_P1'].copy()
	r_eff_P2 = data['r_eff_P2'].copy()
	r_inf_P2 = data['r_inf_P2'].copy()
	sigma_P2 =  data['sigma_P2'].copy()
	host_r_eff = data['host_r_eff'].copy()
	host_sigma = data['host_sigma'].copy()
	satellite_sigma = data['satellite_sigma'].copy()
	satellite_BH = data['satellite_BH'].copy()
	host_BH = data['host_BH'].copy()
	r_eff = data['r_eff'].copy()
	r_inf = data['r_inf'].copy()
	sigma_inf = data['sigma_inf'].copy()
	rho_inf = data['rho_inf'].copy()
	m_dot = data['m_dot'].copy().copy()
	hardening_type = data['hardening_type'].copy()


	###########################################################################################
	# type_P1 and type_P2 are vectors that store information on the
	# two progenitors, P1 and P2:
	# 
	#	-> if the value is 0 then the progenitor is a single BH
	#	-> if the value is 2 then the progenitor is a BHB
	# 
	# Therefore we can have 4 different combinations depending on the type
	# of 	  P1 and P2
	# 1) ->	  0		 0	-->	 binary black hole: simplest case
	# 2) ->	  2		 0	-->	 TRIPLET!
	# 3) ->	  0      2  -->  TRIPLET!
	# 4) ->	  2      2  -->  QUADRUPLET! -> eject the least massive BH -> Triplet! -> 3)
	#
	# In case of insuccess of triplet interaction we shall keep the two most massive 
	# black holes only! The time delay assigned to them will be:
	# time_delay_first-time_between_mergers and it will start from second merger
	#
	###########################################################################################

	type_P1 = np.zeros(ns, dtype=int)
	type_P2 = np.zeros(ns, dtype=int)

	###########################################################################################
	# P1_marker and P2_marker store the index on where to find the information
	# on the progenitors in case these are NOT single black holes
	###########################################################################################

	P1_marker = np.zeros(ns, dtype=int)
	P2_marker = np.zeros(ns, dtype=int)

	###########################################################################################
	# The mass of the remnant black hole, bh_mass, is split between the binary components
	# in a mass proportional way to the progenitor black hole masses
	# 		q=P1_BH_mass[k]/P2_BH_mass[k]
	#		mass1[k]=q/(1+q)*bh_mass
	#		mass2[k]=1/(1+q)*bh_mass
	# These values are already provided by the data file.
	#
	# In case one or both of the progenitors are still a binary, vectors mass1_1, mass2_1,
	# mass1_2, mass2_2 keep track of the single component masses
	###########################################################################################
	# in case the either one of the progenitors is a binary
	# if progenitor 1 is a binary
	mass1_1 = np.zeros(ns)
	mass2_1 = np.zeros(ns)
	# if progenitor 2 is a binary
	mass1_2 = np.zeros(ns)
	mass2_2 = np.zeros(ns)

	
	###########################################################################################
	# Vectors to store information on the exit of the binary evolution:
	#
	#	###### CLASSIFY MERGER TYPE ######
	#	-> form_binary_vector: if 1 then the event is a binary, 0 default
	#	-> form_triplet_vector: if 1 then event is a triplet, 0 default
	#	-> form_quadruplet_vector: if 1 then event is a quadruplet, 0 default
	#	NOTE that the previous information is only on the initial state of the system with
	#	no implications on whether the system will merge on time
	#
	#	###### SUCCESSFUL MERGERS ######
	#	-> binary_vector: if 1 then the merger is a binary that has managed to merge before
	#					  the next galaxy merger (otherwise 0)
	#	-> prompt_vector: if 1 then the merger is a triplet which undergoes successful prompt
	#						merger (otherwise 0)
	#	-> ejection_vector: if 1 then the merger is triplet which undergoes ejection +
	#						successful delayed merger (otherwise 0)
	# 	-> forced_binary_vector: if 1 the merger is a triplet, that remains unresolved: manually
	#							 it becomes a binary that successfully merges (otherwise 0)
	#	-> quadruplet_vector: if 1 the merger contains 4 black holes, which are reduced to a
	#						  triplet (so maximum one of the vectors above has to be 1),
	#						  otherwise it is zero
	#	
	#	###### FAILED MERGERS ######
	#	-> failed_binary_vector: if 1 then the initial binary hasn't merged on time 
	#	-> failed_prompt_vector: if 1 it is a triplet that failed prompt merger
	#	-> failed_ejection_vector: if 1 it is a triplet that failed ejection + delayed merger
	#	-> failed_failed_vector: if 1 it is an unresolved triplet, and after it is made a
	#							binary it fails to merge on time
	#
	#	###### ONGOING MERGERS ######
	#	-> still_merging_vector: if 1 then the event is the last one of a tree and has not
	#							merged yet
	###########################################################################################
	form_binary_vector = np.zeros(ns, dtype=int)
	form_triplet_vector = np.zeros(ns, dtype=int)
	form_quadruplet_vector = np.zeros(ns, dtype=int)

	binary_vector = np.zeros(ns, dtype=int)
	prompt_vector = np.zeros(ns, dtype=int)
	ejection_vector = np.zeros(ns, dtype=int)
	forced_binary_vector = np.zeros(ns, dtype=int)
	
	failed_binary_vector = np.zeros(ns, dtype=int)
	failed_prompt_vector = np.zeros(ns, dtype=int)
	failed_ejection_vector = np.zeros(ns, dtype=int)
	failed_forced_vector = np.zeros(ns, dtype=int)

	still_merging_vector = np.zeros(ns, dtype=int)

	###########################################################################################
	# Vectors to record the delay times:
	#
	#		-> dynamical friction time - phase 1: time_df_ph1
	#		-> dynamical friction time - phase 2: time_df_ph2
	#		-> total dynamical friction time: time_df
	#		-> stellar hardening time: time_star
	#		-> gaseous hardening time: time_gas
	#		-> gw coalescence time: time_gw
	#
	#		-> total merger time: time_to_merge
	#		-> redshift at binary merger: merger_redshift_vector
	#		-> time_to_next_merger: time span between two subsequent galactic mergers
	#		-> time difference between delay time and two subsequent mergers: merger_time_diff
	#
	#		-> descendant index: index where information on first descendant is stored
	###########################################################################################
	

	time_df_ph1 = np.zeros(ns)
	time_df_ph2 = np.zeros(ns)
	time_df = np.zeros(ns)
	time_star = np.zeros(ns)
	time_gas = np.zeros(ns)
	time_gw = np.zeros(ns)

	time_to_merge = np.zeros(ns)
	time_to_next_merger = np.zeros(ns)
	merger_time_diff = np.zeros(ns)
	merger_redshift_vector = np.zeros(ns)

	descendant_index = np.zeros(ns, dtype=int)

	# In case at least one of the progenitors is a binary, store information about the inner
	# and the outer mass ratios
	q_in = np.zeros(ns)
	q_out = np.zeros(ns)


	n_trees = len(tree_start)
	for i in tqdm(np.arange(n_trees)):
		start = tree_start[i]
		tree_index = tree_end[i]

		# Analyze individual trees
		for k in range(start, tree_index):

			# Combination (1)
			if (type_P1[k] == 0 and type_P2[k] == 0):
				# BINARY
				form_binary_vector[k] = 1

				# Find descendant of this merger
				descendant_index[k], P1, P2, z_descendant = lookback.find_descendant(k, tree_index, snapnum[i : tree_index],
												 galaxyId[i : tree_index], P1_Id[i : tree_index], P2_Id[i : tree_index],
												 D_z[i : tree_index])

				# Calculate time elapsed between two subsequent galaxy mergers
				time_to_next_merger[k] = lookback.time_between_mergers(z_descendant,
										 D_z[k], omega_matter, omega_lambda)

							
				# Calculate the binary merger time
				time_to_merge[k], time_df_ph1[k], time_df_ph2[k], time_star[k],\
				time_gas[k], time_gw[k] = delay_time.tot_delay_function(density_model, host_r_eff[k], host_sigma[k],
										satellite_sigma[k], satellite_BH[k], sigma_inf[k], rho_inf[k],
										r_inf[k], mass1[k], mass2[k], m_dot[k], D_mass[k], r_eff[k],
										hardening_type[k])
				time_df[k] = time_df_ph1[k] + time_df_ph2[k]
				
				if(descendant_index[k] != -1): # there is a descendant!

					q_bin = min(mass1[k],mass2[k]) / max(mass1[k],mass2[k])
					
					if (time_to_merge[k] > time_to_next_merger[k]):
						failed_binary_vector[k] = 1

						# Update information of descendant by recording that either
						# one of the progenitors will be a binary
						
						if (P1 == 1 and P2 == 0):
							# P1 of descendant is a binary!
							merger_time_diff[descendant_index[k]] = time_to_merge[k] - time_to_next_merger[k]
							type_P1[descendant_index[k]] = 2
							P1_marker[descendant_index[k]] = k
							mass1_1[descendant_index[k]] = q_bin / (1 + q_bin) * mass1[descendant_index[k]]				
							mass1_2[descendant_index[k]] = 1 / (1 + q_bin) * mass1[descendant_index[k]]


						if (P1 == 0 and P2 == 1):
							# P2 of descendant is a binary
							merger_time_diff[descendant_index[k]] = time_to_merge[k] - time_to_next_merger[k]
							type_P2[descendant_index[k]] = 2
							P2_marker[descendant_index[k]] = k
							mass2_1[descendant_index[k]] = q_bin / (1 + q_bin) * mass2[descendant_index[k]]
							mass2_2[descendant_index[k]] = 1 / (1 + q_bin) * mass2[descendant_index[k]]
			

					else:
						binary_vector[k] = 1 # successful merger!
						merger_redshift_vector[k] = lookback.find_redshift(D_z[k],
													time_to_merge[k], omega_matter, omega_lambda)

				if (descendant_index[k] == -1 and time_to_merge[k] < time_to_next_merger[k]): 
					# successful last merger of a tree:
					binary_vector[k] = 1 # successful merger!
					merger_redshift_vector[k] = lookback.find_redshift(D_z[k],
												time_to_merge[k], omega_matter, omega_lambda)

				if (descendant_index[k] == -1 and time_to_merge[k] > time_to_next_merger[k]):
					# failed last merger of a tree
					still_merging_vector[k] = 1 
					merger_redshift_vector[k] = -1



			else:
				# Combination (2)
				if (type_P1[k] == 0 and type_P2[k] == 2): # P1 is the intruder, P2 is a binary
					# TRIPLET
					form_triplet_vector[k] = 1

					# Binary from P1 and single BH (intruder) from P2
					# Find m_1, q_in and q_out to pass to triplet_function
					m_1 = max(mass2_1[k], mass2_2[k])
					m_2 = min(mass2_1[k], mass2_2[k])
					m_intr = mass1[k]
					q_in[k] = m_2/m_1
					q_out[k] = m_intr/(m_1 + m_2)


				# Combination (3)
				if (type_P1[k] == 2 and type_P2[k] == 0): # P1 is a binary, P2 is the intruder
					# TRIPLET
					form_triplet_vector[k] = 1
					
					# Binary from P1 and single BH (intruder) from P2
					# Find m_1, q_in and q_out to pass to triplet_function
					m_1 = max(mass1_1[k], mass1_2[k])
					m_2 = min(mass1_1[k], mass1_2[k])
					m_intr = mass2[k]
					q_in[k] = m_2/m_1
					q_out[k] = m_intr/(m_1 + m_2)


				# Combination (4)
				# Both P1 and P2 are binaries
				if (type_P1[k] == 2 and type_P2[k] == 2):
					# QUADRUPLET
					form_quadruplet_vector[k] = 1

					# Search for the least massive black hole among the 4 to eject

					# mass1_1 is the least massive
					if (mass1_1[k] <= mass1_2[k] and mass1_1[k] <= mass2_1[k] and mass1_1[k] <= mass2_2[k]):
						# ejection of mass1_1
						# P1 is the intruder, P2 is a binary
						# Find m_1, q_in, q_out
						m_1 = max(mass2_1[k], mass2_2[k])
						m_2 = min(mass2_1[k], mass2_2[k])
						m_intr = mass1_2[k]
						q_in[k] = m_2/m_1
						q_out[k] = m_intr/(m_1 + m_2)

					# mass1_2 is the least massive
					if (mass1_2[k] <= mass1_1[k] and mass1_2[k] <= mass2_1[k] and mass1_2[k] <= mass2_2[k]):
						# ejection of mass1_2
						# P1 is the intruder, P2 is a binary
						# Find m_1, q_in, q_out
						m_1 = max(mass2_1[k], mass2_2[k])
						m_2 = min(mass2_1[k], mass2_2[k])
						m_intr = mass1_1[k]
						q_in[k] = m_2/m_1
						q_out[k] = m_intr/(m_1 + m_2)

					# mass2_1 is the least massive
					if (mass2_1[k] <= mass1_1[k] and mass2_1[k] <= mass1_2[k] and mass2_1[k] <= mass2_2[k]):
						# ejection of mass2_1
						# P1 is a binary, P2 is the intruder
						# Find m_1, q_in, q_out
						m_1 = max(mass1_1[k], mass1_2[k])
						m_2 = min(mass1_1[k], mass1_2[k])
						m_intr = mass2_2[k]
						q_in[k] = m_2/m_1
						q_out[k] = m_intr/(m_1 + m_2)

					# mass2_2 is the least massive
					if (mass2_2[k] <= mass1_1[k] and mass2_2[k] <= mass1_2[k] and mass2_2[k] <= mass2_1[k]):
						# ejection of mass2_2
						# P1 is a binary, P2 is the intruder
						# Find m_1, q_in, q_out
						m_1 = max(mass1_1[k], mass1_2[k])
						m_2 = min(mass1_1[k], mass1_2[k])
						m_intr = mass2_1[k]
						q_in[k] = m_2/m_1
						q_out[k] = m_intr/(m_1 + m_2)


				# Calculate time to sink due to dynamical friction	
				time_df_ph1[k] = delay_time.time_df_phase1(r_eff[k], host_sigma[k],
								 satellite_sigma[k], satellite_BH[k])
				time_df_ph2[k] = delay_time.time_df_phase2(density_model, r_eff[k], r_inf[k],
								 host_sigma[k], satellite_sigma[k], mass1[k], mass2[k])
				time_df[k] = time_df_ph1[k] + time_df_ph2[k]


				# Launch triplet interaction
				# Check whether q_out is bigger than 1!
				if (q_out[k] < 1):
					triplet_output = bonetti.triplet_function(m_1, q_in[k], q_out[k])
				else:
					triplet_output = bonetti.big_triplet_function(q_out[k], q_in[k])


				# Analyze triplet output
				output = triplets.output_analyzer(triplet_output, k, tree_index, D_z[k],
						 m_intr, m_1, m_2, time_df[k], sigma_inf[k], rho_inf[k],
						 r_inf[k], m_dot[k], merger_time_diff[k], hardening_type[k],
						 omega_matter, omega_lambda, snapnum[i : tree_index], galaxyId[i : tree_index],
						 P1_Id[i : tree_index], P2_Id[i : tree_index], D_z[i : tree_index])

				merger_redshift_vector[k], time_to_merge[k], time_star[k], time_gas[k], time_gw[k], time_to_next_merger[k] = output[10 : 16]
				descendant_index[k], P1, P2 = output[0:3]

				prompt_vector[k], ejection_vector[k], forced_binary_vector[k], failed_prompt_vector[k],\
				failed_ejection_vector[k], failed_forced_vector[k], still_merging_vector[k] = output[3:10]

				if(failed_prompt_vector[k] == 1 or failed_ejection_vector[k] == 1 or  failed_forced_vector[k] == 1):
					
					q_bin = output[17]	
					
					if (P1 == 2 and P2 == 0):
						# P1 of descendant is a binary!
						merger_time_diff[descendant_index[k]] = time_to_merge[k] - time_to_next_merger[k]
						type_P1[descendant_index[k]] = 2
						P1_marker[descendant_index[k]] = k
						mass1_1[descendant_index[k]] = q_bin / (1 + q_bin) * mass1[descendant_index[k]]				
						mass1_2[descendant_index[k]] = 1 / (1 + q_bin) * mass1[descendant_index[k]]
					
					if (P1 == 0 and P2 == 2):
						# P2 of descendant is a binary
						merger_time_diff[descendant_index[k]] = time_to_merge[k] - time_to_next_merger[k]
						type_P2[descendant_index[k]] = 2
						P2_marker[descendant_index[k]] = k
						mass2_1[descendant_index[k]] = q_bin / (1 + q_bin) * mass2[descendant_index[k]]
						mass2_2[descendant_index[k]] = 1 / (1 + q_bin) * mass2[descendant_index[k]]



	#data['type_P1'] = type_P1
	#data['type_P2'] = type_P2
	#data['P1_marker'] = P1_marker
	#data['P2_marker'] = P2_marker
	#data['mass1_1'] = mass1_1
	#data['mass2_1'] = mass2_1
	#data['mass1_2'] = mass1_2
	#ata['mass2_2'] = mass2_2
	data['form_binary_vector'] = form_binary_vector
	data['form_triplet_vector'] = form_triplet_vector
	data['form_quadruplet_vector'] = form_quadruplet_vector
	data['binary_vector'] = binary_vector
	data['prompt_vector'] = prompt_vector
	data['ejection_vector'] = ejection_vector
	data['forced_binary_vector'] = forced_binary_vector
	data['failed_binary_vector'] = failed_binary_vector
	data['failed_prompt_vector'] = failed_prompt_vector
	data['failed_ejection_vector'] = failed_ejection_vector
	data['failed_forced_vector'] = failed_forced_vector
	data['still_merging_vector'] = still_merging_vector
	data['time_to_merge'] = time_to_merge
	data['time_to_next_merger'] = time_to_next_merger
	data['time_df'] = time_df
	data['time_df_ph1'] = time_df_ph1
	data['time_df_ph2'] = time_df_ph2
	data['time_star'] = time_star
	data['time_gas'] = time_gas
	data['time_gw'] = time_gw
	#data['merger_time_diff'] = merger_time_diff
	data['merger_redshift_vector'] = merger_redshift_vector
	#data['descendant_index'] = descendant_index
	data['q_in'] = q_in
	data['q_out'] = q_out



	data.to_csv('Data/OutputData/output_%s_%s_%s.csv' %(str(catalog), str(mass_model), str(density_model)), index=False) 




