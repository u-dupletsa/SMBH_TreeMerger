#################################################################
# Main module: tree.py										    
#################################################################															    
# A catalog (name and data) and cosmology are passed to the main
# function:
#
# def tree(catalog, omega_matter, omega_lambda, data)		
#															   
# The program returns two files of data:  						
#															    
#	--> model_data_CATALOGUE.csv: it contains, respectively,    
# 		the data on  										       
# 			:column1: remnant effective radius, r_eff (in pc)	
#			:column2: remnant influence radius, r_inf (in pc)   
#			:column3: remnant density at r_inf, rho_inf (in 	
#					  solar masses per pc^3)    			    
#			:column4: remnant velocity dispersion at r_inf,		
#					  sigma_inf (in km/s)					    
#															    
#	--> output_CATALOGUE.csv: it contains, respectively, 	    
#		the data on the binary component masses (in solar		
#		masses)													
#			:column1: 1st component mass, mass1					
#			:column2: 2nd compenent mass, mass2					
#		the data on the exit of binary evolution, can be  		
#		either 1 or 0 											
#			:column3: binary, if binary=1, the binary has 		
#					  successfully merged before the next       
#					  galaxy merger 							
#			:column4: triplet (prompt merger), if triplet=1, 	
#					  the system is a triplet which undergoes 	
#					  prompt merger of two of its balck holes, 	
#	                  with the remaining two forming a binary   
#					  that successfully merger before the next  
#					  galaxy merger   							
#			:column5: ejection, if ejection=1, the system is a  
#					  triplet which undergoes ejection of one   
#					  of its components, with the remaining     
# 					  two black holes that successfully merger 	
#					  before the next galaxy merger  			
#			:column6: quadruplet, if quadruplet=1, the system 	
#					  contains 4 black hole, of each the least 	
#	                  massive is discharged, so that the  		
#					  system is treated as a triplet 			
#			:column7: forced_binary, if forced_binary=1, the 	
#					  system is a triplet which undergoes  		
#					  neither prompt merger nor ejection.  		
#					  The least massive black hole is dischrged 
#					  and the system is treated as a binary     
# 		the data on redshift  									
#			:column8: redshift at galaxy merger, z_gal 			
#			:column9: redshift at binary merger, z_bin, if the 	
#					  binary is not able to coalesce before  	
#					  the next galaxy merger z_bin=-1 			
#		the data on merger times (in Gyr)   					
#			:column10: total merger time, merger_time  			
#			:column11: dynamical friction merger time, t_df     
#			:column12: stellar hardening merger time, t_star 	
#			:column13: gaseous hardening merger time, t_gas  	
#			:column14: gravitational waves merger time, t_gw  	
#			 													
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
import bh_mass
import constants as cst


#################################################################

def tree(catalog, density_model, mass_model, omega_matter, omega_lambda, data):

	e = cst.ecc #deafult eccentricity value

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
	# mass_1_2, mass2_2 keep track of the single component masses
	###########################################################################################
	# in case the either one of the progenitors is a binary
	# if progenitor 1 is a binary
	mass1_1 = np.zeros(ns, dtype=int)
	mass2_1 = np.zeros(ns, dtype=int)
	# if progenitor 2 is a binary
	mass1_2 = np.zeros(ns, dtype=int)
	mass2_2 = np.zeros(ns, dtype=int)

	
	###########################################################################################
	# Vectors to store information on the exit of the binary evolution:
	#	-> binary_vector: if 1 then the merger is a binary that has managed to merge before
	#					  the next galaxy merger (otherwise 0)
	#	-> triplet_vector: if 1 then the merger is a triplet which undergoes successful prompt
	#						merger (otherwise 0)
	#	-> ejection_vector: if 1 then the merger is triplet which undergoes successful ejection
	#						+ delayed merger (otherwise 0)
	# 	-> forced_binary_vector: if 1 the merger is a triplet, that remains unresolved: it is
	#							 made though of a binary that successfully merges (otherwise 0)
	#	-> quadruplet_vector: if 1 the merger contains 4 black holes, which are reduced to a 
	#						  triplet (so maximum one of the vectors above has to be 1), 
	#						  otherwise it is zero
	#	-> failed_prompt_vector: if 1 it is a triplet that failed prompt merger
	#	-> failed_ejection_vector: if 1 it is a triplet that failed ejectio + delayed merger
	#	-> failed_failed_vector: if 1 it is an unresolved triplet, and after it is made a
	#							binary it fails to merge in time
	###########################################################################################
	binary_vector = np.zeros(ns, dtype=int)
	triplet_vector = np.zeros(ns, dtype=int)
	ejection_vector = np.zeros(ns, dtype=int)
	forced_binary_vector = np.zeros(ns, dtype=int)
	quadruplet_vector = np.zeros(ns, dtype=int)
	failed_prompt_vector = np.zeros(ns, dtype=int)
	failed_ejection_vector = np.zeros(ns, dtype=int)
	failed_failed_vector = np.zeros(ns, dtype=int)

	###########################################################################################
	# Vectors to record the delay times:
	#		-> total merger time: time_to_merge
	#		-> dynamical friction time: time_df
	#		-> dynamical friction time - phase 2: time_df_ph2
	#		-> stellar hardening time: time_star
	#		-> gaseous hardening time: time_gas
	#		-> gw coalescence time: time_gw
	#		-> time difference between delay time and two subsequent mergers: merger_time_diff
	#		-> redshift at binary merger: merger_redshift_vector
	#		-> time_to_next_merger: time span between two subsequent galactic mergers
	#		-> descendant index
	###########################################################################################
	
	time_to_merge = np.zeros(ns, dtype=int)
	time_df = np.zeros(ns, dtype=int)
	time_df_ph2 = np.zeros(ns, dtype=int)
	time_star = np.zeros(ns, dtype=int)
	time_gas = np.zeros(ns, dtype=int)
	time_gw = np.zeros(ns, dtype=int)
	merger_time_diff = np.zeros(ns, dtype=int)
	merger_redshift_vector = np.zeros(ns, dtype=int)
	time_to_next_merger = np.zeros(ns, dtype=int)

	descendant_index = np.zeros(ns, dtype=int)


	i = 0
	while i < len(galaxyId):
		j = i 
		while (galaxyId[j] != -1): # still the same tree
			j = j + 1
			if (j >= (len(galaxyId))): # in case the data set has ended
				break

		tree_index = j
		tree_length = tree_index - i

		# Analyze individual trees
		for k in range(i, tree_index):

			# Combination (1)
			if (type_P1[k] == 0 and type_P2[k] == 0):
				# BINARY!

				# Find descendant of this merger
				info_descendant = lookback.find_descendant(k, tree_index, snapnum[i : tree_index], galaxyId[i : tree_index],
														   P1_Id[i : tree_index], P2_Id[i : tree_index], D_z[i : tree_index])

				descendant_index[k] = info_descendant[0]
				# Calculate time elapsed between two subsequent galaxy mergers
				time_to_next_merger[k] = lookback.time_between_mergers(info_descendant[-1],\
										 D_z[k],omega_matter,omega_lambda)

				'''
				# Update the mass with hot accretion
				# which occurs in between the mergers
				if (info_descendant[0] !=- 1 and info_descendant[1] == 1 and \
	info_descendant[2] == 0):
					# it is the progenitor of P1
					P1_BH_mass_KH[info_descendant[0]] = bh_mass_KH[k] + \
	bh_mass.radio(time_to_next_merger[k],M_hot[k],V_vir[k],bh_mass_KH[k],D_mass[k])

				if(info_descendant[0] !=- 1 and info_descendant[1] == 0 and \
	info_descendant[2] == 1):
					P2_BH_mass_KH[info_descendant[0]] = bh_mass_KH[k] + \
	bh_mass.radio(time_to_next_merger[k],M_hot[k],V_vir[k],bh_mass_KH[k],D_mass[k])
				'''
			
				
				# Calculate the binary merger time
				delay_output = delay_time.tot_delay_function(host_r_eff[k],host_sigma[k],\
	satellite_sigma[k],satellite_BH[k],sigma_inf[k],rho_inf[k],r_inf[k],\
	mass1[k],mass2[k],e,m_dot[k],D_mass[k],r_eff[k],hardening_type[k])
				
				time_to_merge[k] = delay_output[0]
				time_df[k] = delay_output[1]
				time_star[k] = delay_output[2]
				time_gas[k] = delay_output[3]
				time_gw[k] = delay_output[4]
				time_df_ph2[k] = delay_output[5]
				time_df_tot = time_df[k] + time_df_ph2[k]

				#print('COMB1:time_df,time_star,time_gas,time_gw', time_df[k],time_star[k],\
	#time_gas[k],time_gw[k])

				#print('descendant',info_descendant[0])
				
				if(info_descendant[0] != -1):

					q_bin = mass2[k]/mass1[k]
					
					if (time_to_merge[k] > time_to_next_merger[k]):
						if(time_df_tot < time_to_next_merger[k] or (time_df_tot >= 
	time_to_next_merger[k] and q_bin > 0.03)):
							#print('im here')
							#print(info_descendant)
							# Update information of descendant by recording that either
							# one progenitor will be a binary
							
					
							if (info_descendant[1] == 1 and info_descendant[2] == 0):
								# P1 of descendant is a binary!
								'''
								print('im here in the if')
								print('q_bin',q_bin)
								print('mass1 descendant',mass1[info_descendant[0]])
								print('info_descendant', info_descendant[0])
								print('mass1[4]',mass1[4])
								'''
								merger_time_diff_P1[info_descendant[0]] = time_to_merge[k] - \
	time_to_next_merger[k]
								type_P1[info_descendant[0]] = 2
								P1_marker[info_descendant[0]] = k
								mass1_1[info_descendant[0]] = q_bin/(1+q_bin)*mass1[info_descendant[0]]				
								mass1_2[info_descendant[0]] = 1/(1+q_bin)*mass1[info_descendant[0]]
								#print('mass1_1',mass1_1[info_descendant[0]])
								#print('mass1_2',mass1_2[info_descendant[0]])
							if (info_descendant[1] == 0 and info_descendant[2] == 1):
								# P2 of descendant is a binary
								merger_time_diff_P2[info_descendant[0]] = time_to_merge[k] - \
	time_to_next_merger[k]
								type_P2[info_descendant[0]] = 2
								P2_marker[info_descendant[0]] = k
								mass2_1[info_descendant[0]] = q_bin/(1+q_bin)*mass2[info_descendant[0]]
								mass2_2[info_descendant[0]] = 1/(1+q_bin)*mass2[info_descendant[0]]
						else:
							long_df_time[k] = 1

					else:
						binary_vector[k] = 1 # successful merger!
						merger_redshift_vector[k] = lookback.find_redshift(D_z[k],\
	time_to_merge[k],omega_matter,omega_lambda)

				if (info_descendant[0] == -1 and time_to_merge[k] < time_to_next_merger[k]):
					binary_vector[k] = 1 # successful merger!
					merger_redshift_vector[k] = lookback.find_redshift(D_z[k],\
	time_to_merge[k],omega_matter,omega_lambda)




			# Combination (2)
			if (type_P1[k] == 0 and type_P2[k] == 2): # P1 is the intruder, P2 is a binary
				# Triplet
				'''
				#divide cold accretion between binary progenitors
				mass2_1_old = mass2_1[k]
				mass2_2_old = mass2_2[k]
				mass_acc = mass2[k] - (mass2_1_old + mass2_2_old)
				mass2_1[k] = mass2_1_old + mass2_1_old/(mass2_1_old + mass2_2_old)*mass_acc
				mass2_2[k] = mass2_2_old + mass2_2_old/(mass2_1_old + mass2_2_old)*mass_acc
				'''

				# Binary from P1 and single BH (intruder) from P2
				# Find m_1, q_in and q_out to pass to triplet_function
				if(mass2_1[k] > mass2_2[k]): # m1>m2
					m_1 = mass2_1[k]
					m_2 = mass2_2[k]
				else: # m2>m1
					m_1 = mass2_2[k]
					m_2 = mass2_1[k]

				#print('m_1',m_1)
				#print('m_2',m_2)

				q_in = m_2/m_1

				m_3 = mass1[k]
				q_out = m_3/(m_1 + m_2)
				print(k, q_out)


				# Calculate time to sink due to dynamical friction	
				time_df[k] = delay_time.time_df2(r_eff[k],host_sigma[k],\
	satellite_sigma[k],satellite_BH[k])
				time_df_ph2[k] = delay_time.time_df_phase2(r_eff[k],r_inf[k],\
	host_sigma[k],satellite_sigma[k],mass1[k],mass2[k])
				time_to_sink = time_df[k] + time_df_ph2[k]

				# Launch triplet interaction
				# Check whether q_out is bigger than 1!
				if (q_out < 1):
					triplet_output = bonetti.triplet_function(m_1,q_in,q_out)
				else:
					triplet_output = bonetti.big_triplet_function(q_out,q_in)

				# Analyze triplet output
				output = triplets.output_analyzer(triplet_output,k,tree_index,D_z[k],\
	mass1[k],mass2_1[k],mass2_2[k],time_to_sink,sigma_inf[k],rho_inf[k],\
	r_inf[k],m_dot[k],D_z[P2_marker[k]],merger_time_diff_P2[k],hardening_type[k],\
	omega_matter,omega_lambda,snapnum[i : tree_index], galaxyId[i : tree_index],
	P1_Id[i : tree_index], P2_Id[i : tree_index], D_z[i : tree_index])

				output_new = [int(element) for element in output[:9]]
				output_new.append(output[9])
				output_new.append(output[10])
				output_new.append(output[11])
				output_new.append(output[12])
				output_new.append(output[13])
				output_new.append(output[14])
				output_new.append(output[15])
				output_new.append(output[16])
				output_new.append(output[17])
				output_new.append(output[18])

				output = output_new

				time_to_merge[k] = output[12]
				time_star[k] = output[13]
				time_gas[k] = output[14]
				time_gw[k] = output[15]
				time_to_next_merger[k] = output[17]

				descendant_index[k] = output[0]

				#print('COMB2:time_df,time_star,time_gas,time_gw', time_df[k],time_star[k],\
	#time_gas[k],time_gw[k])
				#print('descendant',output[0])

				if(output[0] != -1):

					q_bin = output[18]
					
					if (time_to_merge[k] > time_to_next_merger[k]):
						if(time_to_sink < time_to_next_merger[k] or (time_to_sink >= 
	time_to_next_merger[k] and q_bin > 0.03)):
							# Update information of descendant by recording that either
							# one progenitor will be a binary
							
					
							if (output[1] == 2 and output[2] == 0):
								# P1 of descendant is a binary!
								merger_time_diff_P1[output[0]] = time_to_merge[k] - time_to_next_merger[k]
								type_P1[output[0]] = 2
								P1_marker[output[0]] = k
								mass1_1[output[0]] = q_bin/(1+q_bin)*mass1[output[0]]				
								mass1_2[output[0]] = 1/(1+q_bin)*mass1[output[0]]
							
							if (output[1] == 0 and output[2] == 2):
								# P2 of descendant is a binary
								merger_time_diff_P2[output[0]] = time_to_merge[k] - time_to_next_merger[k]
								type_P2[output[0]] = 2
								P2_marker[output[0]] = k
								mass2_1[output[0]] = q_bin/(1+q_bin)*mass2[output[0]]
								mass2_2[output[0]] = 1/(1+q_bin)*mass2[output[0]]
						else:
							long_df_time[k] = 1

					else:
						triplet_vector[k] = triplet_vector[k] + output[3]
						ejection_vector[k] = ejection_vector[k] + output[4]
						forced_binary_vector[k] = binary_vector[k] + output[5]
						failed_prompt_vector[k] = failed_prompt_vector[k] + output[6]
						failed_ejection_vector[k] = failed_ejection_vector[k] + output[7]
						failed_failed_vector[k] = failed_failed_vector[k] + output[8]

						merger_redshift_vector[k] = lookback.find_redshift(D_z[k],\
		time_to_merge[k],omega_matter,omega_lambda)

				if(output[0] == -1 and time_to_merge[k] < time_to_next_merger[k]):
					triplet_vector[k] = triplet_vector[k] + output[3]
					ejection_vector[k] = ejection_vector[k] + output[4]
					forced_binary_vector[k] = binary_vector[k] + output[5]
					failed_prompt_vector[k] = failed_prompt_vector[k] + output[6]
					failed_ejection_vector[k] = failed_ejection_vector[k] + output[7]
					failed_failed_vector[k] = failed_failed_vector[k] + output[8]

					merger_redshift_vector[k] = lookback.find_redshift(D_z[k],\
	time_to_merge[k],omega_matter,omega_lambda)



				#file4.write(f'{m_1},{m_2},{m_3},{output[5]},{output[6]},{output[7]}\n')



			# Combination (3)
			if (type_P1[k] == 2 and type_P2[k] == 0): # P1 is a binary, P2 is the intruder
				# Triplet
				
				# Binary from P1 and single BH (intruder) from P2
				# Find m_1, q_in and q_out to pass to triplet_function
				if(mass1_1[k] > mass1_2[k]): # m1>m2
					m_1 = mass1_1[k]
					m_2 = mass1_2[k]
				else: # m2>m1
					m_1 = mass1_2[k]
					m_2 = mass1_1[k]

				#print('m_1',m_1)
				#print('m_2',m_2)

				q_in = m_2/m_1

				m_3 = mass1[k]
				q_out = m_3/(m_1 + m_2)
				#print(k, q_out)
					
				# Calculate time to sink due to dynamical friction	
				time_df[k] = delay_time.time_df2(r_eff[k],host_sigma[k],\
	satellite_sigma[k],satellite_BH[k])
				time_df_ph2[k] = delay_time.time_df_phase2(r_eff[k],r_inf[k],\
	host_sigma[k],satellite_sigma[k],mass1[k],mass2[k])
				time_to_sink = time_df[k] + time_df_ph2[k]

				# Launch triplet interaction
				# Check whether q_out is bigger than 1!
				if (q_out < 1):
					triplet_output = bonetti.triplet_function(m_1,q_in,q_out)
				else:
					triplet_output = bonetti.big_triplet_function(q_out,q_in)

				# Analyze triplet output
				output = triplets.output_analyzer(triplet_output,k,tree_index,D_z[k],\
	mass2[k],mass1_1[k],mass1_2[k],time_to_sink,sigma_inf[k],rho_inf[k],\
	r_inf[k],m_dot[k],D_z[P1_marker[k]],merger_time_diff_P1[k],hardening_type[k],\
	omega_matter,omega_lambda,snapnum[i : tree_index], galaxyId[i : tree_index],
	P1_Id[i : tree_index], P2_Id[i : tree_index], D_z[i : tree_index])

				output_new = [int(element) for element in output[:9]]
				output_new.append(output[9])
				output_new.append(output[10])
				output_new.append(output[11])
				output_new.append(output[12])
				output_new.append(output[13])
				output_new.append(output[14])
				output_new.append(output[15])
				output_new.append(output[16])
				output_new.append(output[17])
				output_new.append(output[18])

				output = output_new

				time_to_merge[k] = output[12]
				time_star[k] = output[13]
				time_gas[k] = output[14]
				time_gw[k] = output[15]
				time_to_next_merger[k] = output[17]


				descendant_index[k] = output[0]


				#print('COMB2:time_df,time_star,time_gas,time_gw', time_df[k],time_star[k],\
	#time_gas[k],time_gw[k])

				#print('descendant',output[0])

				if(output[0] != -1):

					q_bin = output[18]
					
					if (time_to_merge[k] > time_to_next_merger[k]):
						if(time_to_sink < time_to_next_merger[k] or (time_to_sink >= 
	time_to_next_merger[k] and q_bin > 0.03)):
							# Update information of descendant by recording that either
							# one progenitor will be a binary
							
					
							if (output[1] == 2 and output[2] == 0):
								# P1 of descendant is a binary!
								merger_time_diff_P1[output[0]] = time_to_merge[k] - time_to_next_merger[k]
								type_P1[output[0]] = 2
								P1_marker[output[0]] = k
								mass1_1[output[0]] = q_bin/(1+q_bin)*mass1[output[0]]				
								mass1_2[output[0]] = 1/(1+q_bin)*mass1[output[0]]
							
							if (output[1] == 0 and output[2] == 2):
								# P2 of descendant is a binary
								merger_time_diff_P2[output[0]] = time_to_merge[k] - time_to_next_merger[k]
								type_P2[output[0]] = 2
								P2_marker[output[0]] = k
								mass2_1[output[0]] = q_bin/(1+q_bin)*mass2[output[0]]
								mass2_2[output[0]] = 1/(1+q_bin)*mass2[output[0]]
						else:
							long_df_time[k] = 1

					else:
						triplet_vector[k] = triplet_vector[k] + output[3]
						ejection_vector[k] = ejection_vector[k] + output[4]
						forced_binary_vector[k] = binary_vector[k] + output[5]
						failed_prompt_vector[k] = failed_prompt_vector[k] + output[6]
						failed_ejection_vector[k] = failed_ejection_vector[k] + output[7]
						failed_failed_vector[k] = failed_failed_vector[k] + output[8]

						merger_redshift_vector[k] = lookback.find_redshift(D_z[k],\
		time_to_merge[k],omega_matter,omega_lambda)

				if(output[0] == -1 and time_to_merge[k] < time_to_next_merger[k]):
					triplet_vector[k] = triplet_vector[k] + output[3]
					ejection_vector[k] = ejection_vector[k] + output[4]
					forced_binary_vector[k] = binary_vector[k] + output[5]
					failed_prompt_vector[k] = failed_prompt_vector[k] + output[6]
					failed_ejection_vector[k] = failed_ejection_vector[k] + output[7]
					failed_failed_vector[k] = failed_failed_vector[k] + output[8]

					merger_redshift_vector[k] = lookback.find_redshift(D_z[k],\
	time_to_merge[k],omega_matter,omega_lambda)

				#file4.write(f'{m_1},{m_2},{m_3},{output[5]},{output[6]},{output[7]}\n')



			# Combination (4)
			# Both P1 and P2 are binaries
			if (type_P1[k] == 2 and type_P2[k] == 2):
				
				quadruplet_vector[k] = 1



				if (mass1_1[k] <= mass1_2[k] and mass1_1[k] <= mass2_1[k] and\
	mass1_1[k] <= mass2_2[k]):
					# ejection of mass1_1
					# P1 is the intruder, P2 is a binary

					# Find m_1, q_in, q_out
					if (mass2_1[k] > mass2_2[k]):
						m_1 = mass2_1[k]
						m_2 = mass2_2[k]
					else:
						m_1 = mass2_2[k]
						m_2 = mass2_1[k]
					q_in = m_2/m_1
					m_3 = mass1_2[k]
					q_out = m_3/(m_1 + m_2)
					#print(k, q_out)


					# Calculate time to sink due to dynamical friction	
					time_df[k] = delay_time.time_df2(r_eff[k],host_sigma[k],\
	satellite_sigma[k],satellite_BH[k])
					time_df_ph2[k] = delay_time.time_df_phase2(r_eff[k],r_inf[k],\
	host_sigma[k],satellite_sigma[k],mass1[k],mass2[k])
					time_to_sink = time_df[k] + time_df_ph2[k]

					# Launch triplet interaction
					# Check whether q_out is bigger than 1!
					if (q_out < 1):
						triplet_output = bonetti.triplet_function(m_1,q_in,q_out)
					else:
						triplet_output = bonetti.big_triplet_function(q_out,q_in)

					# Analyze triplet output
					output = triplets.output_analyzer(triplet_output,k,tree_index,D_z[k],\
	mass1_2[k],mass2_1[k],mass2_2[k],time_to_sink,sigma_inf[k],rho_inf[k],\
	r_inf[k],m_dot[k],D_z[P2_marker[k]],merger_time_diff_P2[k],hardening_type[k],\
	omega_matter,omega_lambda,snapnum[i : tree_index], galaxyId[i : tree_index],
	P1_Id[i : tree_index], P2_Id[i : tree_index], D_z[i : tree_index])

					output_new = [int(element) for element in output[:9]]
					output_new.append(output[9])
					output_new.append(output[10])
					output_new.append(output[11])
					output_new.append(output[12])
					output_new.append(output[13])
					output_new.append(output[14])
					output_new.append(output[15])
					output_new.append(output[16])
					output_new.append(output[17])
					output_new.append(output[18])

					output = output_new

					time_to_merge[k] = output[12]
					time_star[k] = output[13]
					time_gas[k] = output[14]
					time_gw[k] = output[15]
					time_to_next_merger[k] = output[17]

					descendant_index[k] = output[0]


					#print('COMB2:time_df,time_star,time_gas,time_gw', time_df[k],time_star[k],\
		#time_gas[k],time_gw[k])

					#print('descendant',output[0])

					if(output[0] != -1):

						q_bin = output[18]
						
						if (time_to_merge[k] > time_to_next_merger[k]):
							if(time_to_sink < time_to_next_merger[k] or (time_to_sink >= 
	time_to_next_merger[k] and q_bin > 0.03)):
								# Update information of descendant by recording that either
								# one progenitor will be a binary
								
						
								if (output[1] == 2 and output[2] == 0):
									# P1 of descendant is a binary!
									merger_time_diff_P1[output[0]] = time_to_merge[k] - time_to_next_merger[k]
									type_P1[output[0]] = 2
									P1_marker[output[0]] = k
									mass1_1[output[0]] = q_bin/(1+q_bin)*mass1[output[0]]				
									mass1_2[output[0]] = 1/(1+q_bin)*mass1[output[0]]
								
								if (output[1] == 0 and output[2] == 2):
									# P2 of descendant is a binary
									merger_time_diff_P2[output[0]] = time_to_merge[k] - time_to_next_merger[k]
									type_P2[output[0]] = 2
									P2_marker[output[0]] = k
									mass2_1[output[0]] = q_bin/(1+q_bin)*mass2[output[0]]
									mass2_2[output[0]] = 1/(1+q_bin)*mass2[output[0]]
							else:
								long_df_time[k] = 1

						else:
							triplet_vector[k] = triplet_vector[k] + output[3]
							ejection_vector[k] = ejection_vector[k] + output[4]
							forced_binary_vector[k] = binary_vector[k] + output[5]
							failed_prompt_vector[k] = failed_prompt_vector[k] + output[6]
							failed_ejection_vector[k] = failed_ejection_vector[k] + output[7]
							failed_failed_vector[k] = failed_failed_vector[k] + output[8]

							merger_redshift_vector[k] = lookback.find_redshift(D_z[k],\
	time_to_merge[k],omega_matter,omega_lambda)

					if(output[0] == -1 and time_to_merge[k] < time_to_next_merger[k]):
						triplet_vector[k] = triplet_vector[k] + output[3]
						ejection_vector[k] = ejection_vector[k] + output[4]
						forced_binary_vector[k] = binary_vector[k] + output[5]
						failed_prompt_vector[k] = failed_prompt_vector[k] + output[6]
						failed_ejection_vector[k] = failed_ejection_vector[k] + output[7]
						failed_failed_vector[k] = failed_failed_vector[k] + output[8]

						merger_redshift_vector[k] = lookback.find_redshift(D_z[k],\
	time_to_merge[k],omega_matter,omega_lambda)


					#file4.write(f'{m_1},{m_2},{m_3},{output[5]},{output[6]},{output[7]}\n')


				if (mass1_2[k] <= mass1_1[k] and mass1_2[k] <= mass2_1[k] and\
	mass1_2[k] <= mass2_2[k]):
					# ejection of mass1_2
					# P1 is the intruder, P2 is a binary
					# Find m_1, q_in, q_out
					if (mass2_1[k] > mass2_2[k]):
						m_1 = mass2_1[k]
						m_2 = mass2_2[k]
					else:
						m_1 = mass2_2[k]
						m_2 = mass2_1[k]
					q_in = m_2/m_1
					m_3 = mass1_1[k]
					q_out = m_3/(m_1 + m_2)
					#print(k, q_out)

					# Calculate time to sink due to dynamical friction	
					time_df[k] = delay_time.time_df2(r_eff[k],host_sigma[k],\
	satellite_sigma[k],satellite_BH[k])
					time_df_ph2[k] = delay_time.time_df_phase2(r_eff[k],r_inf[k],\
	host_sigma[k],satellite_sigma[k],mass1[k],mass2[k])
					time_to_sink = time_df[k] + time_df_ph2[k]



					# Launch triplet interaction
					# Check whether q_out is bigger than 1!
					if (q_out<1):
						triplet_output=bonetti.triplet_function(m_1,q_in,q_out)
					else:
						triplet_output=bonetti.big_triplet_function(q_out,q_in)

					# Analyze triplet output
					output = triplets.output_analyzer(triplet_output,k,tree_index,D_z[k],\
	mass1_1[k],mass2_1[k],mass2_2[k],time_to_sink,sigma_inf[k],rho_inf[k],\
	r_inf[k],m_dot[k],D_z[P2_marker[k]],merger_time_diff_P2[k],hardening_type[k],\
	omega_matter,omega_lambda,snapnum[i : tree_index], galaxyId[i : tree_index],
	P1_Id[i : tree_index], P2_Id[i : tree_index], D_z[i : tree_index])

					output_new = [int(element) for element in output[:9]]
					output_new.append(output[9])
					output_new.append(output[10])
					output_new.append(output[11])
					output_new.append(output[12])
					output_new.append(output[13])
					output_new.append(output[14])
					output_new.append(output[15])
					output_new.append(output[16])
					output_new.append(output[17])
					output_new.append(output[18])

					output = output_new

					time_to_merge[k] = output[12]
					time_star[k] = output[13]
					time_gas[k] = output[14]
					time_gw[k] = output[15]
					time_to_next_merger[k] = output[17]

					descendant_index[k] = output[0]


					#print('COMB2:time_df,time_star,time_gas,time_gw', time_df[k],time_star[k],\
		#time_gas[k],time_gw[k])

					#print('descendant',output[0])

					if(output[0] != -1):

						q_bin = output[18]
						
						if (time_to_merge[k] > time_to_next_merger[k]):
							if(time_to_sink < time_to_next_merger[k] or (time_to_sink >= 
	time_to_next_merger[k] and q_bin > 0.03)):
								# Update information of descendant by recording that either
								# one progenitor will be a binary
								
								if (output[1] == 2 and output[2] == 0):
									# P1 of descendant is a binary!
									merger_time_diff_P1[output[0]] = time_to_merge[k] - time_to_next_merger[k]
									type_P1[output[0]] = 2
									P1_marker[output[0]] = k
									mass1_1[output[0]] = q_bin/(1+q_bin)*mass1[output[0]]				
									mass1_2[output[0]] = 1/(1+q_bin)*mass1[output[0]]
								
								if (output[1] == 0 and output[2] == 2):
									# P2 of descendant is a binary
									merger_time_diff_P2[output[0]] = time_to_merge[k] - time_to_next_merger[k]
									type_P2[output[0]] = 2
									P2_marker[output[0]] = k
									mass2_1[output[0]] = q_bin/(1+q_bin)*mass2[output[0]]
									mass2_2[output[0]] = 1/(1+q_bin)*mass2[output[0]]
							else:
								long_df_time[k] = 1

						else:
							triplet_vector[k] = triplet_vector[k] + output[3]
							ejection_vector[k] = ejection_vector[k] + output[4]
							forced_binary_vector[k] = binary_vector[k] + output[5]
							failed_prompt_vector[k] = failed_prompt_vector[k] + output[6]
							failed_ejection_vector[k] = failed_ejection_vector[k] + output[7]
							failed_failed_vector[k] = failed_failed_vector[k] + output[8]

							merger_redshift_vector[k] = lookback.find_redshift(D_z[k],\
	time_to_merge[k],omega_matter,omega_lambda)

					if(output[0] == -1 and time_to_merge[k] < time_to_next_merger[k]):
						triplet_vector[k] = triplet_vector[k] + output[3]
						ejection_vector[k] = ejection_vector[k] + output[4]
						forced_binary_vector[k] = binary_vector[k] + output[5]
						failed_prompt_vector[k] = failed_prompt_vector[k] + output[6]
						failed_ejection_vector[k] = failed_ejection_vector[k] + output[7]
						failed_failed_vector[k] = failed_failed_vector[k] + output[8]

						merger_redshift_vector[k] = lookback.find_redshift(D_z[k],\
	time_to_merge[k],omega_matter,omega_lambda)


					#file4.write(f'{m_1},{m_2},{m_3},{output[5]},{output[6]},{output[7]}\n')


				if (mass2_1[k] <= mass1_1[k] and mass2_1[k] <= mass1_2[k] and\
	mass2_1[k] <= mass2_2[k]):
					# ejection of mass2_1
					# P1 is a binary and P2 is the intruder
					if (mass1_1[k] > mass1_2[k]):
						m_1 = mass1_1[k]
						m_2 = mass1_2[k]
					else:
						m_1 = mass1_2[k]
						m_2 = mass1_1[k]
					q_in = m_2/m_1
					m_3 = mass2_2[k]
					q_out = m_3/(m_1 + m_2)
					#print(k, q_out)


					# Calculate time to sink due to dynamical friction	
					time_df[k] = delay_time.time_df2(r_eff[k],host_sigma[k],\
	satellite_sigma[k],satellite_BH[k])
					time_df_ph2[k] = delay_time.time_df_phase2(r_eff[k],r_inf[k],\
	host_sigma[k],satellite_sigma[k],mass1[k],mass2[k])
					time_to_sink = time_df[k] + time_df_ph2[k]



					# Launch triplet interaction
					# Check whether q_out is bigger than 1!
					if (q_out < 1):
						triplet_output = bonetti.triplet_function(m_1,q_in,q_out)
					else:
						triplet_output = bonetti.big_triplet_function(q_out,q_in)

					# Analyze triplet output
					output = triplets.output_analyzer(triplet_output,k,tree_index,D_z[k],\
	mass2_2[k],mass1_1[k],mass1_2[k],time_to_sink,sigma_inf[k],rho_inf[k],\
	r_inf[k],m_dot[k],D_z[P1_marker[k]],merger_time_diff_P1[k],hardening_type[k],\
	omega_matter,omega_lambda,snapnum[i : tree_index], galaxyId[i : tree_index],
	P1_Id[i : tree_index], P2_Id[i : tree_index], D_z[i : tree_index])

					output_new = [int(element) for element in output[:9]]
					output_new.append(output[9])
					output_new.append(output[10])
					output_new.append(output[11])
					output_new.append(output[12])
					output_new.append(output[13])
					output_new.append(output[14])
					output_new.append(output[15])
					output_new.append(output[16])
					output_new.append(output[17])
					output_new.append(output[18])

					output = output_new

					time_to_merge[k] = output[12]
					time_star[k] = output[13]
					time_gas[k] = output[14]
					time_gw[k] = output[15]
					time_to_next_merger[k] = output[17]

					descendant_index[k] = output[0]


					#print('COMB2:time_df,time_star,time_gas,time_gw', time_df[k],time_star[k],\
		#time_gas[k],time_gw[k])

					#print('descendant',output[0])

					if(output[0] != -1):

						q_bin = output[18]
						
						if (time_to_merge[k] > time_to_next_merger[k]):
							if(time_to_sink < time_to_next_merger[k] or (time_to_sink >= 
	time_to_next_merger[k] and q_bin > 0.03)):
								# Update information of descendant by recording that either
								# one progenitor will be a binary
								
						
								if (output[1] == 2 and output[2] == 0):
									# P1 of descendant is a binary!
									merger_time_diff_P1[output[0]] = time_to_merge[k] - time_to_next_merger[k]
									type_P1[output[0]] = 2
									P1_marker[output[0]] = k
									mass1_1[output[0]] = q_bin/(1+q_bin)*mass1[output[0]]				
									mass1_2[output[0]] = 1/(1+q_bin)*mass1[output[0]]
								
								if (output[1] == 0 and output[2] == 2):
									# P2 of descendant is a binary
									merger_time_diff_P2[output[0]] = time_to_merge[k] - time_to_next_merger[k]
									type_P2[output[0]] = 2
									P2_marker[output[0]] = k
									mass2_1[output[0]] = q_bin/(1+q_bin)*mass2[output[0]]
									mass2_2[output[0]] = 1/(1+q_bin)*mass2[output[0]]
							else:
								long_df_time[k] = 1

						else:
							triplet_vector[k] = triplet_vector[k] + output[3]
							ejection_vector[k] = ejection_vector[k] + output[4]
							forced_binary_vector[k] = binary_vector[k] + output[5]
							failed_prompt_vector[k] = failed_prompt_vector[k] + output[6]
							failed_ejection_vector[k] = failed_ejection_vector[k] + output[7]
							failed_failed_vector[k] = failed_failed_vector[k] + output[8]

							merger_redshift_vector[k] = lookback.find_redshift(D_z[k],\
	time_to_merge[k],omega_matter,omega_lambda)

					if(output[0] == -1 and time_to_merge[k] < time_to_next_merger[k]):
						triplet_vector[k] = triplet_vector[k] + output[3]
						ejection_vector[k] = ejection_vector[k] + output[4]
						forced_binary_vector[k] = binary_vector[k] + output[5]
						failed_prompt_vector[k] = failed_prompt_vector[k] + output[6]
						failed_ejection_vector[k] = failed_ejection_vector[k] + output[7]
						failed_failed_vector[k] = failed_failed_vector[k] + output[8]

						merger_redshift_vector[k] = lookback.find_redshift(D_z[k],\
	time_to_merge[k],omega_matter,omega_lambda)

					#file4.write(f'{m_1},{m_2},{m_3},{output[5]},{output[6]},{output[7]}\n')



				if (mass2_2[k] <= mass1_1[k] and mass2_2[k] <= mass1_2[k] and\
	mass2_2[k] <= mass2_1[k]):
					# ejection of mass2_2
					# P1 is a binary, P2 is the intruder
					if (mass1_1[k] > mass1_2[k]):
						m_1 = mass1_1[k]
						m_2 = mass1_2[k]
					else:
						m_1 = mass1_2[k]
						m_2 = mass1_1[k]
					q_in = m_2/m_1
					m_3 = mass2_1[k]
					q_out = m_3/(m_1 + m_2)
					print(k, q_out)


					# Calculate time to sink due to dynamical friction	
					time_df[k] = delay_time.time_df2(r_eff[k],host_sigma[k],\
	satellite_sigma[k],satellite_BH[k])
					time_df_ph2[k] = delay_time.time_df_phase2(r_eff[k],r_inf[k],\
	host_sigma[k],satellite_sigma[k],mass1[k],mass2[k])
					time_to_sink = time_df[k] + time_df_ph2[k]


					# Launch triplet interaction
					# Check whether q_out is bigger than 1!
					if (q_out < 1):
						triplet_output = bonetti.triplet_function(m_1,q_in,q_out)
					else:
						triplet_output = bonetti.big_triplet_function(q_out,q_in)

					# Analyze triplet output
					output = triplets.output_analyzer(triplet_output,k,tree_index,D_z[k],\
	mass2_1[k],mass1_1[k],mass1_2[k],time_to_sink,sigma_inf[k],rho_inf[k],\
	r_inf[k],m_dot[k],D_z[P1_marker[k]],merger_time_diff_P1[k],hardening_type[k],
	omega_matter,omega_lambda,snapnum[i : tree_index], galaxyId[i : tree_index],
	P1_Id[i : tree_index], P2_Id[i : tree_index], D_z[i : tree_index])

					output_new = [int(element) for element in output[:9]]
					output_new.append(output[9])
					output_new.append(output[10])
					output_new.append(output[11])
					output_new.append(output[12])
					output_new.append(output[13])
					output_new.append(output[14])
					output_new.append(output[15])
					output_new.append(output[16])
					output_new.append(output[17])
					output_new.append(output[18])

					output = output_new

					time_to_merge[k] = output[12]
					time_star[k] = output[13]
					time_gas[k] = output[14]
					time_gw[k] = output[15]
					time_to_next_merger[k] = output[17]

					descendant_index[k] = output[0]


					#print('COMB2:time_df,time_star,time_gas,time_gw', time_df[k],time_star[k],\
		#time_gas[k],time_gw[k])

					#print('descendant',output[0])

					if(output[0] != -1):

						q_bin = output[18]
						
						if (time_to_merge[k] > time_to_next_merger[k]):
							if(time_to_sink < time_to_next_merger[k] or (time_to_sink >= 
	time_to_next_merger[k] and q_bin > 0.03)):
								# Update information of descendant by recording that either
								# one progenitor will be a binary
								
						
								if (output[1] == 2 and output[2] == 0):
									# P1 of descendant is a binary!
									merger_time_diff_P1[output[0]] = time_to_merge[k] - time_to_next_merger[k]
									type_P1[output[0]] = 2
									P1_marker[output[0]] = k
									mass1_1[output[0]] = q_bin/(1+q_bin)*mass1[output[0]]				
									mass1_2[output[0]] = 1/(1+q_bin)*mass1[output[0]]
								
								if (output[1] == 0 and output[2] == 2):
									# P2 of descendant is a binary
									merger_time_diff_P2[output[0]] = time_to_merge[k] - time_to_next_merger[k]
									type_P2[output[0]] = 2
									P2_marker[output[0]] = k
									mass2_1[output[0]] = q_bin/(1+q_bin)*mass2[output[0]]
									mass2_2[output[0]] = 1/(1+q_bin)*mass2[output[0]]
							else:
								long_df_time[k] = 1

						else:
							triplet_vector[k] = triplet_vector[k] + output[3]
							ejection_vector[k] = ejection_vector[k] + output[4]
							forced_binary_vector[k] = binary_vector[k] + output[5]
							failed_prompt_vector[k] = failed_prompt_vector[k] + output[6]
							failed_ejection_vector[k] = failed_ejection_vector[k] + output[7]
							failed_failed_vector[k] = failed_failed_vector[k] + output[8]

							merger_redshift_vector[k] = lookback.find_redshift(D_z[k],\
	time_to_merge[k],omega_matter,omega_lambda)

					if(output[0] == -1 and time_to_merge[k] < time_to_next_merger[k]):
						triplet_vector[k] = triplet_vector[k] + output[3]
						ejection_vector[k] = ejection_vector[k] + output[4]
						forced_binary_vector[k] = binary_vector[k] + output[5]
						failed_prompt_vector[k] = failed_prompt_vector[k] + output[6]
						failed_ejection_vector[k] = failed_ejection_vector[k] + output[7]
						failed_failed_vector[k] = failed_failed_vector[k] + output[8]

						merger_redshift_vector[k] = lookback.find_redshift(D_z[k],\
	time_to_merge[k],omega_matter,omega_lambda)

					#file4.write(f'{m_1},{m_2},{m_3},{output[5]},{output[6]},{output[7]}\n')

			

		i = i + tree_length + 1

		if (i >= len(galaxyId)):
			break

	'''
	print('merger_time_diff_P1',merger_time_diff_P1)
	print('merger_time_diff_P2',merger_time_diff_P2)
	print('time_to_next_merger',time_to_next_merger)
	print('merger_time',time_to_merge)
	print('long_df',long_df_time)
	'''


	file1 = open('Data/OutputData/output_%s.csv' %str(catalog),'w')
	file1.write(f'mass1,mass2,binary,triplet,ejection,quadruplet,forced,\
	failed_prompt_vector,failed_ejection_vector,failed_failed_vector,redshift,\
	redshift_merger,merger_time,time_to_next_merger,time_df,time_df_ph2,time_star,\
	time_gas,time_gw,long_df_time\n')
	for i in range(len(galaxyId)):
		if(galaxyId[i] != -1):
			file1.write(f'{mass1[i]},{mass2[i]},{binary_vector[i]},{triplet_vector[i]},\
	{ejection_vector[i]},{quadruplet_vector[i]},{forced_binary_vector[i]},\
	{failed_prompt_vector[i]},{failed_ejection_vector[i]},{failed_failed_vector[i]},\
	{D_z[i]},{merger_redshift_vector[i]},{time_to_merge[i]},{time_to_next_merger[i]},\
	{time_df[i]},{time_df_ph2[i]},{time_star[i]},{time_gas[i]},{time_gw[i]},{long_df_time[i]}\n')
		else:
			file1.write(f'-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1\n')
	file1.close()


	file5 = open('Data/OutputData/index_data_%s.csv' %str(catalog), 'w')
	file5.write(f'galaxyId,descendant_index,descendantId,P1_marker,P2_marker,type_P1,type_P2\n')
	for i in range(len(galaxyId)):
		if(galaxyId[i] != -1):
			if(descendant_index[i] != -1):
				file5.write(f'{galaxyId[i]},{descendant_index[i]},{galaxyId[int(descendant_index[i])]},\
	{P1_marker[i]},{P2_marker[i]},{type_P1[i]},{type_P2[i]}\n')
			else:
				file5.write(f'{galaxyId[i]},{descendant_index[i]},{-1.0},\
	{P1_marker[i]},{P2_marker[i]},{type_P1[i]},{type_P2[i]}\n')
		else:
			file5.write(f'-1,-1,-1,-1,-1,-1,-1\n')
	file5.close()


