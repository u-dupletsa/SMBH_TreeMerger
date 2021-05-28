#################################################################
# MAIN PROGRAM: tree.py										    #
#################################################################															    
# Analyzes a selected catalogue of merger trees 			    #
# There are 4 different catalogues:							    #												          #
#		--> bertone 										    #		
#		--> de_lucia   										    # 
#		--> guo2010											    #
#		--> guo2013											    #
# All catalogues are structured in the same way: one has to     #
# indicate in the variable CATALOGUE the name of the one to     #
# analyze and the cosmological parameters to use in the   		# 
# omega_matter and omega_lambda variables, respectively			#
#															    #
# The program returns two files of data:  						#
#															    #
#	--> model_data_CATALOGUE.csv: it contains, respectively,    #
# 		the data on  										    #   
# 			:column1: remnant effective radius, r_eff (in pc)	#
#			:column2: remnant influence radius, r_inf (in pc)   #
#			:column3: remnant density at r_inf, rho_inf (in 	#
#					  solar masses per pc^3)    			    #
#			:column4: remnant velocity dispersion at r_inf,		#
#					  sigma_inf (in km/s)					    #
#															    #
#	--> output_CATALOGUE.csv: it contains, respectively, 	    #
#		the data on the binary component masses (in solar		#
#		masses)													#
#			:column1: 1st component mass, mass1					#
#			:column2: 2nd compenent mass, mass2					#
#		the data on the exit of binary evolution, can be  		#
#		either 1 or 0 											#
#			:column3: binary, if binary=1, the binary has 		#
#					  successfully merged before the next       #
#					  galaxy merger 							#
#			:column4: triplet (prompt merger), if triplet=1, 	#
#					  the system is a triplet which undergoes 	#
#					  prompt merger of two of its balck holes, 	#
#	                  with the remaining two forming a binary   #
#					  that successfully merger before the next  #
#					  galaxy merger   							#
#			:column5: ejection, if ejection=1, the system is a  #
#					  triplet which undergoes ejection of one   #
#					  of its components, with the remaining     #
# 					  two black holes that successfully merger 	#
#					  before the next galaxy merger  			#
#			:column6: quadruplet, if quadruplet=1, the system 	#
#					  contains 4 black hole, of each the least 	#
#	                  massive is discharged, so that the  		#
#					  system is treated as a triplet 			#
#			:column7: forced_binary, if forced_binary=1, the 	#
#					  system is a triplet which undergoes  		#
#					  neither prompt merger nor ejection.  		#
#					  The least massive black hole is dischrged #
#					  and the system is treated as a binary     #
# 		the data on redshift  									#
#			:column8: redshift at galaxy merger, z_gal 			#
#			:column9: redshift at binary merger, z_bin, if the 	#
#					  binary is not able to coalesce before  	#
#					  the next galaxy merger z_bin=-1 			#
#		the data on merger times (in Gyr)   					#
#			:column10: total merger time, merger_time  			#
#			:column11: dynamical friction merger time, t_df     #
#			:column12: stellar hardening merger time, t_star 	#
#			:column13: gaseous hardening merger time, t_gas  	#
#			:column14: gravitational waves merger time, t_gw  	#
#			 													#
#															 	#
#################################################################

import math
import numpy as np
import random

import pandas as pd

import time

import bonetti
import delay_time
import lookback
import triplets
import isothermal
import bh_mass


#################################################################
#################################################################
# Select catalogue and cosmology
catalogue = 'guo2013'
omega_matter = 0.272
omega_lambda = 0.728
h = 0.704
#################################################################
#################################################################


# Useful constants (in kgs)
M_sun = 1.99*10**(30)
mass_conv = (10**(10))/h
G = 6.67*10**(-11)
c = 3.0*10**8
pc = 3.086*10**(16)
H = 15
freq_1yr = 3.17098*10**(-8)
t_1yr = 3.15*10**7
Gyr = 3.15*10**16
TH0 = 13.4 #(Hubble time, in Gyr)
acc_rate = 0.1
gamma = 1 # parameter of the Dehnen profile (Hernquist if gamma=1)
e = 0

#################################################################

# Open the catalogue data
data = pd.read_csv('Data/InputData/sel_starting_ordered_%s.csv' %catalogue)

# Store relevant data in individual arrays
galaxyId = data.iloc[:,0].values
DlastProgenitor = data.iloc[:,1].values
snapnum = data.iloc[:,2].values
Ddescendant = data.iloc[:,3].values
P1_galaxyId = data.iloc[:,4].values
P2_galaxyId = data.iloc[:,5].values
redshift = data.iloc[:,6].values
stellar_mass = data.iloc[:,7].values
bulge_mass = data.iloc[:,8].values
sfr = data.iloc[:,9].values
sfr_bulge = data.iloc[:,10].values
BH_mass = data.iloc[:,11].values
P1_redshift = data.iloc[:,12].values
P2_redshift = data.iloc[:,13].values
P1_BH_mass = data.iloc[:,14].values
P2_BH_mass = data.iloc[:,15].values
P1_bulge_mass = data.iloc[:,16].values
P2_bulge_mass = data.iloc[:,17].values
P1_star_mass = data.iloc[:,18].values
P2_star_mass = data.iloc[:,19].values
M_cold = data.iloc[:,20].values
M_hot = data.iloc[:,21].values
V_vir = data.iloc[:,22].values


# Modify mass terms so they have the solar mass units
for i in range(len(galaxyId)):
	if(galaxyId[i] != -1):
		bulge_mass[i] = bulge_mass[i]*mass_conv
		stellar_mass[i] = stellar_mass[i]*mass_conv
		BH_mass[i] = BH_mass[i]*mass_conv
		P1_BH_mass[i] = P1_BH_mass[i]*mass_conv
		P2_BH_mass[i] = P2_BH_mass[i]*mass_conv
		P1_bulge_mass[i] = P1_bulge_mass[i]*mass_conv
		P2_bulge_mass[i] = P2_bulge_mass[i]*mass_conv
		P1_star_mass[i] = P1_star_mass[i]*mass_conv
		P2_star_mass[i] = P2_star_mass[i]*mass_conv
		sfr[i] = sfr[i]/(t_1yr) #It's measured in solar masses per second
		sfr_bulge[i] = sfr_bulge[i]*(M_sun)/(t_1yr)
		M_cold[i] = M_cold[i]*mass_conv
		M_hot[i] = M_hot[i]*mass_conv

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
# 4) ->	  2      2  -->  QUADRUPLET! -> eject the least massive BH -> Triplet!
#
# In case of insuccess of triplet interaction we shall keep the two most massive 
# black holes only! The time delay assigned to them will be:
# time_delay_first-time_between_mergers and it will start from second merger
#
###########################################################################################

type_P1 = np.zeros((len(galaxyId)), dtype=int)
type_P2 = np.zeros((len(galaxyId)), dtype=int)

###########################################################################################
# P1_marker and P2_marker store the index on where to find the information
# on the progenitors in case these are NOT single black holes
###########################################################################################

P1_marker = np.zeros((len(galaxyId)), dtype=int)
P2_marker = np.zeros((len(galaxyId)), dtype=int)

###########################################################################################
# bh_mass_KH is the vector to store information on the mass of the remnant black hole
# as a result of application of the empirical relation between black hole mass 
# and host galaxy mass
# Here we are using the Kormendy&Ho empirical relation (KH)
###########################################################################################

bh_mass_KH = np.zeros(len(galaxyId))
P1_BH_mass_KH = np.zeros(len(galaxyId))
P2_BH_mass_KH = np.zeros(len(galaxyId))
q = np.zeros(len(galaxyId))

###########################################################################################
# The mass of the remnant black hole, bh_mass_KH, is split between the binary components
# in a mass proportional way to the progenitor black hole masses
# 		q=P1_BH_mass_KH[k]/P2_BH_mass_KH[k]
#		mass1[k]=q/(1+q)*bh_mass_KH
#		mass2[k]=1/(1+q)*bh_mass_KH
# In case one or both of the progenitors are still a binary, vectors mass1_1, mass2_1,
# mass_1_2, mass2_2 keep track of the single component masses
###########################################################################################
mass1 = np.zeros(len(galaxyId))
mass2 = np.zeros(len(galaxyId))
# in case the either one of the progenitors is a binary
# if progenitor 1 is a binary
mass1_1 = np.zeros(len(galaxyId))
mass2_1 = np.zeros(len(galaxyId))
# if progenitor 2 is a binary
mass1_2 = np.zeros(len(galaxyId))
mass2_2 = np.zeros(len(galaxyId))
# accretion rate vector
m_dot = np.zeros(len(galaxyId))


for k in range(len(galaxyId)):
	if(galaxyId[k] != -1):

		if(bulge_mass[k] > 0.0):
			bh_mass_KH[k] = bh_mass.bh_mass_function(1,bulge_mass[k])
		else:
			bh_mass_KH[k] = bh_mass.bh_mass_function(1,stellar_mass[k])

		if(P1_bulge_mass[k] > 0.0):
			P1_BH_mass_KH[k] = bh_mass.bh_mass_function(1,P1_bulge_mass[k])
		else:
			P1_BH_mass_KH[k] = bh_mass.bh_mass_function(1,P1_star_mass[k])

		if(P2_bulge_mass[k] > 0.0):
			P2_BH_mass_KH[k] = bh_mass.bh_mass_function(1,P2_bulge_mass[k])
		else:
			P2_BH_mass_KH[k] = bh_mass.bh_mass_function(1,P2_star_mass[k])

		
		# Split the remnant black hole mass in the binary component masses
		# in mass proportional way to the progenitor masses
		q[k] = P1_BH_mass_KH[k]/P2_BH_mass_KH[k]
		mass1[k] = q[k]/(1 + q[k])*bh_mass_KH[k]
		mass2[k] = 1/(1 + q[k])*bh_mass_KH[k]
		# Rearrange the masses so that mass1>mass2
		if(mass1[k] < mass2[k]):
			aux_mass = mass1[k]
			mass1[k] = mass2[k]
			mass2[k] = aux_mass

###########################################################################################
# Radii data
# remnant
r_eff = np.zeros(len(galaxyId))
r_inf = np.zeros(len(galaxyId))
sigma = np.zeros(len(galaxyId))
# Progenitor1
r_eff_P1 = np.zeros(len(galaxyId))
r_inf_P1 = np.zeros(len(galaxyId))
sigma_P1 = np.zeros(len(galaxyId))
# Progenitor2
r_eff_P2 = np.zeros(len(galaxyId))
r_inf_P2 = np.zeros(len(galaxyId))
sigma_P2 = np.zeros(len(galaxyId))

# Sigma and rho of the remnant at r_inf
sigma_inf = np.zeros(len(galaxyId))
rho_inf = np.zeros(len(galaxyId))

# If sfr=0, then the hardening is stellar only and hardening_type=1,
# otherwise hardening_type=0 and it could be either stellar or gaseous
# depending on which is more efficient
hardening_type = np.zeros(len(galaxyId))

###########################################################################################
# Vectors to store information on the exit of the binary evolution:
#	-> binary_vector: if 1 than the merger is a binary that has managed to merge before
#					  the next galaxy merger (otherwise 0)
#	-> triplet_vector: if 1 than the merger is triplet which undergoes successful prompt
#						merger (otherwise 0)
#	-> ejection_vector: if 1 than the merger is triplet which undergoes successful ejection
#						+ delayed merger (otherwise 0)
# 	-> forced_binary_vector: of 1 the merger is a triplet, that remains unresolved, it is
#							 made a binary that successfully merges (otherwise 0)
#	-> quadruplet_vector: if 1 the merger contains 4 black holes, which are reduced to a 
#						  triplet (so maximum one of the vectors above has to be 1), 
#						  otherwise it is zero
# If all 5 merger types are zero than the merger is NOT successful
###########################################################################################
binary_vector = np.zeros(len(galaxyId))
triplet_vector = np.zeros(len(galaxyId))
ejection_vector = np.zeros(len(galaxyId))
forced_binary_vector = np.zeros(len(galaxyId))
quadruplet_vector = np.zeros(len(galaxyId))
failed_prompt_vector = np.zeros(len(galaxyId))
failed_ejection_vector = np.zeros(len(galaxyId))
failed_failed_vector = np.zeros(len(galaxyId))

###########################################################################################
# Vectors to record the delay times:
#		-> total merger time: time_to_merge
#		-> dynamical friction time: time_df
#		-> stellar hardening time: time_star
#		-> gaseous hardening time: time_gas
#		-> gw coalescence time: time_gw
#		-> time difference between delay time and two subsequent mergers: merger_time_diff
#		-> redshift at binary merger: merger_redshift_vector
###########################################################################################
merger_redshift_vector = np.zeros(len(galaxyId))
merger_time_diff_P1 = np.zeros(len(galaxyId))
merger_time_diff_P2 = np.zeros(len(galaxyId))
time_to_merge = np.zeros(len(galaxyId))
time_df = np.zeros(len(galaxyId))
time_df_ph2 = np.zeros(len(galaxyId))
time_star = np.zeros(len(galaxyId))
time_gas = np.zeros(len(galaxyId))
time_gw = np.zeros(len(galaxyId))
time_to_next_merger = np.zeros(len(galaxyId))

descendant_index = np.zeros(len(galaxyId))
long_df_time = np.zeros(len(galaxyId))


###########################################################################################
# TREE ANALYSIS: 	MAIN PROGRAM
###########################################################################################
file4 = open('Data/OutputData/MassData/triplets_mass_%s_hern.csv' %catalogue,'w')
file4.write(f'm1,m2,m3,promt,ejection,unresolved\n')


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
	# At each merger we calculate the progenitors and remnant black hole masses
	# according to the Kormendy&Ho empirical relation

		#print(k)

		#print('k, mass1, mass2, mass1_1, mass1_2, mass2_1, mass2_2', k,mass1[k],\
#mass2[k],mass1_1[k],mass1_2[k], mass2_1[k], mass2_2[k])
		#print('merger_time_diff',merger_time_diff)
		
		# Calculate the effective radii of progenitors, and the velocity
		# dispersions at r_eff necessary for dynamical friction calculations
		r_eff_P1[k] = isothermal.effective_radius(P1_bulge_mass[k],P1_star_mass[k],\
P1_redshift[k])
		r_eff_P2[k] = isothermal.effective_radius(P2_bulge_mass[k],P2_star_mass[k],\
P2_redshift[k])
		'''
		r_scale_P1 = dehnen.scale_radius(r_eff_P1[k],gamma)
		r_scale_P2 = dehnen.scale_radius(r_eff_P2[k],gamma)
		'''

		r_inf_P1[k] = isothermal.influence_radius(r_eff_P1[k],P1_star_mass[k],\
P1_BH_mass_KH[k])
		r_inf_P2[k] = isothermal.influence_radius(r_eff_P2[k],P2_star_mass[k],\
P2_BH_mass_KH[k])
		
		sigma_P1[k] = isothermal.sigma(P1_star_mass[k],r_eff_P1[k])
		sigma_P2[k] = isothermal.sigma(P2_star_mass[k],r_eff_P2[k])

		# determine host and satellite progenitors
		if(P1_star_mass[k] > P2_star_mass[k]): #P1 is host
			host_r_eff = r_eff_P1[k]
			host_sigma = sigma_P1[k]
			satellite_sigma = sigma_P2[k]
			satellite_BH = P2_BH_mass_KH[k]
			host_BH = P1_BH_mass_KH[k]
		else: # P2 is host
			host_r_eff = r_eff_P2[k]
			host_sigma = sigma_P2[k]
			satellite_sigma = sigma_P1[k]
			satellite_BH = P1_BH_mass_KH[k]
			host_BH = P2_BH_mass_KH[k]
		
		# Calculate the quantities at the influence radius for the remnant galaxy
		# necessary for the hardening timescales
		r_eff[k] = isothermal.effective_radius(bulge_mass[k],stellar_mass[k],
redshift[k])
		#r_scale = dehnen.scale_radius(r_eff[k],gamma)
		r_inf[k] = isothermal.influence_radius(r_eff[k],stellar_mass[k],\
bh_mass_KH[k])
		sigma_inf[k] = isothermal.sigma_inf(bh_mass_KH[k],r_inf[k])
		rho_inf[k] = isothermal.rho_inf(stellar_mass[k],r_inf[k],r_eff[k])


		# Calculate accreton rate to pass to the delay time functions
		if(sfr[k] == 0.0): # we will employ stellar hardening only!
			hardening_type[k] = 1
			m_dot[k] = 0.0
		else: # in this case hardening_type remains 0 by default
			L = 1.16*10**(13)*(sfr[k])**(0.93)
			m_dot[k] = L/(0.1*c**2)




		# Combination (1)
		if (type_P1[k] == 0 and type_P2[k] == 0):
			# BINARY!

			# Find descendant of this merger
			info_descendant = lookback.find_descendant(k,tree_index)
			# convert first 3 outputs to integer value
			info_new = [int(element) for element in info_descendant[:3]]
			# append the last (redshift)
			info_new.append(info_descendant[3])
			info_descendant = info_new

			descendant_index[k] = info_descendant[0]
			# Calculate time elapsed between two subsequent galaxy mergers
			time_to_next_merger[k] = lookback.time_between_mergers(info_descendant[3],\
redshift[k],omega_matter,omega_lambda)
			#print('time between mergers', time_in_mergers)

			'''
			# Record descendant information and update the mass with hot accretion
			# which occurs in between the mergers
			if (info_descendant[0] !=- 1 and info_descendant[1] == 1 and \
info_descendant[2] == 0):
				# it is the progenitor of P1
				P1_BH_mass_KH[info_descendant[0]] = bh_mass_KH[k] + \
bh_mass.radio(time_in_mergers,M_hot[k],V_vir[k],bh_mass_KH[k],stellar_mass[k])

			if(info_descendant[0] !=- 1 and info_descendant[1] == 0 and \
info_descendant[2] == 1):
				P2_BH_mass_KH[info_descendant[0]] = bh_mass_KH[k] + \
bh_mass.radio(time_in_mergers,M_hot[k],V_vir[k],bh_mass_KH[k],stellar_mass[k])
			'''
		
			
			# Calculate the binary merger time
			delay_output = delay_time.tot_delay_function(host_r_eff,host_sigma,\
satellite_sigma,satellite_BH,sigma_inf[k],rho_inf[k],r_inf[k],\
mass1[k],mass2[k],e,m_dot[k],stellar_mass[k],r_eff[k],hardening_type[k])
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
					merger_redshift_vector[k] = lookback.find_redshift(redshift[k],\
time_to_merge[k],omega_matter,omega_lambda)

			if (info_descendant[0] == -1 and time_to_merge[k] < time_to_next_merger[k]):
				binary_vector[k] = 1 # successful merger!
				merger_redshift_vector[k] = lookback.find_redshift(redshift[k],\
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
			time_df[k] = delay_time.time_df2(r_eff[k],host_sigma,\
satellite_sigma,satellite_BH)
			time_df_ph2[k] = delay_time.time_df_phase2(r_eff[k],r_inf[k],\
host_sigma,satellite_sigma,mass1[k],mass2[k])
			time_to_sink = time_df[k] + time_df_ph2[k]

			# Launch triplet interaction
			# Check whether q_out is bigger than 1!
			if (q_out < 1):
				triplet_output = bonetti.triplet_function(m_1,q_in,q_out)
			else:
				triplet_output = bonetti.big_triplet_function(q_out,q_in)

			# Analyze triplet output
			output = triplets.output_analyzer(triplet_output,k,tree_index,redshift[k],\
mass1[k],mass2_1[k],mass2_2[k],time_to_sink,sigma_inf[k],rho_inf[k],\
r_inf[k],m_dot[k],redshift[P2_marker[k]],merger_time_diff_P2[k],hardening_type[k],\
omega_matter,omega_lambda)

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

					merger_redshift_vector[k] = lookback.find_redshift(redshift[k],\
	time_to_merge[k],omega_matter,omega_lambda)

			if(output[0] == -1 and time_to_merge[k] < time_to_next_merger[k]):
				triplet_vector[k] = triplet_vector[k] + output[3]
				ejection_vector[k] = ejection_vector[k] + output[4]
				forced_binary_vector[k] = binary_vector[k] + output[5]
				failed_prompt_vector[k] = failed_prompt_vector[k] + output[6]
				failed_ejection_vector[k] = failed_ejection_vector[k] + output[7]
				failed_failed_vector[k] = failed_failed_vector[k] + output[8]

				merger_redshift_vector[k] = lookback.find_redshift(redshift[k],\
time_to_merge[k],omega_matter,omega_lambda)



			file4.write(f'{m_1},{m_2},{m_3},{output[5]},{output[6]},{output[7]}\n')



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
			print(k, q_out)
				
			# Calculate time to sink due to dynamical friction	
			time_df[k] = delay_time.time_df2(r_eff[k],host_sigma,\
satellite_sigma,satellite_BH)
			time_df_ph2[k] = delay_time.time_df_phase2(r_eff[k],r_inf[k],\
host_sigma,satellite_sigma,mass1[k],mass2[k])
			time_to_sink = time_df[k] + time_df_ph2[k]

			# Launch triplet interaction
			# Check whether q_out is bigger than 1!
			if (q_out < 1):
				triplet_output = bonetti.triplet_function(m_1,q_in,q_out)
			else:
				triplet_output = bonetti.big_triplet_function(q_out,q_in)

			# Analyze triplet output
			output = triplets.output_analyzer(triplet_output,k,tree_index,redshift[k],\
mass2[k],mass1_1[k],mass1_2[k],time_to_sink,sigma_inf[k],rho_inf[k],\
r_inf[k],m_dot[k],redshift[P1_marker[k]],merger_time_diff_P1[k],hardening_type[k],\
omega_matter,omega_lambda)

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

					merger_redshift_vector[k] = lookback.find_redshift(redshift[k],\
	time_to_merge[k],omega_matter,omega_lambda)

			if(output[0] == -1 and time_to_merge[k] < time_to_next_merger[k]):
				triplet_vector[k] = triplet_vector[k] + output[3]
				ejection_vector[k] = ejection_vector[k] + output[4]
				forced_binary_vector[k] = binary_vector[k] + output[5]
				failed_prompt_vector[k] = failed_prompt_vector[k] + output[6]
				failed_ejection_vector[k] = failed_ejection_vector[k] + output[7]
				failed_failed_vector[k] = failed_failed_vector[k] + output[8]

				merger_redshift_vector[k] = lookback.find_redshift(redshift[k],\
time_to_merge[k],omega_matter,omega_lambda)

			file4.write(f'{m_1},{m_2},{m_3},{output[5]},{output[6]},{output[7]}\n')



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
				print(k, q_out)


				# Calculate time to sink due to dynamical friction	
				time_df[k] = delay_time.time_df2(r_eff[k],host_sigma,\
satellite_sigma,satellite_BH)
				time_df_ph2[k] = delay_time.time_df_phase2(r_eff[k],r_inf[k],\
host_sigma,satellite_sigma,mass1[k],mass2[k])
				time_to_sink = time_df[k] + time_df_ph2[k]

				# Launch triplet interaction
				# Check whether q_out is bigger than 1!
				if (q_out < 1):
					triplet_output = bonetti.triplet_function(m_1,q_in,q_out)
				else:
					triplet_output = bonetti.big_triplet_function(q_out,q_in)

				# Analyze triplet output
				output = triplets.output_analyzer(triplet_output,k,tree_index,redshift[k],\
mass1_2[k],mass2_1[k],mass2_2[k],time_to_sink,sigma_inf[k],rho_inf[k],\
r_inf[k],m_dot[k],redshift[P2_marker[k]],merger_time_diff_P2[k],hardening_type[k],\
omega_matter,omega_lambda)

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

						merger_redshift_vector[k] = lookback.find_redshift(redshift[k],\
time_to_merge[k],omega_matter,omega_lambda)

				if(output[0] == -1 and time_to_merge[k] < time_to_next_merger[k]):
					triplet_vector[k] = triplet_vector[k] + output[3]
					ejection_vector[k] = ejection_vector[k] + output[4]
					forced_binary_vector[k] = binary_vector[k] + output[5]
					failed_prompt_vector[k] = failed_prompt_vector[k] + output[6]
					failed_ejection_vector[k] = failed_ejection_vector[k] + output[7]
					failed_failed_vector[k] = failed_failed_vector[k] + output[8]

					merger_redshift_vector[k] = lookback.find_redshift(redshift[k],\
time_to_merge[k],omega_matter,omega_lambda)


				file4.write(f'{m_1},{m_2},{m_3},{output[5]},{output[6]},{output[7]}\n')


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
				print(k, q_out)

				# Calculate time to sink due to dynamical friction	
				time_df[k] = delay_time.time_df2(r_eff[k],host_sigma,\
satellite_sigma,satellite_BH)
				time_df_ph2[k] = delay_time.time_df_phase2(r_eff[k],r_inf[k],\
host_sigma,satellite_sigma,mass1[k],mass2[k])
				time_to_sink = time_df[k] + time_df_ph2[k]



				# Launch triplet interaction
				# Check whether q_out is bigger than 1!
				if (q_out<1):
					triplet_output=bonetti.triplet_function(m_1,q_in,q_out)
				else:
					triplet_output=bonetti.big_triplet_function(q_out,q_in)

				# Analyze triplet output
				output = triplets.output_analyzer(triplet_output,k,tree_index,redshift[k],\
mass1_1[k],mass2_1[k],mass2_2[k],time_to_sink,sigma_inf[k],rho_inf[k],\
r_inf[k],m_dot[k],redshift[P2_marker[k]],merger_time_diff_P2[k],hardening_type[k],\
omega_matter,omega_lambda)

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

						merger_redshift_vector[k] = lookback.find_redshift(redshift[k],\
time_to_merge[k],omega_matter,omega_lambda)

				if(output[0] == -1 and time_to_merge[k] < time_to_next_merger[k]):
					triplet_vector[k] = triplet_vector[k] + output[3]
					ejection_vector[k] = ejection_vector[k] + output[4]
					forced_binary_vector[k] = binary_vector[k] + output[5]
					failed_prompt_vector[k] = failed_prompt_vector[k] + output[6]
					failed_ejection_vector[k] = failed_ejection_vector[k] + output[7]
					failed_failed_vector[k] = failed_failed_vector[k] + output[8]

					merger_redshift_vector[k] = lookback.find_redshift(redshift[k],\
time_to_merge[k],omega_matter,omega_lambda)


				file4.write(f'{m_1},{m_2},{m_3},{output[5]},{output[6]},{output[7]}\n')


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
				print(k, q_out)


				# Calculate time to sink due to dynamical friction	
				time_df[k] = delay_time.time_df2(r_eff[k],host_sigma,\
satellite_sigma,satellite_BH)
				time_df_ph2[k] = delay_time.time_df_phase2(r_eff[k],r_inf[k],\
host_sigma,satellite_sigma,mass1[k],mass2[k])
				time_to_sink = time_df[k] + time_df_ph2[k]



				# Launch triplet interaction
				# Check whether q_out is bigger than 1!
				if (q_out < 1):
					triplet_output = bonetti.triplet_function(m_1,q_in,q_out)
				else:
					triplet_output = bonetti.big_triplet_function(q_out,q_in)

				# Analyze triplet output
				output = triplets.output_analyzer(triplet_output,k,tree_index,redshift[k],\
mass2_2[k],mass1_1[k],mass1_2[k],time_to_sink,sigma_inf[k],rho_inf[k],\
r_inf[k],m_dot[k],redshift[P1_marker[k]],merger_time_diff_P1[k],hardening_type[k],\
omega_matter,omega_lambda)

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

						merger_redshift_vector[k] = lookback.find_redshift(redshift[k],\
time_to_merge[k],omega_matter,omega_lambda)

				if(output[0] == -1 and time_to_merge[k] < time_to_next_merger[k]):
					triplet_vector[k] = triplet_vector[k] + output[3]
					ejection_vector[k] = ejection_vector[k] + output[4]
					forced_binary_vector[k] = binary_vector[k] + output[5]
					failed_prompt_vector[k] = failed_prompt_vector[k] + output[6]
					failed_ejection_vector[k] = failed_ejection_vector[k] + output[7]
					failed_failed_vector[k] = failed_failed_vector[k] + output[8]

					merger_redshift_vector[k] = lookback.find_redshift(redshift[k],\
time_to_merge[k],omega_matter,omega_lambda)

				file4.write(f'{m_1},{m_2},{m_3},{output[5]},{output[6]},{output[7]}\n')



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
				time_df[k] = delay_time.time_df2(r_eff[k],host_sigma,\
satellite_sigma,satellite_BH)
				time_df_ph2[k] = delay_time.time_df_phase2(r_eff[k],r_inf[k],\
host_sigma,satellite_sigma,mass1[k],mass2[k])
				time_to_sink = time_df[k] + time_df_ph2[k]


				# Launch triplet interaction
				# Check whether q_out is bigger than 1!
				if (q_out < 1):
					triplet_output = bonetti.triplet_function(m_1,q_in,q_out)
				else:
					triplet_output = bonetti.big_triplet_function(q_out,q_in)

				# Analyze triplet output
				output = triplets.output_analyzer(triplet_output,k,tree_index,redshift[k],\
mass2_1[k],mass1_1[k],mass1_2[k],time_to_sink,sigma_inf[k],rho_inf[k],\
r_inf[k],m_dot[k],redshift[P1_marker[k]],merger_time_diff_P1[k],hardening_type[k],
omega_matter,omega_lambda)

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

						merger_redshift_vector[k] = lookback.find_redshift(redshift[k],\
time_to_merge[k],omega_matter,omega_lambda)

				if(output[0] == -1 and time_to_merge[k] < time_to_next_merger[k]):
					triplet_vector[k] = triplet_vector[k] + output[3]
					ejection_vector[k] = ejection_vector[k] + output[4]
					forced_binary_vector[k] = binary_vector[k] + output[5]
					failed_prompt_vector[k] = failed_prompt_vector[k] + output[6]
					failed_ejection_vector[k] = failed_ejection_vector[k] + output[7]
					failed_failed_vector[k] = failed_failed_vector[k] + output[8]

					merger_redshift_vector[k] = lookback.find_redshift(redshift[k],\
time_to_merge[k],omega_matter,omega_lambda)

				file4.write(f'{m_1},{m_2},{m_3},{output[5]},{output[6]},{output[7]}\n')

		

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

file4.close()

file1 = open('Data/OutputData/TreeAnalysis/output_%s.csv' %catalogue,'w')
file1.write(f'mass1,mass2,binary,triplet,ejection,quadruplet,forced,\
failed_prompt_vector,failed_ejection_vector,failed_failed_vector,redshift,\
redshift_merger,merger_time,time_to_next_merger,time_df,time_df_ph2,time_star,\
time_gas,time_gw,long_df_time\n')
for i in range(len(galaxyId)):
	if(galaxyId[i] != -1):
		file1.write(f'{mass1[i]},{mass2[i]},{binary_vector[i]},{triplet_vector[i]},\
{ejection_vector[i]},{quadruplet_vector[i]},{forced_binary_vector[i]},\
{failed_prompt_vector[i]},{failed_ejection_vector[i]},{failed_failed_vector[i]},\
{redshift[i]},{merger_redshift_vector[i]},{time_to_merge[i]},{time_to_next_merger[i]},\
{time_df[i]},{time_df_ph2[i]},{time_star[i]},{time_gas[i]},{time_gw[i]},{long_df_time[i]}\n')
	else:
		file1.write(f'-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1\n')
file1.close()



file2 = open('Data/OutputData/ModelData/model_data_%s.csv' %catalogue,'w')
file2.write(f'r_eff,r_inf,sigma_inf,rho_inf,sigma\n')
for i in range(len(galaxyId)):
	if(galaxyId[i] != -1):
		file2.write(f'{r_eff[i]},{r_inf[i]},{sigma_inf[i]},{rho_inf[i]},{sigma[i]}\n')
	else:
		file2.write(f'-1,-1,-1,-1,-1\n')

file2.close()


file3 = open('Data/OutputData/MassData/kh_mass_%s.csv' %catalogue,'w')
file3.write(f'kh_mass,p1_kh_mass,p2_kh_mass,mass1,mass2,mass1_1,mass1_2,mass2_1,\
mass2_2\n')
for i in range(len(galaxyId)):
	if(galaxyId[i] != -1):
		file3.write(f'{bh_mass_KH[i]},{P1_BH_mass_KH[i]},{P2_BH_mass_KH[i]},{mass1[i]},\
{mass2[i]},{mass1_1[i]},{mass1_2[i]},{mass2_1[i]},{mass2_2[i]}\n')
	else:
		file3.write(f'-1,-1,-1,-1,-1,-1,-1,-1,-1\n')
file3.close()


file4 = open('Data/OutputData/ModelData/model_data_prog_%s.csv' %catalogue,'w')
file4.write(f'r_eff_P1,r_eff_P2,r_inf_P1,r_inf_P2,sigma_P1,sigma_P2\n')
for i in range(len(galaxyId)):
	if(galaxyId[i] != -1):
		file4.write(f'{r_eff_P1[i]},{r_eff_P2[i]},{r_inf_P1[i]},{r_inf_P2[i]},\
{sigma_P1[i]},{sigma_P2[i]}\n')
	else:
		file4.write(f'-1,-1,-1,-1,-1,-1\n')

file4.close()

file5 = open('Data/OutputData/IndexData/index_data_%s.csv' %catalogue, 'w')
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


