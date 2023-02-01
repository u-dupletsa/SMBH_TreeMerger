#############################################################
# Module: triplets.py
#
# Contains a single function that analyzes the output of
# the triple interaction, as from the bonetti's module
#
#    --> output_analyzer(triplet_output,k,tree_index,	
#			redshift,mass_int,mass_b1,mass_b2,	
#			time_to_sink,sigma_inf,rho_inf,		
#			r_inf,m_dot,previous_redshift,		
#			previous_merger_diff,hardening_type,
#			omega_matter,omega_lambda,snapnum,
#			galaxyId,P1_galaxyId,P2_galaxyId,
#			z_tree)
#
# !Further information is provided below the function
#############################################################

import math
import numpy as np
import random

import delay_time
import lookback
import constants as cst



def output_analyzer(triplet_output,k,tree_index,redshift,mass_int,mass_b1,mass_b2,\
time_to_sink,sigma_inf,rho_inf,r_inf,m_dot,previous_redshift,previous_merger_diff,\
hardening_type,omega_matter,omega_lambda,snapnum,galaxyId,P1_galaxyId,P2_galaxyId,z_tree):
	"""
	Function to analyze the output (integer between 1 and 7) of triple interaction:
		j==1: prompt merger between m_1 and m_2
		j==2: ejection of m_3
		j==3: prompt merger between m_1 and m_3
		j==4: ejection of m_2
		j==5: prompt merger between m_2 and m_3
		j==6: ejection of m_1
		j==7: unresolved triplet

	input parameters:
		triplet_output -> integer number between 1 and 7, j, which represents
						  the output of the bonetti's module
		k -> index of the merger in the data set
		tree_index -> index that signals the end of the tree to which merger
					  k belongs
		redshift -> redshift of the merger k
		mass_int -> mass of the intruder (in solar masses)
		mass_b1, mass_b2 -> masses of the internal binary black holes (in solar
							masses)
		time_to_sink -> dynamical friction time (in Gyr) between the intruder and
						the internal binary galaxies
		sigma_inf -> velocity dispersion at r_inf of the remnant galaxy
		             (in km/s)
		rho_inf -> density at r_inf of the remnant galaxy (in solar masses
				   pc^3)
		r_inf -> influence radius of the remnant galaxy (in pc)
		m_dot -> accretion rate of the remnant (in solar masses per second)
		previous_redshift -> redshift of internal binary merger
		previous_merger_diff -> difference between the internal binary delay
								time and the time elapsed between two 
								consecutive galaxy mergers
		hardening_type -> integer value that indicates whether the 
						  hardening process could be both stellar and 
						  gaseous (hardening_type == 0), or stellar
						  only in case sfr=0 (hardening_type == 1)
		omega_matter,omega_lambda -> values of cosmological parameters
		snapnum -> snapnum of the merger
		galaxyId ->
		P1_galaxyId ->
		P2_galaxyId ->
		z_tree -> 


	return:
		vector containing information on the outcome and on the descendant
		merger

			-> ([info_descendant[0],P1,P2,type_P1,type_P2,T,E,F,merger_redshift,
                 mass1,mass2,time_to_merge,time_star,time_gas,time_gw])

                info_descendant[0] -> index of the descendant merger
                P1,P2 -> could be 0 or 1, signaling whether the descendant is
                		 the first progenitor (P1=1 and P2=0) or the second
                		 progenitor (P1=0 and P2=1)
                type_P1,type_P2 -> could be either 0 or 2, signaling whether
                				   descendants will be a single black hole (0)
                				   or a binary (1)
                T,E,F -> could be 0 or 1, specifying if the triple interaction
                		 was a prompt merger (1,0,0), an ejection (0,1,0) or
                		 an unresolved triplet (0,0,1)
                merger_redshift -> z of the merger in case of prompt or ejection
                				   outcome (-1 otherwise)
                mass1, mass2 -> final masses of the binary components after the
                				interaction (in solar masses)
                time_to_merge -> merger time (in Gyr) for the binary after the triple
                				 interaction, given also as single contributes:
								 -->time_star 
								 -->time_gas
								 -->time_gw
	"""

	Gyr = 3.15*10**16
	start_time = 3*10**8/Gyr
	e = cst.ecc

	merger_diff = 0
	
	type_P1 = -1
	type_P2 = -1

	prompt_plus_delayed = 0 # successful prompt merger
	ejection_plus_delayed = 0 # ejection + successful delayed merger
	forced_binary = 0 # unresolved triplet -> forced binary!
	prompt_plus_failed_delayed = 0 #failed prompt
	ejection_plus_failed_delayed = 0 #failed ejection
	failde_forced_binary = 0 #failed failed triplet
	still_merging = 0

	merger_redshift = 0
	
	if(mass_b1 > mass_b2):
		m_1 = mass_b1
		m_2 = mass_b2
	else:
		m_1 = mass_b2
		m_2 = mass_b1
	m_3 = mass_int


	if (triplet_output == 1 or triplet_output == 3 or triplet_output == 5): # Prompt merger!
		if (triplet_output == 1):
			# prompt merger m_1+m_2
			# forms binary m_12+m_3
			mass1 = m_1 + m_2
			mass2 = m_3
			if(mass1 > mass2):
				q_bin = mass2/mass1
			else:
				q_bin = mass1/mass2
		if (triplet_output == 3):
			# prompt merger m_1+m_3
			# froms binary m_13+m_2
			mass1 = m_1 + m_3
			mass2 = m_2
			if(mass1 > mass2):
				q_bin = mass2/mass1
			else:
				q_bin = mass1/mass2
		if (triplet_output == 5):
			# prompt merger between m_2+m_3
			# forms binary m_23+m_1
			mass1 = m_2 + m_3
			mass2 = m_1
			if(mass1 > mass2):
				q_bin = mass2/mass1
			else:
				q_bin = mass1/mass2

		time_no_df, time_star, time_gas, time_gw = delay_time.tot_delay_no_df(sigma_inf,rho_inf,r_inf,mass1,mass2,e,m_dot,hardening_type)
		time_to_merge = time_to_sink + time_no_df

		info_descendant = lookback.find_descendant(k,tree_index,snapnum,galaxyId,P1_galaxyId,P2_galaxyId,z_tree)

		next_redshift = info_descendant[3]
		time_between_mergers = lookback.time_between_mergers(next_redshift,redshift,omega_matter,omega_lambda)

		if(info_descendant[0] != -1):

			if (time_to_merge > time_between_mergers):

				#if(time_to_sink < time_between_mergers or (time_to_sink >= time_between_mergers and q_bin > 0.03)):

				prompt_plus_failed_delayed = 1

				merger_diff = time_to_merge - time_between_mergers

				if (info_descendant[1] == 1 and info_descendant[2] == 0):
					# P1 of descendant is a binary!
					type_P1 = 2
					type_P2 = 0

				if (info_descendant[1] == 0 and info_descendant[2] == 1):
					# P2 of descendant is a binary
					type_P1 = 0
					type_P2 = 2
			else:
				type_P1 = 0
				type_P2 = 0
				prompt_plus_delayed = 1
				merger_redshift = lookback.find_redshift(redshift,time_to_merge,omega_matter,omega_lambda)

		if(info_descendant[0] == -1 and time_to_merge < time_between_mergers):
			type_P1 = 0
			type_P2 = 0
			prompt_plus_delayed = 1
			merger_redshift = -1
		if(info_descendant[0] == -1 and time_to_merge > time_between_mergers):
			type_P1 = 0
			type_P2 = 0
			still_merging = 1
			merger_redshift = -1


	

	if (triplet_output == 2 or triplet_output == 4 or triplet_output == 6): # Ejection plus delayed merger!
		if (triplet_output == 2):
			# ejection of m_3
			# delayed merger between m_1+m_2
			mass1 = m_1
			mass2 = m_2
			if(mass1 > mass2):
				q_bin = mass2/mass1
			else:
				q_bin = mass1/mass2
		if (triplet_output == 4):
			# ejection of m_2
			# delayed merger between m_1+m_3
			mass1 = m_1
			mass2 = m_3
			if(mass1 > mass2):
				q_bin = mass2/mass1
			else:
				q_bin = mass1/mass2
		if (triplet_output == 6):
			# ejection of m_1
			# delayed merger m_2+m_3
			mass1 = m_2
			mass2 = m_3
			if(mass1 > mass2):
				q_bin = mass2/mass1
			else:
				q_bin = mass1/mass2


		info_descendant = lookback.find_descendant(k,tree_index,snapnum,galaxyId,P1_galaxyId,P2_galaxyId,z_tree)
		next_redshift = info_descendant[3]
		
		time_between_mergers = lookback.time_between_mergers(next_redshift,redshift,omega_matter,omega_lambda)
		time_to_merge = time_to_sink + random.uniform(start_time,time_between_mergers)

		time_no_df, time_star, time_gas, time_gw = delay_time.tot_delay_no_df(sigma_inf,rho_inf,r_inf,mass1,mass2,e,m_dot,hardening_type)
		

		if(info_descendant[0] != -1):

			if (time_to_merge > time_between_mergers):
				#if(time_to_sink < time_between_mergers or (time_to_sink >= time_between_mergers and q_bin > 0.03)):

					ejection_plus_failed_delayed = 1

					merger_diff = time_to_merge - time_between_mergers

					if (info_descendant[1] == 1 and info_descendant[2] == 0):
						# P1 of descendant is a binary!
						type_P1 = 2
						type_P2 = 0

					if (info_descendant[1] == 0 and info_descendant[2] == 1):
						# P2 of descendant is a binary
						type_P1 = 0
						type_P2 = 2
			else:
				type_P1 = 0
				type_P2 = 0
				ejection_plus_delayed = 1
				merger_redshift = lookback.find_redshift(redshift,time_to_merge,omega_matter,omega_lambda)
		
		if(info_descendant[0] == -1 and time_to_merge < time_between_mergers):
			type_P1 = 0
			type_P2 = 0
			ejection_plus_delayed = 1
			merger_redshift = -1
		if(info_descendant[0] == -1 and time_to_merge > time_between_mergers):
			type_P1 = 0
			type_P2 = 0
			still_merging = 1
			merger_redshift = -1


	if (triplet_output == 7):
		# no interaction has happened
		# select two most massive BHs
	
		mass_vector = ([m_1,m_2,m_3])
		mass_vector.sort(reverse=True)
		mass1 = mass_vector[0]
		mass2 = mass_vector[1]
		if(mass1 > mass2):
			q_bin = mass2/mass1
		else:
			q_bin = mass1/mass2
		
		info_descendant = lookback.find_descendant(k,tree_index,snapnum,galaxyId,P1_galaxyId,P2_galaxyId,z_tree)
		next_redshift = info_descendant[3]
		
		time_between_mergers = lookback.time_between_mergers(next_redshift,redshift,omega_matter,omega_lambda)
		
		time_no_df, time_star, time_gas, time_gw = delay_time.tot_delay_no_df(sigma_inf,rho_inf,r_inf,mass1,mass2,e,m_dot,hardening_type)


		time_to_merge = previous_merger_diff

		if(info_descendant[0] != -1):

			if (time_to_merge > time_between_mergers):
				#if(time_to_sink < time_between_mergers or (time_to_sink >= time_between_mergers and q_bin > 0.03)):

					failde_forced_binary = 1

					merger_diff = time_to_merge - time_between_mergers

					if (info_descendant[1] == 1 and info_descendant[2] == 0):
						# P1 of descendant is a binary!
						type_P1 = 2
						type_P2 = 0

					if (info_descendant[1] == 0 and info_descendant[2] == 1):
						# P2 of descendant is a binary
						type_P1 = 0
						type_P2 = 2
			else:
				type_P1 = 0
				type_P2 = 0
				forced_binary = 1
				merger_redshift = lookback.find_redshift(redshift,time_to_merge,omega_matter,omega_lambda)

		if(info_descendant[0] == -1 and time_to_merge < time_between_mergers):
			type_P1 = 0
			type_P2 = 0
			forced_binary = 1
			merger_redshift = -1
		if(info_descendant[0] == -1 and time_to_merge > time_between_mergers):
			type_P1 = 0
			type_P2 = 0
			still_merging = 1
			merger_redshift = -1


	
	return int(info_descendant[0]), int(type_P1), int(type_P2), int(prompt_plus_delayed), int(ejection_plus_delayed),\
		   int(forced_binary), int(prompt_plus_failed_delayed), int(ejection_plus_failed_delayed), int(failde_forced_binary),\
		   int(still_merging), merger_redshift, mass1, mass2, time_to_merge, time_star, time_gas,\
		   time_gw, merger_diff, time_between_mergers, q_bin





