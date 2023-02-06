import numpy as np
import lal


G = lal.G_SI
c = lal.C_SI
pc = lal.PC_SI

M_sun = lal.MSUN_SI
mass_conv = (10.**(10.)) # For Millennium masses

H = 15 # stellar hardening constant
kappa = 1.5*10**(-5) # radio mode accretion constant
f = 0.03 # quasar mode accretion constant

LM = 500

freq_1yr = 3.17098*10**(-8)
t_1yr = 3.15*10**7
Gyr = 3.15*10**16
TH0 = 13.4 #(Hubble time, in Gyr)
H0 = 100 * 10**3 / (10**6 * pc) # in h units

gamma = 1 # parameter of the Dehnen profile (Hernquist if gamma=1)

e = 0. # default eccentricity value

#acc_rate = 0.1
#sigma_thompson = 6.65*10**(-29)
#m_protone = 1.67*10**(-27)


# To convert from meters to parsec and from kg to Msol
length_conv_new = 1./pc
mass_conv_new = 1./M_sun
# [G] = [L]^3 [M]^(-1) [T]^(-2) dimensional analysis of G
# [c] = [L] [T]^(-1) dimensional analysis of c
G_new = G*(length_conv_new)**3/mass_conv_new
c_new = c*length_conv_new

