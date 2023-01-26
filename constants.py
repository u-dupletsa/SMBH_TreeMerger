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

freq_1yr = 3.17098*10**(-8)
t_1yr = 3.15*10**7
Gyr = 3.15*10**16
TH0 = 13.4 #(Hubble time, in Gyr)

gamma = 1 # parameter of the Dehnen profile (Hernquist if gamma=1)

ecc = 0. # default eccentricity value

acc_rate = 0.1
sigma_thompson = 6.65*10**(-29)
m_protone = 1.67*10**(-27)


# To convert from kms in solar masses and parsec
length_conv_new = 0.324*10**(-16)
mass_conv_new = 0.5025*10**(-30)
# [G] = [L]^3 [M]^(-1) [T]^(-2)
# [c] = [L] [T]
G_new = G*(length_conv_new)**3/mass_conv_new
c_new = c*length_conv_new

