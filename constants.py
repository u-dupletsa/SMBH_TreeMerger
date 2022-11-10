import numpy as np
import lal


omega_matter = 0.272
omega_lambda = 0.728
h = 0.704

M_sun = lal.MSUN_SI
mass_conv = (10**(10))/h
G = lal.G_SI
c = 299792458.
pc = lal.PC_SI
H = 15
freq_1yr = 3.17098*10**(-8)
t_1yr = 3.15*10**7
Gyr = 3.15*10**16
TH0 = 13.4 #(Hubble time, in Gyr)
acc_rate = 0.1
gamma = 1 # parameter of the Dehnen profile (Hernquist if gamma=1)


