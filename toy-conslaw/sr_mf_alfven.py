# Alfven wave test in MF (see KD paper, thesis)
# Using the MF code, so see Amano 2016

from models import sr_mf
from bcs import periodic
from simulation import simulation
from methods import fvs_method
from rk import imex222, rk3, rk_backward_euler_split, rk_euler_split
from grid import grid
from matplotlib import pyplot

Ngz = 3
Npoints = 100
L = 0.5
interval = grid([-L, L], Npoints, Ngz)

m_p = 1.
m_e = 1.
gamma = 4.0/3.0

model_mf = sr_mf.sr_mf_gamma_law(initial_data = sr_mf.initial_alfven(gamma=gamma, Kappa_f=1e80),
                                 gamma=gamma,
                                 kappa_m = 0.05066059182116889, kappa_f = 1.0e80, kappa_q = 1.0)

fast_source_mf  = model_mf.relaxation_source()
#timestepper = rk_backward_euler_split(rk3, fast_source_mf)
#timestepper = rk_euler_split(rk3, fast_source_mf)
timestepper = rk3
sim_mf = simulation(model_mf, interval, fvs_method(2), timestepper, 
                    periodic, cfl=0.25)

sim_mf.evolve(0.5)
