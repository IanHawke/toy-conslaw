# Granular gas example from Pareschi
#
# How are we getting the characteristic speeds?
#
# Does not currently work

import numpy
from models import euler_relaxation
from bcs import outflow_reflect_right
from simulation import simulation
from methods import fvs_method
from rk import imex222
from grid import grid
from matplotlib import pyplot

Ngz = 4
Npoints = 100
tau = 1
L = 10
interval = grid([0, L], Npoints, Ngz)

# Initial data. (Mis)use RP for trivial constant data
rho_init = 34.37746770
v_init = 18
p_init = 1589.2685472 # Not actually used, but in the paper
T_init = 43.0351225511 # Hand calculation, works only for default values
eps_init = T_init / (5/3 - 1)
E_init = rho_init * (v_init**2 / 2 + eps_init)
qL = numpy.array([rho_init,  v_init, eps_init])
qR = numpy.array([rho_init,  v_init, eps_init])
model = euler_relaxation.euler_relaxation(initial_data = euler_relaxation.initial_riemann(qL, qR))
slow_source = model.source()
fast_source  = model.relaxation_source(tau)

sim = simulation(model, interval, fvs_method(2, slow_source), 
                 imex222(fast_source), outflow_reflect_right, cfl=0.25)
sim.evolve(0.035)
sim.plot_system()
pyplot.show()
