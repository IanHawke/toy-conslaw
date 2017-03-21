# Second blast wave test
# Oscillations are too large - get superluminal v

import numpy
from models import sr_euler_gamma_law
from bcs import outflow
from simulation import simulation
from methods import fvs_method
from rk import rk3
from grid import grid
from matplotlib import pyplot

Ngz = 3
Npoints = 800
L = 0.5
interval = grid([-L, L], Npoints, Ngz)

rhoL = 1
rhoR = 1
epsL = 1.5e3
epsR = 1.5e-2
qL = numpy.array([rhoL, 0, 0, 0, epsL])
qR = numpy.array([rhoR, 0, 0, 0, epsR])
model = sr_euler_gamma_law.sr_euler_gamma_law(initial_data = sr_euler_gamma_law.initial_riemann(qL, qR))

sim = simulation(model, interval, fvs_method(2), rk3, outflow, cfl=0.5)
sim.evolve(0.4)
sim.plot_system()
pyplot.show()
