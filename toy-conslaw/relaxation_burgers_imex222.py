# Relaxation system: relaxes to Burgers as tau->0
#

import numpy
from models import relaxation_burgers
from bcs import outflow
from simulation import simulation
from methods import weno3_upwind
from rk import imex222
from grid import grid
from matplotlib import pyplot

Ngz = 4
Npoints = 200
tau = 1.0
tau2 = 0.01
L = 1
interval = grid([-L, L], Npoints, Ngz)
qL = numpy.array([1.0, 0.5])
qR = numpy.array([0.0, 0.0])
model = relaxation_burgers.relaxation_burgers(initial_data = relaxation_burgers.initial_riemann(qL, qR))
source  = relaxation_burgers.relaxation_source(tau)
source2 = relaxation_burgers.relaxation_source(tau2)

sim = simulation(model, interval, weno3_upwind, imex222(source), outflow)
sim.evolve(0.8)
sim.plot_system()
pyplot.show()

sim2 = simulation(model, interval, weno3_upwind, imex222(source2), outflow)
sim2.evolve(0.8)
sim2.plot_system()
pyplot.show()
