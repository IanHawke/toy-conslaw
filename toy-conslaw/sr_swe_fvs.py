# Sod shock tube

import numpy
from models import sr_swe
from bcs import outflow
from simulation import simulation
from methods import fvs_method
from rk import rk3
from grid import grid
from matplotlib import pyplot

Ngz = 3
Npoints = 200
L = 0.5
interval = grid([-L, L], Npoints, Ngz)

phiL = 0.41
phiR = 0.01
qL = numpy.array([phiL, 0])
qR = numpy.array([phiR, 0])
model = sr_swe.sr_swe(initial_data = sr_swe.initial_riemann(qL, qR))

sim = simulation(model, interval, fvs_method(3), rk3, outflow, cfl=0.5)
sim.evolve(0.3)
sim.plot_system()
pyplot.show()
