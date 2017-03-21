# GR2 test from Liv Rev

import numpy
from models import sr_mhd
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
pL = 5
rhoR = 0.9
pR = 5.3
vyL = 0.3
vyR = 0
vzL = 0.4
vzR = 0
Bx = 1
ByL = 6
ByR = 5
BzL = 2
BzR = 2
gamma = 5/3
epsL = pL / rhoL / (gamma - 1)
epsR = pR / rhoR / (gamma - 1)
qL = numpy.array([rhoL, 0, vyL, vzL, epsL, Bx , ByL , BzL ])
qR = numpy.array([rhoR, 0, vyR, vzR, epsR, Bx , ByR , BzR ])
model = sr_mhd.sr_mhd_gamma_law(initial_data = sr_mhd.initial_riemann(qL, qR), gamma=gamma)

sim = simulation(model, interval, fvs_method(2), rk3, outflow, cfl=0.5)
sim.evolve(0.4)
sim.plot_system()
pyplot.show()
