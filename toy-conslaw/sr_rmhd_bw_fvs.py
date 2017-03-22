# Brio-Wu test (see KD paper, thesis)

import numpy
from models import sr_rmhd
from bcs import outflow
from simulation import simulation
from methods import fvs_method
from rk import imex222, rk3
from grid import grid
from matplotlib import pyplot

Ngz = 3
Npoints = 80
L = 0.5
interval = grid([-L, L], Npoints, Ngz)

rhoL = 1
pL = 1
rhoR = 0.125
pR = 0.1
vyL = 0
vyR = 0
vzL = 0
vzR = 0
Bx = 0
ByL = 0.5
ByR =-0.5
BzL = 0
BzR = 0
gamma = 5/3
sigma = 0
epsL = pL / rhoL / (gamma - 1)
epsR = pR / rhoR / (gamma - 1)
qL = numpy.array([rhoL, 0, vyL, vzL, epsL, Bx , ByL , BzL, 0, 0, 0, 0, 0 ])
qR = numpy.array([rhoR, 0, vyR, vzR, epsR, Bx , ByR , BzR, 0, 0, 0, 0, 0 ])
model = sr_rmhd.sr_rmhd_gamma_law(initial_data = sr_rmhd.initial_riemann(qL, qR),
                                gamma=gamma, sigma=sigma)
fast_source  = model.relaxation_source()

sim = simulation(model, interval, fvs_method(2), imex222(fast_source), 
                 outflow, cfl=0.25)
sim.evolve(0.4)
sim.plot_system()
pyplot.show()
