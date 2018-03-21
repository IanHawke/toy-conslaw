# Sod shock tube

import numpy
from models import sr_swe
from bcs import outflow
from simulation import simulation
from methods import fvs_method
from rk import rk3
from grid import grid
from matplotlib import pyplot

Ngz = 4
Npoints = 400
L = 0.5
interval = grid([-L, L], Npoints, Ngz)

phiL = 0.41
phiR = 0.01
phiL = 0.9
phiR = 0.05
qL = numpy.array([phiL, 0])
qR = numpy.array([phiR, 0])
model = sr_swe.sr_swe(initial_data = sr_swe.initial_riemann(qL, qR))

sim = simulation(model, interval, fvs_method(3), rk3, outflow, cfl=0.5)
sim.evolve(0.4)
sim.plot_system()
pyplot.show()
#
#exact = numpy.genfromtxt('../sr_swe.txt')
#fig, ax = pyplot.subplots(2, 1)
#ax[0].plot(exact[0, :]*0.4, exact[1, :], 'k-', label='Exact')
#ax[0].plot(sim.coordinates, sim.prim[0, :], 'ro', label='Sim')
#ax[0].set_ylabel(r"$\Phi$")
#ax[1].plot(exact[0, :]*0.4, exact[2, :], 'k-', label='Exact')
#ax[1].plot(sim.coordinates, sim.prim[1, :], 'ro', label='Sim')
#ax[1].set_ylabel(r"$v$")
#fig.tight_layout()
#pyplot.show()
