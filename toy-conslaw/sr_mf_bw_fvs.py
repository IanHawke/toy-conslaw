# Brio-Wu test (see KD paper, thesis)
# Using the MF code, so see Amano 2016

import numpy
from models import sr_mf, sr_rmhd
from bcs import outflow
from simulation import simulation
from methods import fvs_method, vanleer_lf
from rk import imex222, rk3
from grid import grid
from matplotlib import pyplot
from cycler import cycler

Ngz = 3
Npoints = 200
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
Bx = 0.5
ByL = 1.0
ByR =-1.0
BzL = 0
BzR = 0
gamma = 2.0
epsL = pL / rhoL / (gamma - 1)
epsR = pR / rhoR / (gamma - 1)

rhoL_p = rhoL
rhoL_e = rhoL
rhoR_p = rhoR
rhoR_e = rhoR

kappa_m = 1e-2
kappa_q = 1e-3
kappa_f = 1e50

qL = numpy.array([rhoL_e, 0, 0, 0, epsL, rhoL_p, 0, 0, 0, epsL, Bx , ByL , BzL, 0, 0, 0, 0, 0 ])
qR = numpy.array([rhoR_e, 0, 0, 0, epsR, rhoR_p, 0, 0, 0, epsR, Bx , ByR , BzR, 0, 0, 0, 0, 0 ])

model_mf = sr_mf.sr_mf_gamma_law(initial_data = sr_mf.initial_riemann(qL, qR),
                                 gamma=gamma,
                                 kappa_m = kappa_m,
                                 kappa_f = kappa_f,
                                 kappa_q = kappa_q)
fast_source_mf  = model_mf.relaxation_source()

sim_mf = simulation(model_mf, interval, fvs_method(2), imex222(fast_source_mf), 
                    outflow, cfl=0.25)

qL = numpy.array([rhoL, 0, 0, 0, epsL, Bx , ByL , BzL, 0, 0, 0, 0, 0 ])
qR = numpy.array([rhoR, 0, 0, 0, epsR, Bx , ByR , BzR, 0, 0, 0, 0, 0 ])

#model_rmhd = sr_rmhd.sr_rmhd_gamma_law(initial_data = sr_rmhd.initial_riemann(qL, qR),
#                                 gamma=gamma)
#fast_source_rmhd  = model_rmhd.relaxation_source()
#
#sim_rmhd = simulation(model_rmhd, interval, fvs_method(2), imex222(fast_source_rmhd), 
#                    outflow, cfl=0.25)
                    

sim_mf.evolve(0.019)
#
#fig = pyplot.figure()
#ax = fig.add_subplot(121)
#ax.plot(sim.coordinates, sim.cons[0, :] + sim.cons[5, :])
#ax.set_xlabel(r"$x$")
#ax.set_ylabel(r"$\rho$")
#ax.set_xlim(sim.grid.interval[0],sim.grid.interval[1])
#ax = fig.add_subplot(122)
#ax.plot(sim.coordinates, sim.cons[11, :])
#ax.set_xlabel(r"$x$")
#ax.set_ylabel(r"$B_y$")
#ax.set_xlim(sim.grid.interval[0],sim.grid.interval[1])
#pyplot.show()
    