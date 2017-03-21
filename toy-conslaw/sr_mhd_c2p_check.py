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

#sim = simulation(model, interval, fvs_method(2), rk3, outflow, cfl=0.5)

rho = 1 + 0.8 * numpy.random.rand(1000)
eps = 1 + 0.8 * numpy.random.rand(1000)
p = (gamma - 1) * rho * eps
Bx = 1 + 5 * numpy.random.randn(1000)
By = -1 + 5 * numpy.random.randn(1000)
Bz = 5 * numpy.random.randn(1000)
vm = 0.9 * numpy.random.randn(1000)
theta = 2 * numpy.pi * numpy.random.rand(1000)
phi = numpy.pi * numpy.random.randn(1000)
vx = vm * numpy.cos(theta) * numpy.sin(phi)
vy = vm * numpy.sin(theta) * numpy.sin(phi)
vz = vm * numpy.cos(phi)
prim = numpy.vstack( (rho, vx, vy, vz, eps, Bx, By, Bz) )
cons, aux = model.prim2all(prim)
prim_old = prim + 1e-4 * numpy.random.rand(8, 1000)
prim_check, aux_check = model.cons2all(cons, prim_old)
print(numpy.linalg.norm(prim_check - prim))
print(numpy.linalg.norm(aux_check - aux))