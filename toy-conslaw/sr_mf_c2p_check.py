# GR2 test from Liv Rev

import numpy
from models import sr_mf
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
gamma = 4.0 / 3.0
epsL = pL / rhoL / (gamma - 1)
epsR = pR / rhoR / (gamma - 1)

m_p = 1
m_e = 1
rhoL_p = m_p / (m_p + m_e) * rhoL
rhoL_e = m_e / (m_p + m_e) * rhoL
rhoR_p = m_p / (m_p + m_e) * rhoR
rhoR_e = m_e / (m_p + m_e) * rhoR

qL = numpy.array([rhoL_e, 0, 0, 0, epsL, rhoL_p, 0, 0, 0, epsL, Bx , ByL , BzL, 0, 0, 0, 0, 0 ])
qR = numpy.array([rhoR_e, 0, 0, 0, epsR, rhoR_p, 0, 0, 0, epsR, Bx , ByR , BzR, 0, 0, 0, 0, 0 ])

model_mf = sr_mf.sr_mf_gamma_law(initial_data = sr_mf.initial_riemann(qL, qR),
                                 gamma=gamma,
                                 kappa_m = 0.01, kappa_f = 0.01, kappa_q = 0.001)

#sim = simulation(model, interval, fvs_method(2), rk3, outflow, cfl=0.5)

rho_e = 1 + 0.8 * numpy.random.rand(1000)
rho_p = 1 + 0.8 * numpy.random.rand(1000)
eps_e = 1 + 0.8 * numpy.random.rand(1000)
eps_p = 1 + 0.8 * numpy.random.rand(1000)
Bx = 1 + 5 * numpy.random.randn(1000)
By = -1 + 5 * numpy.random.randn(1000)
Bz = 5 * numpy.random.randn(1000)
Ex = 1 + 5 * numpy.random.randn(1000)
Ey = -1 + 5 * numpy.random.randn(1000)
Ez = 5 * numpy.random.randn(1000)
vm_e = 0.3 * numpy.random.randn(1000)
theta_e = 2 * numpy.pi * numpy.random.rand(1000)
phi_e = numpy.pi * numpy.random.randn(1000)
vx_e = vm_e * numpy.cos(theta_e) * numpy.sin(phi_e)
vy_e = vm_e * numpy.sin(theta_e) * numpy.sin(phi_e)
vz_e = vm_e * numpy.cos(phi_e)
vm_p = 0.3 * numpy.random.randn(1000)
theta_p = 2 * numpy.pi * numpy.random.rand(1000)
phi_p = numpy.pi * numpy.random.randn(1000)
vx_p = vm_p * numpy.cos(theta_p) * numpy.sin(phi_p)
vy_p = vm_p * numpy.sin(theta_p) * numpy.sin(phi_p)
vz_p = vm_p * numpy.cos(phi_p)
prim = numpy.vstack( (rho_e, vx_e, vy_e, vz_e, eps_e, 
                      rho_p, vx_p, vy_p, vz_p, eps_p, 
                      Bx, By, Bz, Ex, Ey, Ez, numpy.zeros_like(Bx), numpy.zeros_like(Bx)) )
cons, aux = model_mf.prim2all(prim)
prim_old = prim + 1e-4 * numpy.random.rand(18, 1000)
prim_check, aux_check = model_mf.cons2all(cons, prim_old)
print(numpy.linalg.norm(prim_check - prim))
print(numpy.linalg.norm(aux_check - aux))