# Alfven wave test in MF (see KD paper, thesis)
# Using the MF code, so see Amano 2016

from models import sr_mf
from bcs import periodic
from simulation import simulation
from methods import fvs_method
from rk import imex222, imex433, rk3, rk_backward_euler_split, rk_euler_split
from grid import grid
from matplotlib import pyplot

Ngz = 3
Npoints = 100
L = 0.5
interval = grid([-L, L], Npoints, Ngz)

m_p = 1.
m_e = 1.
gamma = 4.0/3.0

model_mf = sr_mf.sr_mf_gamma_law(initial_data = sr_mf.initial_alfven(gamma=gamma, Kappa_f=1e80),
                                 gamma=gamma,
                                 kappa_m = 0.05066059182116889, kappa_f = 1.0e80, kappa_q = 1.0)

fast_source_mf  = model_mf.relaxation_source()
#timestepper = rk_backward_euler_split(rk3, fast_source_mf)
#timestepper = rk_euler_split(rk3, fast_source_mf)
#timestepper = rk3
timestepper = imex222(fast_source_mf)
#timestepper = imex433(fast_source_mf)
sim_mf = simulation(model_mf, interval, fvs_method(2), timestepper, 
                    periodic, cfl=0.25)

sim_mf.evolve(0.0025)

import numpy
De=numpy.loadtxt('test100/dens_e.x.asc')
Se_x=numpy.loadtxt('test100/scon_e[0].x.asc')
Se_y=numpy.loadtxt('test100/scon_e[1].x.asc')
Se_z=numpy.loadtxt('test100/scon_e[2].x.asc')
taue=numpy.loadtxt('test100/tau_e.x.asc')
Dp=numpy.loadtxt('test100/dens_p.x.asc')
Sp_x=numpy.loadtxt('test100/scon_p[0].x.asc')
Sp_y=numpy.loadtxt('test100/scon_p[1].x.asc')
Sp_z=numpy.loadtxt('test100/scon_p[2].x.asc')
taup=numpy.loadtxt('test100/tau_p.x.asc')
Bx=numpy.loadtxt('test100/Bvec[0].x.asc')
By=numpy.loadtxt('test100/Bvec[1].x.asc')
Bz=numpy.loadtxt('test100/Bvec[2].x.asc')
Ex=numpy.loadtxt('test100/Evec[0].x.asc')
Ey=numpy.loadtxt('test100/Evec[1].x.asc')
Ez=numpy.loadtxt('test100/Evec[2].x.asc')

vars = [De, Se_x, Se_y, Se_z, taue, 
        Dp, Sp_x, Sp_y, Sp_z, taup, 
        Bx, By, Bz, Ex, Ey, Ez]

for i, d in enumerate(vars):
    pyplot.plot(sim_mf.coordinates, d[106:212,12]-sim_mf.cons[i,:])
    pyplot.title("cons {}".format(i))
    pyplot.show()


rhoe=numpy.loadtxt('test100/rho_e.x.asc')
ve_x=numpy.loadtxt('test100/vel_e[0].x.asc')
ve_y=numpy.loadtxt('test100/vel_e[1].x.asc')
ve_z=numpy.loadtxt('test100/vel_e[2].x.asc')
epse=numpy.loadtxt('test100/eps_e.x.asc')
rhop=numpy.loadtxt('test100/rho_p.x.asc')
vp_x=numpy.loadtxt('test100/vel_p[0].x.asc')
vp_y=numpy.loadtxt('test100/vel_p[1].x.asc')
vp_z=numpy.loadtxt('test100/vel_p[2].x.asc')
epsp=numpy.loadtxt('test100/eps_p.x.asc')

vars = [rhoe, ve_x, ve_y, ve_z, epse, 
        rhop, vp_x, vp_y, vp_z, epsp]

for i, d in enumerate(vars):
    pyplot.plot(sim_mf.coordinates, d[106:212,12]-sim_mf.prim[i,:])
    pyplot.title("prim {}".format(i))
    pyplot.show()
