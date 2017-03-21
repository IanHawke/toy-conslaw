import numpy
from scipy.optimize import fsolve

def euler(simulation, cons, prim, aux):
    dt = simulation.dt
    rhs = simulation.rhs
    return cons + dt * rhs(cons, prim, aux, simulation)

def rk2(simulation, cons, prim, aux):
    dt = simulation.dt
    rhs = simulation.rhs
    cons1 = cons + dt * rhs(cons, prim, aux, simulation)
    cons1 = simulation.bcs(cons1, simulation.grid.Npoints, simulation.grid.Ngz)
    prim1, aux1 = simulation.model.cons2all(cons1, prim)
    return 0.5 * (cons + cons1 + dt * rhs(cons1, prim1, aux1, simulation))

def rk3(simulation, cons, prim, aux):
    dt = simulation.dt
    rhs = simulation.rhs
    cons1 = cons + dt * rhs(cons, prim, aux, simulation)
    cons1 = simulation.bcs(cons1, simulation.grid.Npoints, simulation.grid.Ngz)
    if simulation.model.fix_cons:
        cons1 = simulation.model.fix_cons(cons1)
    prim1, aux1 = simulation.model.cons2all(cons1, prim)
    cons2 = (3 * cons + cons1 + dt * rhs(cons1, prim1, aux1, simulation)) / 4
    cons2 = simulation.bcs(cons2, simulation.grid.Npoints, simulation.grid.Ngz)
    if simulation.model.fix_cons:
        cons2 = simulation.model.fix_cons(cons2)
    prim2, aux2 = simulation.model.cons2all(cons2, prim1)
    return (cons + 2 * cons2 + 2 * dt * rhs(cons2, prim2, aux2, simulation)) / 3

def rk_euler_split(rk_method, source):
    def timestepper(simulation, cons, prim, aux):
        consstar = rk_method(simulation, cons, prim, aux)
        primstar, auxstar = simulation.model.cons2all(consstar, prim)
        return consstar + simulation.dt * source(consstar, primstar, auxstar)
    return timestepper

def rk_backward_euler_split(rk_method, source):
    def timestepper(simulation, cons, prim, aux):
        consstar = rk_method(simulation, cons, prim, aux)
        primstar, auxstar = simulation.model.cons2all(consstar, prim)
        def residual(consguess):
            primguess, auxguess = simulation.model.cons2all(consguess.reshape(cons.shape), prim)
            return consguess - consstar.ravel() - simulation.dt*source(consguess, primguess, auxguess).ravel()
        cons_initial_guess = consstar + 0.5*simulation.dt*source(consstar, primstar, auxstar)
        consnext = fsolve(residual, cons_initial_guess.ravel())
        return numpy.reshape(consnext, cons.shape)
    return timestepper

def imex222(source):
    gamma = 1 - 1/numpy.sqrt(2)
    def residual1(consguess, dt, cons, prim, simulation):
        consguess = consguess.reshape(cons.shape)
        primguess, auxguess = simulation.model.cons2all(consguess, prim)
        res = consguess - cons - dt * gamma * source(consguess, 
                                                     primguess, auxguess)
        return res.ravel()
    def residual2(consguess, dt, cons, prim, k1, source1, simulation):
        consguess = consguess.reshape(cons.shape)
        primguess, auxguess = simulation.model.cons2all(consguess, prim)
        return (consguess - cons - dt * (k1 + (1 - 2*gamma)*source1 + \
            gamma*source(consguess, primguess, auxguess))).ravel()
    def timestepper(simulation, cons, prim, aux):
        dt = simulation.dt
        rhs = simulation.rhs
        consguess = cons.copy()
        cons1 = fsolve(residual1, consguess.ravel(), 
                       args=(dt, cons, prim, simulation)).reshape(cons.shape)
        cons1 = simulation.bcs(cons1, simulation.grid.Npoints, simulation.grid.Ngz)
        prim1, aux1 = simulation.model.cons2all(cons1, prim)
        k1 = rhs(cons1, prim1, aux1, simulation)
        source1 = source(cons1, prim1, aux1)
        cons2 = fsolve(residual2, cons1.copy().ravel(), 
                       args=(dt, cons, prim, k1, source1, simulation)).reshape(cons.shape)
        cons2 = simulation.bcs(cons2, simulation.grid.Npoints, simulation.grid.Ngz)
        prim2, aux2 = simulation.model.cons2all(cons2, prim1)
        k2 = rhs(cons2, prim2, aux2, simulation)
        source2 = source(cons2, prim2, aux2)
        return cons + simulation.dt * (k1 + k2 + source1 + source2) / 2
    return timestepper
#
#def imex433(source):
#    def timestepper(simulation, cons):
#        alpha = 0.24169426078821
#        beta = 0.06042356519705
#        eta = 0.12915286960590
#        dt = simulation.dt
#        rhs = simulation.rhs
#        def residual1(consguess):
#            return consguess - cons.ravel() - dt * alpha * source(consguess)
#        consguess = cons.copy() + 0.5*dt*source(cons)
#        cons1 = fsolve(residual1, consguess.ravel()).reshape(cons.shape)
#        cons1 = simulation.bcs(cons1, simulation.grid.Npoints, simulation.grid.Ngz)
##        k1 = rhs(cons1, simulation)
#        source1 = source(cons1)
#        def residual2(consguess):
#            return consguess - cons.ravel() - dt * (-alpha*source1.ravel() + alpha*source(consguess))
#        cons2 = fsolve(residual2, cons1.copy().ravel()).reshape(cons.shape)
#        cons2 = simulation.bcs(cons2, simulation.grid.Npoints, simulation.grid.Ngz)
#        k2 = rhs(cons2, simulation)
#        source2 = source(cons2)
#        def residual3(consguess):
#            return consguess - cons.ravel() - dt * (k2.ravel() + (1 - alpha)*source2.ravel() + alpha*source(consguess))
#        cons3 = fsolve(residual3, cons2.copy().ravel()).reshape(cons.shape)
#        cons3 = simulation.bcs(cons3, simulation.grid.Npoints, simulation.grid.Ngz)
#        k3 = rhs(cons3, simulation)
#        source3 = source(cons3)
#        def residual4(consguess):
#            return consguess - cons.ravel() - dt * ((k2.ravel() + k3.ravel())/4 + beta*source1.ravel() + eta*source2.ravel() + (1/2-beta-eta-alpha)*source3.ravel() + alpha*source(consguess))
#        cons4 = fsolve(residual4, cons3.copy().ravel()).reshape(cons.shape)
#        cons4 = simulation.bcs(cons4, simulation.grid.Npoints, simulation.grid.Ngz)
#        k4 = rhs(cons4, simulation)
#        source4 = source(cons4)
#        return cons + simulation.dt * (k2+k3+4*k4 + source2+source3+4*source4) / 6
#    return timestepper
