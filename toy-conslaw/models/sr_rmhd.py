import numpy
from scipy.optimize import brentq, newton

class sr_rmhd_gamma_law(object):
    """
    Resistive case, following Kiki's thesis/paper
    
    Note: here we evolve q, which KD says is not the stable way to work.
    """
    
    def __init__(self, initial_data, gamma = 5/3, sigma = 0):
        self.gamma = gamma
        self.sigma = sigma
        self.Nvars = 12
        self.Nprim = 12
        self.Naux = 11
        self.initial_data = initial_data
        self.prim_names = (r"$\rho$", r"$v_x$", r"$v_y$", r"$v_z$",
                           r"$\epsilon$",
                           r"$B_x$", r"$B_y$", r"$B_z$",
                           r"$E_x$", r"$E_y$", r"$E_z$", r"$q$")
        self.cons_names = (r"$D$", r"$S_x$", r"$S_y$", r"$S_z$", r"$\tau$",
                           r"$B_x$", r"$B_y$", r"$B_z$",
                           r"$E_x$", r"$E_y$", r"$E_z$", r"$q$")
        self.aux_names = (r"$p$", r"$W$", r"$h$", r"$B^2$", r"$E^2$",
                          r"$\epsilon_{xjk} E^j B^k$",
                          r"$\epsilon_{yjk} E^j B^k$",
                          r"$\epsilon_{zjk} E^j B^k$",
                          r"$J^x$",
                          r"$J^y$",
                          r"$J^z$")
        
    def prim2cons(self, prim):
        rho = prim[0, :]
        vx  = prim[1, :]
        vy  = prim[2, :]
        vz  = prim[3, :]
        eps = prim[4, :]
        Bx  = prim[5, :]
        By  = prim[6, :]
        Bz  = prim[7, :]
        Ex  = prim[8, :]
        Ey  = prim[9, :]
        Ez  = prim[10, :]
        q   = prim[11, :]
        v2 = vx**2 + vy**2 + vz**2
        W = 1 / numpy.sqrt(1 - v2)
        p = (self.gamma - 1) * rho * eps
        h = 1 + eps + p / rho
        B2 = Bx * Bx + By * By + Bz * Bz
        E2 = Ex * Ex + Ey * Ey + Ez * Ez
        EcrossB_x = Ey * Bz - Ez * By
        EcrossB_y = Ez * Bx - Ex * Bz
        EcrossB_z = Ex * By - Ey * Bx
        cons = numpy.zeros_like(prim)
        cons[0, :] = rho * W
        cons[1, :] = rho * h * W**2 * vx + EcrossB_x
        cons[2, :] = rho * h * W**2 * vx + EcrossB_y
        cons[3, :] = rho * h * W**2 * vx + EcrossB_z
        cons[4, :] = rho * h * W**2 - p - rho * W + (E2 + B2) / 2
        cons[5, :] = Bx
        cons[6, :] = By
        cons[7, :] = Bz
        cons[8, :] = Ex
        cons[9, :] = Ey
        cons[10, :] = Ez
        cons[11, :] = q
        return cons
        
    def prim2all(self, prim):
        rho = prim[0, :]
        vx  = prim[1, :]
        vy  = prim[2, :]
        vz  = prim[3, :]
        eps = prim[4, :]
        Bx  = prim[5, :]
        By  = prim[6, :]
        Bz  = prim[7, :]
        Ex  = prim[8, :]
        Ey  = prim[9, :]
        Ez  = prim[10, :]
        q   = prim[11, :]
        v2 = vx**2 + vy**2 + vz**2
        W = 1 / numpy.sqrt(1 - v2)
        p = (self.gamma - 1) * rho * eps
        h = 1 + eps + p / rho
        B2 = Bx * Bx + By * By + Bz * Bz
        E2 = Ex * Ex + Ey * Ey + Ez * Ez
        EcrossB_x = Ey * Bz - Ez * By
        EcrossB_y = Ez * Bx - Ex * Bz
        EcrossB_z = Ex * By - Ey * Bx
        vcrossB_x = vy * Bz - vz * By
        vcrossB_y = vz * Bx - vx * Bz
        vcrossB_z = vx * By - vy * Bx
        vdotE = vx * Ex + vy * Ey + vz * Ez
        cons = numpy.zeros_like(prim)
        cons[0, :] = rho * W
        cons[1, :] = rho * h * W**2 * vx + EcrossB_x
        cons[2, :] = rho * h * W**2 * vx + EcrossB_y
        cons[3, :] = rho * h * W**2 * vx + EcrossB_z
        cons[4, :] = rho * h * W**2 - p - rho * W + (E2 + B2) / 2
        cons[5, :] = Bx
        cons[6, :] = By
        cons[7, :] = Bz
        cons[8, :] = Ex
        cons[9, :] = Ey
        cons[10, :] = Ez
        cons[11, :] = q
        aux = numpy.zeros((self.Naux, prim.shape[1]))
        aux[0, :] = p
        aux[1, :] = W
        aux[2, :] = h
        aux[3, :] = B2
        aux[4, :] = E2
        aux[5, :] = EcrossB_x
        aux[6, :] = EcrossB_y
        aux[7, :] = EcrossB_z
        Jx = q * vx + W * self.sigma * (Ex + vcrossB_x - vdotE * vx)
        Jy = q * vy + W * self.sigma * (Ey + vcrossB_y - vdotE * vy)
        Jz = q * vz + W * self.sigma * (Ez + vcrossB_z - vdotE * vz)
        aux[8, :] = Jx
        aux[9, :] = Jy
        aux[10, :] = Jz
        return cons, aux
        
    def cons_fn(self, guess, D, tautilde, S2tilde):
        v2 = S2tilde / guess**2
        W = 1 / numpy.sqrt(1 - v2)
        rho = D / W
        if rho < 0 or guess < 0 or v2 >= 1:
            residual = 1e6
        else:
            residual = (1 - (self.gamma - 1) / (self.gamma * W**2)) * guess + \
                ((self.gamma - 1) / (self.gamma * W) - 1) * D - tautilde
        return residual
    
    def cons2all(self, cons, prim_old):
        Np = cons.shape[1]
        prim = numpy.zeros_like(cons)
        aux = numpy.zeros((self.Naux, Np))
        for i in range(Np):
            D   = cons[0, i]
            Sx  = cons[1, i]
            Sy  = cons[2, i]
            Sz  = cons[3, i]
            tau = cons[4, i]
            Bx  = cons[5, i]
            By  = cons[6, i]
            Bz  = cons[7, i]
            Ex  = cons[8, i]
            Ey  = cons[9, i]
            Ez  = cons[10, i]
            q  = cons[11, i]
            B2 = Bx * Bx + By * By + Bz * Bz
            E2 = Ex * Ex + Ey * Ey + Ez * Ez
            EcrossB_x = Ey * Bz - Ez * By
            EcrossB_y = Ez * Bx - Ex * Bz
            EcrossB_z = Ex * By - Ey * Bx
            Stilde_x = Sx - EcrossB_x
            Stilde_y = Sy - EcrossB_y
            Stilde_z = Sz - EcrossB_z
            S2tilde = Stilde_x * Stilde_x + Stilde_y * Stilde_y + Stilde_z * Stilde_z
            tautilde = tau - (E2 + B2) / 2
            
            v2 = numpy.sum(prim_old[1:4, i]**2)
            W = 1 / numpy.sqrt(1 - v2)
            omega_guess = prim_old[0, i] * (1 + self.gamma * prim_old[4, i]) * W**2
#            omega = brentq(self.cons_fn, 1e-10, 1e10,
#                           args = (D, tautilde, S2tilde))
            omega = newton(self.cons_fn, omega_guess,
                           args = (D, tautilde, S2tilde))
            v2 = S2tilde / omega**2
            W = 1 / numpy.sqrt(1 - v2)
            rho = D / W
            eps = (omega / W**2 - rho) / (rho * self.gamma)
            p = (self.gamma - 1) * rho * eps
            h = 1 + eps + p / rho
            vx = Stilde_x / (rho * h * W**2)
            vy = Stilde_y / (rho * h * W**2)
            vz = Stilde_z / (rho * h * W**2)
            prim[0, i] = rho
            prim[1, i] = vx
            prim[2, i] = vy
            prim[3, i] = vz
            prim[4, i] = eps
            prim[5, i] = Bx
            prim[6, i] = By
            prim[7, i] = Bz
            prim[8, i] = Ex
            prim[9, i] = Ey
            prim[10, i] = Ez
            prim[11, i] = q
            aux[0, i] = p
            aux[1, i] = W
            aux[2, i] = h
            aux[3, i] = B2
            aux[4, i] = E2
            aux[5, i] = EcrossB_x
            aux[6, i] = EcrossB_y
            aux[7, i] = EcrossB_z
            vcrossB_x = vy * Bz - vz * By
            vcrossB_y = vz * Bx - vx * Bz
            vcrossB_z = vx * By - vy * Bx
            vdotE = vx * Ex + vy * Ey + vz * Ez
            Jx = q * vx + W * self.sigma * (Ex + vcrossB_x - vdotE * vx)
            Jy = q * vy + W * self.sigma * (Ey + vcrossB_y - vdotE * vy)
            Jz = q * vz + W * self.sigma * (Ez + vcrossB_z - vdotE * vz)
            aux[8, i] = Jx
            aux[9, i] = Jy
            aux[10, i] = Jz
        return prim, aux
        
    def flux(self, cons, prim, aux):
        D   = cons[0, :]
        Sx  = cons[1, :]
        Sy  = cons[2, :]
        Sz  = cons[3, :]
        Bx  = cons[5, :]
        By  = cons[6, :]
        Bz  = cons[7, :]
        Ex  = cons[8, :]
        Ey  = cons[9, :]
        Ez  = cons[10, :]
        vx  = prim[1, :]
        p   = aux[0, :]
        B2  = aux[3, :]
        E2  = aux[4, :]
        EcrossB_x = aux[5, :]
        EcrossB_y = aux[6, :]
        EcrossB_z = aux[7, :]
        Jx  = aux[9, :]
        Stilde_x = Sx - EcrossB_x
        Stilde_y = Sy - EcrossB_y
        Stilde_z = Sz - EcrossB_z
        
        f = numpy.zeros_like(cons)
        f[0, :] = D * vx
        f[1, :] = Stilde_x * vx + p - Ex * Ex - Bx * Bx + (E2 + B2) / 2
        f[2, :] = Stilde_y * vx     - Ey * Ex - By * Bx
        f[3, :] = Stilde_z * vx     - Ez * Ex - Bz * Bx
        f[4, :] = Sx - D * vx
        f[5, :] = 0 # Bx
        f[6, :] = -Ez
        f[7, :] =  Ey
        f[8, :] = 0 #Ex
        f[9, :] =  Bz
        f[10, :] = -By
        f[11, :] = Jx
        return f
        
    def fix_cons(self, cons):
        Np = cons.shape[1]
        minvals = 1e-10 * numpy.ones((1,Np))
        cons[0, :] = numpy.maximum(cons[0, :], minvals)
        return cons
        
    def max_lambda(self, cons, prim, aux):
        """
        Laziness - speed of light
        """
        return 1
        
    def riemann_problem_flux(self, q_L, q_R):
        """
        Not implemented - cost
        """
        raise NotImplementedError('Exact Riemann solver is too expensive.')
        
    def relaxation_source(self):
        """
        Simple isotropic case
        """
        def fast_source(cons, prim, aux):
            s = numpy.zeros_like(cons)
            s[8:11, :] = -aux[8:11, :]
            return s
        return fast_source
        

def initial_riemann(ql, qr):
    return lambda x : numpy.where(x < 0.0,
                                  ql[:,numpy.newaxis]*numpy.ones((12,len(x))),
                                  qr[:,numpy.newaxis]*numpy.ones((12,len(x))))
