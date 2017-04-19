import numpy
from scipy.optimize import brentq, newton

class sr_mf_gamma_law(object):
    """
    Multifluid case, following Amano 2016
    """
    
    def __init__(self, initial_data, gamma = 5/3, mu_e = -1.0, mu_p = 1.0, 
                 eta = 0.0, kappa = 1.0):
        self.gamma = gamma
        self.eta = eta
        self.kappa = kappa
        self.mu_e = mu_e
        self.mu_p = mu_p
        self.Nvars = 16
        self.Nprim = 16
        self.Naux = 30
        self.initial_data = initial_data
        self.prim_names = (r"$\rho_{e}$", r"$(v_x)_{e}$", r"$(v_y)_{e}$", r"$(v_z)_{e}$",
                           r"$\epsilon_{e}$",
                           r"$\rho_{p}$", r"$(v_x)_{p}$", r"$(v_y)_{p}$", r"$(v_z)_{p}$",
                           r"$\epsilon_{p}$",
                           r"$B_x$", r"$B_y$", r"$B_z$",
                           r"$E_x$", r"$E_y$", r"$E_z$")
        self.cons_names = (r"$D_{sum}$", r"$(S_x)_{sum}$", r"$(S_y)_{sum}$", r"$(S_z)_{sum}$", r"$\tau_{sum}$",
                           r"$D_{diff}$", r"$(S_x)_{diff}$", r"$(S_y)_{diff}$", r"$(S_z)_{diff}$", r"$\tau_{diff}$",
                           r"$B_x$", r"$B_y$", r"$B_z$",
                           r"$E_x$", r"$E_y$", r"$E_z$")
        self.aux_names = (r"$p_e$", r"$p_p$", r"$\bar{p}_e$", r"$\bar{p}_p$",
                          r"$W_{e}$", r"$W_{p}$", r"$h_e$", r"$h_p$", 
                          r"$\bar{H}_e$", r"$\bar{H}_p$", 
                          r"$\bar{\rho}_e$", r"$\bar{\rho}_e$", 
                          r"$B^2$", r"$E^2$",
                          r"$\epsilon_{xjk} E^j B^k$",
                          r"$\epsilon_{yjk} E^j B^k$",
                          r"$\epsilon_{zjk} E^j B^k$",
                          r"$J^x$", r"$J^y$", r"$J^z$",
                          r"$\rho_0$", r"$\omega_p^2$", 
                          r"$\epsilon_{xjk} u^j B^k$",  
                          r"$\epsilon_{yjk} u^j B^k$" 
                          r"$\epsilon_{zjk} u^j B^k$",
                          r"$u_x$", r"$u_y$", r"$u_z$", r"$\rho$", r"$W$")

    def prim2all(self, prim):
        rho_e = prim[0, :]
        v_e  = prim[1:4, :]
        eps_e = prim[4, :]
        rho_p = prim[5, :]
        v_p  = prim[6:9, :]
        eps_p = prim[9, :]
        B  = prim[10:13, :]
        E  = prim[13:16, :]
        v2_e = numpy.sum(v_e**2, axis=0)
        v2_p = numpy.sum(v_p**2, axis=0)
        W_e = 1 / numpy.sqrt(1 - v2_e)
        W_p = 1 / numpy.sqrt(1 - v2_p)
        p_e = (self.gamma - 1) * rho_e * eps_e
        p_p = (self.gamma - 1) * rho_p * eps_p
        h_e = 1 + eps_e + p_e / rho_e
        h_p = 1 + eps_p + p_p / rho_p
        rhobar_e = rho_e * self.mu_e
        rhobar_p = rho_p * self.mu_p
        pbar_e = (self.gamma - 1) * rhobar_e * eps_e
        pbar_p = (self.gamma - 1) * rhobar_p * eps_p
        Hbar_e = rhobar_e * (1.0 + self.gamma * eps_e)
        Hbar_p = rhobar_p * (1.0 + self.gamma * eps_p)
        B2 = numpy.sum(B**2, axis=0)
        E2 = numpy.sum(E**2, axis=0)
        EcrossB = numpy.cross(E, B, axis=0)
        cons = numpy.zeros_like(prim)
        cons[0, :] = rho_e * W_e + rho_p * W_p
        cons[1, :] = rho_e * h_e * W_e**2 * v_e[0, :] + \
                     rho_p * h_p * W_p**2 * v_p[0, :] + \
                     EcrossB[0, :]
        cons[2, :] = rho_e * h_e * W_e**2 * v_e[1, :] + \
                     rho_p * h_p * W_p**2 * v_p[1, :] + \
                     EcrossB[1, :]
        cons[3, :] = rho_e * h_e * W_e**2 * v_e[2, :] + \
                     rho_p * h_p * W_p**2 * v_p[2, :] + \
                     EcrossB[2, :]
        cons[4, :] = rho_e * h_e * W_e**2 + \
                     rho_p * h_p * W_p**2 - p_e - p_p + (E2 + B2) / 2
        cons[5, :] = rhobar_e * W_e + rhobar_p * W_p
        cons[6, :] = Hbar_e * W_e**2 * v_e[0, :] + \
                     Hbar_p * W_p**2 * v_p[0, :]
        cons[7, :] = Hbar_e * W_e**2 * v_e[1, :] + \
                     Hbar_p * W_p**2 * v_p[1, :]
        cons[8, :] = Hbar_e * W_e**2 * v_e[2, :] + \
                     Hbar_p * W_p**2 * v_p[2, :]
        cons[9, :] = Hbar_e * W_e**2 + Hbar_p * W_p**2 - p_e - p_p
        cons[10, :] = B[0, :]
        cons[11, :] = B[1, :]
        cons[12, :] = B[2, :]
        cons[13, :] = E[0, :]
        cons[14, :] = E[1, :]
        cons[15, :] = E[2, :]
        aux = numpy.zeros((self.Naux, prim.shape[1]))
        aux[0, :] = p_e
        aux[1, :] = p_p
        aux[2, :] = pbar_e
        aux[3, :] = pbar_p
        aux[4, :] = W_e
        aux[5, :] = W_p
        aux[6, :] = h_e
        aux[7, :] = h_p
        aux[8, :] = Hbar_e
        aux[9, :] = Hbar_p
        aux[10, :] = rhobar_e
        aux[11, :] = rhobar_p
        aux[12, :] = B2
        aux[13, :] = E2
        aux[14, :] = EcrossB[0, :]
        aux[15, :] = EcrossB[1, :]
        aux[16, :] = EcrossB[2, :]
        rho = self.mu_e * W_e * rho_e + self.mu_p * W_p * rho_p
        J = self.mu_e * rho_e * W_e * v_e + self.mu_p * rho_p * W_p * v_p
        w2p = self.mu_e**2 * rho_e + self.mu_p**2 * rho_p
        W = (self.mu_e**2 * rho_e * W_e + self.mu_p**2 * rho_p * W_p) / w2p
        u = (self.mu_e**2 * rho_e * W_e * v_e + self.mu_p**2 * rho_p * W_p * v_p) / w2p
        rho_0 = W * rho - numpy.sum(J*u, axis=0)
        ucrossB = numpy.cross(u, B, axis=0)
        aux[17, :] = J[0, :]
        aux[18, :] = J[1, :]
        aux[19, :] = J[2, :]
        aux[20, :] = rho_0
        aux[21, :] = w2p
        aux[22:25, :] = ucrossB
        aux[25:28, :] = u
        aux[28, :] = rho
        aux[29, :] = W
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
            B = cons[10:13, i]
            E = cons[13:16, i]
            EcrossB = numpy.cross(E, B)
            B2 = numpy.sum(B**2)
            E2 = numpy.sum(E**2)
            Dsum = cons[0, i]
            Ssum = cons[1:4, i] - EcrossB
            tausum = cons[4, i] - (E2 + B2) / 2
            Ddiff = cons[5, i]
            Sdiff = cons[6:9, i]
            taudiff = cons[9, i]

            D_p = (Ddiff - self.mu_e * Dsum) / (self.mu_p - self.mu_e)
            S_p = (Sdiff - self.mu_e * Ssum) / (self.mu_p - self.mu_e)
            tau_p = (taudiff - self.mu_e * tausum) / (self.mu_p - self.mu_e)
            D_e = (Ddiff - self.mu_p * Dsum) / (self.mu_e - self.mu_p)
            S_e = (Sdiff - self.mu_p * Ssum) / (self.mu_e - self.mu_p)
            tau_e = (taudiff - self.mu_p * tausum) / (self.mu_e - self.mu_p)

            S2_e = numpy.sum(S_e**2)
            v2_e = numpy.sum(prim_old[1:4, i]**2)
            W_e = 1 / numpy.sqrt(1 - v2_e)
            omega_guess = prim_old[0, i] * (1 + self.gamma * prim_old[4, i]) * W_e**2
#            omega = brentq(self.cons_fn, 1e-10, 1e10,
#                           args = (D, tautilde, S2tilde))
            omega = newton(self.cons_fn, omega_guess,
                           args = (D_e, tau_e, S2_e))
            v2_e = S2_e / omega**2
            W_e = 1 / numpy.sqrt(1 - v2_e)
            rho_e = D_e / W_e
            eps_e = (omega / W_e**2 - rho_e) / (rho_e * self.gamma)
            p_e = (self.gamma - 1) * rho_e * eps_e
            h_e = 1 + eps_e + p_e / rho_e
            v_e = S_e / (rho_e * h_e * W_e**2)
            
            S2_p = numpy.sum(S_p**2)
            v2_p = numpy.sum(prim_old[6:9, i]**2)
            W_p = 1 / numpy.sqrt(1 - v2_p)
            omega_guess = prim_old[5, i] * (1 + self.gamma * prim_old[9, i]) * W_p**2
#            omega = brentq(self.cons_fn, 1e-10, 1e10,
#                           args = (D, tautilde, S2tilde))
            omega = newton(self.cons_fn, omega_guess,
                           args = (D_p, tau_p, S2_p))
            v2_p = S2_p / omega**2
            W_p = 1 / numpy.sqrt(1 - v2_p)
            rho_p = D_p / W_p
            eps_p = (omega / W_p**2 - rho_p) / (rho_p * self.gamma)
            p_p = (self.gamma - 1) * rho_p * eps_p
            h_p = 1 + eps_p + p_p / rho_p
            v_p = S_p / (rho_p * h_p * W_p**2)
            
            prim[0, i] = rho_e
            prim[1:4, i] = v_e
            prim[4, i] = eps_e
            prim[5, i] = rho_p
            prim[6:9, i] = v_p
            prim[9, i] = eps_p
            prim[10:13, i] = B
            prim[13:16, i] = E
            
            rhobar_e = rho_e * self.mu_e
            rhobar_p = rho_p * self.mu_p
            pbar_e = (self.gamma - 1) * rhobar_e * eps_e
            pbar_p = (self.gamma - 1) * rhobar_p * eps_p
            Hbar_e = rhobar_e * (1.0 + self.gamma * eps_e)
            Hbar_p = rhobar_p * (1.0 + self.gamma * eps_p)
            aux[0, i] = p_e
            aux[1, i] = p_p
            aux[2, i] = pbar_e
            aux[3, i] = pbar_p
            aux[4, i] = W_e
            aux[5, i] = W_p
            aux[6, i] = h_e
            aux[7, i] = h_p
            aux[8, i] = Hbar_e
            aux[9, i] = Hbar_p
            aux[10, i] = rhobar_e
            aux[11, i] = rhobar_p
            aux[12, i] = B2
            aux[13, i] = E2
            aux[14, i] = EcrossB[0]
            aux[15, i] = EcrossB[1]
            aux[16, i] = EcrossB[2]
            rho = self.mu_e * W_e * rho_e + self.mu_p * W_p * rho_p
            J = self.mu_e * rho_e * W_e * v_e + self.mu_p * rho_p * W_p * v_p
            w2p = self.mu_e**2 * rho_e + self.mu_p**2 * rho_p
            W = (self.mu_e**2 * rho_e * W_e + self.mu_p**2 * rho_p * W_p) / w2p
            u = (self.mu_e**2 * rho_e * W_e * v_e + self.mu_p**2 * rho_p * W_p * v_p) / w2p
            rho_0 = W * rho - numpy.sum(J*u, axis=0)
            ucrossB = numpy.cross(u, B, axis=0)
            aux[17, i] = J[0]
            aux[18, i] = J[1]
            aux[19, i] = J[2]
            aux[20, i] = rho_0
            aux[21, i] = w2p
            aux[22:25, i] = ucrossB
            aux[25:28, i] = u
            aux[28, i] = rho
            aux[29, i] = W
        return prim, aux
        
    def flux(self, cons, prim, aux):
        B = cons[10:13, :]
        E = cons[13:16, :]
        rho_e = prim[0, :]
        v_e   = prim[1:4, :]
        rho_p = prim[5, :]
        v_p   = prim[6:9, :]
        p_e = aux[0, :]
        p_p = aux[1, :]
        pbar_e = aux[2, :]
        pbar_p = aux[3, :]
        W_e = aux[4, :]
        W_p = aux[5, :]
        h_e = aux[6, :]
        h_p = aux[7, :]
        Hbar_e = aux[8, :]
        Hbar_p = aux[9, :]
        rhobar_e = aux[10, :]
        rhobar_p = aux[11, :]
        B2  = aux[3, :]
        E2  = aux[4, :]
        EcrossB_x = aux[14, :]
        
        f = numpy.zeros_like(cons)
        f[0, :] = rho_e * W_e * v_e[0, :] + rho_p * W_p * v_p[0, :]
        f[5, :] = rhobar_e * W_e * v_e[0, :] + rhobar_p * W_p * v_p[0, :]
        f[1, :] = rho_e * h_e * W_e**2 * v_e[0, :] * v_e[0, :] + p_e + \
                  rho_p * h_p * W_p**2 * v_p[0, :] * v_p[0, :] + p_p - \
                  (E[0, :] * E[0, :] + B[0, :] * B[0, :] - (E2 + B2) / 2)
        f[2, :] = rho_e * h_e * W_e**2 * v_e[0, :] * v_e[1, :] + \
                  rho_p * h_p * W_p**2 * v_p[0, :] * v_p[1, :] - \
                  (E[1, :] * E[0, :] + B[0, :] * B[1, :])
        f[3, :] = rho_e * h_e * W_e**2 * v_e[0, :] * v_e[2, :] + \
                  rho_p * h_p * W_p**2 * v_p[0, :] * v_p[2, :] - \
                  (E[0, :] * E[2, :] + B[0, :] * B[2, :])
        f[4, :] = rho_e * h_e * W_e**2 * v_e[0, :] + \
                  rho_p * h_p * W_p**2 * v_p[0, :] + \
                  EcrossB_x
        f[6, :] = Hbar_e * W_e**2 * v_e[0, :] * v_e[0, :] + pbar_e + \
                  Hbar_p * W_p**2 * v_p[0, :] * v_p[0, :] + pbar_p
        f[7, :] = Hbar_e * W_e**2 * v_e[0, :] * v_e[1, :] + \
                  Hbar_p * W_p**2 * v_p[0, :] * v_p[1, :] 
        f[8, :] = Hbar_e * W_e**2 * v_e[0, :] * v_e[2, :] + \
                  Hbar_p * W_p**2 * v_p[0, :] * v_p[2, :] 
        f[9, :] = Hbar_e * W_e**2 * v_e[0, :] + \
                  Hbar_p * W_p**2 * v_p[0, :] 
        f[10, :] = 0 # Bx
        f[11, :] = -E[2, :]
        f[12, :] =  E[1, :]
        f[13, :] = 0 + cons[12, :] #Ex
        f[14, :] =  B[2, :]
        f[15, :] = -B[1, :]
        return f
        
#    def fix_cons(self, cons):
#        Np = cons.shape[1]
#        minvals = 1e-10 * numpy.ones((1,Np))
#        cons[0, :] = numpy.maximum(cons[0, :], minvals)
#        return cons
        
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
        
#    def source(self):
#        def slow_source(cons, prim, aux):
#            s = numpy.zeros_like(cons)
#            s[12, :] = cons[11, :] - self.kappa * cons[12, :]
#            return s
#        return slow_source
        
    def relaxation_source(self):
        """
        Simple isotropic case
        """
        def fast_source(cons, prim, aux):
            s = numpy.zeros_like(cons)
            J = aux[17:20, :]
            E = cons[13:16, :]
            ucrossB = aux[22:25, :]
            rho_0 = aux[20, :]
            u = aux[26, :]
            rho = aux[27, :]
            W = aux[28, :]
            w2p = aux[21, :]
            s[9:12, :] = W * E + ucrossB - self.eta * (J - rho_0 * u)
            s[12, :] = numpy.sum(u*E, axis=0) - self.eta * (rho - rho_0 * W)
            s[9:13, :] *= w2p
            s[13:16, :] = -J
            return s
        return fast_source
#        
#    def relaxation_guess(self):
#        def guess_function(cons, prim, aux):
#            guess = cons.copy()
#            print('guess', guess.shape)
#            if self.sigma > 1:
#                mhd_result = guess.copy()
#                v = prim[1:4,:]
#                B = prim[5:8,:]
#                E = - numpy.cross(v, B)
#                mhd_result[8:11,:] = E
#                guess = guess / self.sigma + mhd_result * (1 - 1 / self.sigma)
#            return guess
#        return guess_function
    
def initial_riemann(ql, qr):
    return lambda x : numpy.where(x < 0.0,
                                  ql[:,numpy.newaxis]*numpy.ones((16,len(x))),
                                  qr[:,numpy.newaxis]*numpy.ones((16,len(x))))
