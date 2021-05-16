# Import package
import numpy as np
import tools

def setup():
    class par: pass

    # Demograhpics
    par.age_min = 25 # Only relevant for figures
    par.T = 90-par.age_min
    par.Tr = 65-par.age_min # Retirement age, no retirement if TR=T
    
    # Preferences
    par.rho = 2
    par.beta = 0.96

    # Income parameters
    par.G = 1.03

    par.sigma_xi = 0.1
    par.sigma_psi = 0.1

    par.low_p = 0.005 # Called pi in slides
    par.low_val = 0 # Called mu in slides.
    
    # Saving and borrowing
    par.R = 1.04
    par.kappa = 0.0

    # Numerical integration and grids
    par.a_max = 20 # maximum point in grid for a
    par.a_phi = 1.1 # curvature parameters

    par.Nxi  = 8 # number of quadrature points for xi
    par.Npsi = 8 # number of quadrature points for psi
    par.Na = 500 # number of points in grid for a

    # 6. simulation
    par.sim_mini = 2.5 # initial m in simulation
    par.simN = 500000 # number of persons in simulation
    par.simT = 100 # number of periods in simulation

    return par

def create_grids(par):
    #1. Check parameters
    assert (par.rho >= 0), 'not rho > 0'
    assert (par.kappa >= 0), 'not lambda > 0'

    #2. Shocks
    eps,eps_w = tools.GaussHermite_lognorm(par.sigma_xi,par.Nxi)
    par.psi,par.psi_w = tools.GaussHermite_lognorm(par.sigma_psi,par.Npsi)

        #define xi
    if par.low_p > 0:
        par.xi =  np.append(par.low_val+1e-8, (eps-par.low_p*par.low_val)/(1-par.low_p), axis=None) # +1e-8 makes it possible to take the log in simulation if low_val = 0
        par.xi_w = np.append(par.low_p, (1-par.low_p)*eps_w, axis=None)
    else:
        par.xi = eps
        par.xi_w = eps_w

        #Vectorize all
    par.xi_vec = np.tile(par.xi,par.psi.size)       # Repeat entire array x times
    par.psi_vec = np.repeat(par.psi,par.xi.size)    # Repeat each element of the array x times
    par.xi_w_vec = np.tile(par.xi_w,par.psi.size)
    par.psi_w_vec = np.repeat(par.psi_w,par.xi.size)

    par.w = par.xi_w_vec * par.psi_w_vec
    assert (1-sum(par.w) < 1e-8), 'the weights does not sum to 1'
    
    par.Nshocks = par.w.size    # count number of shock nodes
    
    #3. Minimum a
    if par.kappa == 0:
        par.a_min = np.zeros([par.T,1])
    else:

        #Using formula from slides
        psi_min = min(par.psi)
        xi_min = min(par.xi)
        par.a_min = np.nan + np.zeros([par.T,1])
        for t in range(par.T-1,-1,-1):
            if t >= par.Tr:
                Omega = 0  # No debt in final period
            elif t == par.T-1:
                Omega = par.R**(-1)*par.G*psi_min*xi_min
            else: 
                Omega = par.R**(-1)*(min(Omega,par.kappa)+xi_min)*par.G*psi_min
            
            par.a_min[t]=-min(Omega,par.kappa)*par.G*psi_min
    
    
    #4. End of period assets
    par.grid_a = np.nan + np.zeros([par.T,par.Na])
    for t in range(par.T):
        par.grid_a[t,:] = tools.nonlinspace(par.a_min[t]+1e-8,par.a_max,par.Na,par.a_phi)


    #5.  Conditions
    par.FHW = par.G/par.R # Finite human wealth <1
    par.AI = (par.R*par.beta)**(1/par.rho) # absolute impatience <1
    par.GI = par.AI*sum(par.w*par.psi_vec**(-1))/par.G # growth impatience <1
    par.RI = par.AI/par.R # Return impatience <1     
    par.WRI = par.low_p**(1/par.rho)*par.AI/par.R # weak return impatience <1 
    par.FVA = par.beta*sum(par.w*(par.G*par.psi_vec)**(1-par.rho)) # finite value of autarky <1

    # 6. Set seed
    np.random.seed(2020)

    return par

def solve(par):

    # Initialize
    class sol: pass
    shape=(par.T,par.Na+1)
    sol.c1 = np.nan+np.zeros(shape)
    sol.c2 = np.nan+np.zeros(shape)
    sol.m = np.nan+np.zeros(shape)
    
    # Last period, (= consume all) 
    sol.m[par.T-1,:]= np.linspace(0,par.a_max,par.Na+1)
    sol.c1[par.T-1,:]= sol.m[par.T-1,:].copy() / (1+(0.6/0.4)**(par.rho))
    sol.c2[par.T-1,:]= sol.m[par.T-1,:].copy() - sol.c1[par.T-1,:].copy()

    # Before last period
    for t in range(par.T-2,-1,-1):
        # Solve model with EGM
        sol = EGM(sol,t,par)

        # add zero consumption
        sol.m[t,0] = par.a_min[t,0]
        sol.c1[t,0] = 0
        sol.c2[t,0] = 0
    
    return sol

def EGM (sol,t,par):
    for i_a,a in enumerate(par.grid_a[t,:]):

        if t+1<= par.Tr: # No pension in the next period
            fac = par.G*par.psi_vec
            w = par.w
            xi = par.xi_vec
            inv_fac = 1/fac

            # Future m and c
            m_plus = inv_fac*par.R*a+xi
            c1_plus = tools.interp_linear_1d(sol.m[t+1,:],sol.c1[t+1,:], m_plus) 
            c2_plus = tools.interp_linear_1d(sol.m[t+1,:],sol.c2[t+1,:], m_plus)
        else:
            fac = par.G
            w = 1
            xi = 1
            inv_fac = 1/fac

            # Future m and c
            m_plus = inv_fac*par.R*a+xi
            c1_plus = tools.interp_linear_1d_scalar(sol.m[t+1,:],sol.c1[t+1,:], m_plus)
            c2_plus = tools.interp_linear_1d_scalar(sol.m[t+1,:],sol.c2[t+1,:], m_plus)

        # Future marginal utility
        marg_u_plus1 = marg_util_c1(fac*c1_plus,par)
        marg_u_plus2 = marg_util_c2(fac*c2_plus,par)
        avg_marg_u_plus1 = np.sum(w*marg_u_plus1)
        avg_marg_u_plus2 = np.sum(w*marg_u_plus2)

        # Current C and m
        sol.c1[t,i_a+1]=inv_marg_util(par.beta*par.R*avg_marg_u_plus1,par)
        sol.c2[t,i_a+1]=inv_marg_util(par.beta*par.R*avg_marg_u_plus2,par)
        sol.m[t,i_a+1]=a+sol.c1[t,i_a+1]+sol.c2[t,i_a+1]

    return sol

def marg_util(c,par):
    return c**(-par.rho)


def marg_util_c1(c1,par):
    return 0.4*c1**(-par.rho)

def marg_util_c2(c2,par):
    return (1-0.4)*c2**(-par.rho)

def rel_weights(N_child,par):
    return (1+np.exp(-(par.theta_0+par.theta1*N_child)))**(-1)

def inv_marg_util(u,par):
    return u**(-1/par.rho)


def simulate (par,sol):

    # Initialize
    class sim: pass
    shape = (par.simT, par.simN)
    sim.m = np.nan +np.zeros(shape)
    sim.c1 = np.nan +np.zeros(shape)
    sim.c2 = np.nan +np.zeros(shape)
    sim.a = np.nan +np.zeros(shape)
    sim.p = np.nan +np.zeros(shape)
    sim.y = np.nan +np.zeros(shape)

    # Shocks
    shocki = np.random.choice(par.Nshocks,(par.T,par.simN),replace=True,p=par.w) #draw values between 0 and Nshocks-1, with probability w
    sim.psi = par.psi_vec[shocki]
    sim.xi = par.xi_vec[shocki]

        #check it has a mean of 1
    assert (abs(1-np.mean(sim.xi)) < 1e-4), 'The mean is not 1 in the simulation of xi'
    assert (abs(1-np.mean(sim.psi)) < 1e-4), 'The mean is not 1 in the simulation of psi'

    # Initial values
    sim.m[0,:] = par.sim_mini
    sim.p[0,:] = 0.0

    # Simulation 
    for t in range(par.simT):
        sim.c1[t,:] = tools.interp_linear_1d(sol.m[t,:],sol.c1[t,:], sim.m[t,:])
        sim.c2[t,:] = tools.interp_linear_1d(sol.m[t,:],sol.c2[t,:], sim.m[t,:])
        sim.a[t,:] = sim.m[t,:] - sim.c1[t,:] - sim.c2[t,:]

        if t< par.simT-1:
            if t+1 > par.Tr: #after pension
                sim.m[t+1,:] = par.R*sim.a[t,:]/(par.G)+1
                sim.p[t+1,:] = np.log(par.G)+sim.p[t,:]
                sim.y[t+1,:] = sim.p[t+1,:]
            else:       #before pension
                sim.m[t+1,:] = par.R*sim.a[t,:]/(par.G*sim.psi[t+1,:])+sim.xi[t+1,:]
                sim.p[t+1,:] = np.log(par.G)+sim.p[t,:]+np.log(sim.psi[t+1,:])
                sim.y[t+1,:] = sim.p[t+1,:]+np.log(sim.xi[t+1,:])
    
    #Renormalize 
    sim.P = sim.p
    sim.Y = sim.y
    sim.M = sim.m*sim.P
    sim.C1 = sim.c1*sim.P
    sim.C2 = sim.c2*sim.P
    sim.A = sim.a*sim.P
    return sim