# Import package
import numpy as np
import pandas as pd
import tools
import scipy.optimize as optimize

def setup():
    class par: pass

    # Demograhpics
    par.age_min = 25 # Only relevant for figures
    par.T = 90-par.age_min
    par.Tr = 60-par.age_min # Retirement age, no retirement if TR=T
    
    # Preferences
    par.rho = 0.5
    par.beta = 0.96

    # Income parameters
    par.G = 1.03
    par.G_ret = 1.0
    par.num_M = 50
    par.M_max = 10
    par.grid_M = tools.nonlinspace(1.0e-6,par.M_max,par.num_M,1.1) # non-linear spaced points: like np.linspace with unequal spacing
    par.nu = 0.75

    par.N_max = 3
    par.sigma_xi = 0.1
    par.sigma_psi = 0.1

    par.low_p = 0.005 # Called pi in slides
    par.low_val = 0 # Called mu in slides.
    
    # Saving and borrowing
    par.R = 1.04

    # Numerical integration
    par.Nxi  = 8 # number of quadrature points for xi
    par.Npsi = 8 # number of quadrature points for psi


    # 6. simulation
    par.sim_M_ini = 2.5 # initial m in simulation
   #par.simN = 2500000 # number of persons in simulation, set in simulation
    par.simT = par.T # number of periods in simulation

    return par

def create_grids(par):
    #1. Check parameters
    assert (par.rho >= 0), 'not rho > 0'

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
    assert (1-sum(par.w) < 1e-8), 'the weights do not sum to 1'
    
    par.Nshocks = par.w.size    # count number of shock nodes
    
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
    shape=(par.T,par.N_max+1,par.num_M)
    sol.c1 = np.nan+np.zeros(shape)
    sol.c2 = np.nan+np.zeros(shape)
    sol.V = np.nan+np.zeros(shape)
    
    #N = 0 # initialize
    # call child probabilities
    p = child_birth(par)
    # Last period, (= consume all) 
    # loop over children in final period
    for N in range(par.N_max+1):

        sol.c1[par.T-1,N,:]= par.grid_M.copy() / (1+((1-theta(par.theta0,par.theta1,N))/theta(par.theta0,par.theta1,N))**(1/par.rho))
        sol.c2[par.T-1,N,:]= par.grid_M.copy() - sol.c1[par.T-1,N,:].copy()
        sol.V[par.T-1,N,:] = util(sol.c1[par.T-1,N,:],sol.c2[par.T-1,N,:],par,N)
    
    # before last period
    for t in range(par.T-2, -1, -1): 
    
        #Initalize
        M_next = par.grid_M

        for N in range(par.N_max+1):
            
            if N < 3:
                V1_next = sol.V[t+1,N+1,:]
                V2_next = sol.V[t+1,N,:]
                V_pens = sol.V[t+1,N,:]

                for im,m in enumerate(par.grid_M):   # enumerate automatically unpack m
            
                    # call the optimizer
                    bounds = ((0,None),(0,None))
                    cons = ({'type': 'ineq', 'fun': lambda x: m-x[0]-x[1]})
                    obj_fun = lambda x: - value_of_choice(x,m,M_next,t,V1_next,V2_next,V_pens,par,N,p)
                    x0 = np.array([1.0e-7,1.0e-7]) # define initial values
                    res = optimize.minimize(obj_fun, x0, bounds=bounds, method='SLSQP',
                                            constraints=cons) # constraints with m = c1 + c2 with SLQSP-method, no loop

                    sol.V[t,N,im] = -res.fun
                    sol.c1[t,N,im] = res.x[0]
                    sol.c2[t,N,im] = res.x[1]
            
            else:
                V1_next = sol.V[t+1,N,:]
                V2_next = sol.V[t+1,N,:]
                V_pens = sol.V[t+1,N,:]

                for im,m in enumerate(par.grid_M):   # enumerate automatically unpack m
            
                    # call the optimizer
                    bounds = ((0,None),(0,None))
                    obj_fun = lambda x: - value_of_choice(x,m,M_next,t,V1_next,V2_next,V_pens,par,N,p)
                    cons = ({'type': 'ineq', 'fun': lambda x: m-x[0]-x[1]})
                    x0 = np.array([1.0e-7,1.0e-7]) # define initial values
                    res = optimize.minimize(obj_fun, x0, bounds=bounds, constraints=cons,
                                            method='SLSQP')

                    sol.V[t,N,im] = -res.fun
                    sol.c1[t,N,im] = res.x[0]
                    sol.c2[t,N,im] = res.x[1]

        
    return sol

def value_of_choice(x,m,M_next,t,V1_next,V2_next,V_pens,par,N,p):

    #"unpack" c1
    if type(x) == np.ndarray: # vector-type: depends on the type of solver used
        c1 = x[0] 
        c2 = x[1]
    else:
        c = x
    
    # define assets
    a = m - c1 - c2
    
    EV_next = 0.0 #Initialize
    if t+1<= par.Tr: # No pension in the next period

        for i,psi in enumerate(par.psi_vec):
            fac = par.G*psi
            w = par.w[i]
            xi = par.xi_vec[i]
            inv_fac = 1/fac

            # Future m and c
            M_plus = inv_fac*par.R*a+xi
            V1_plus = tools.interp_linear_1d_scalar(M_next,V1_next,M_plus) 
            V2_plus = tools.interp_linear_1d_scalar(M_next,V2_next,M_plus)

            EV_next += p[N,t]*w*V1_plus + (1-p[N,t])*w*V2_plus

    else:
        N = 0
        fac = par.G_ret
        w = 1
        nu = par.nu
        inv_fac = 1/fac

        # Future m and c
        M_plus = inv_fac*par.R*a+nu
        V_plus = tools.interp_linear_1d_scalar(M_next,V_pens,M_plus) 
        EV_next += w*V_plus

    # Value of choice
    V_guess = util(c1,c2,par,N)+par.beta*EV_next

    return V_guess


def util(c1,c2,par,N):
    return theta(par.theta0,par.theta1,N)*((c1**(1.0-par.rho))/(1.0-par.rho)) + (1-theta(par.theta0,par.theta1,N))*((c2**(1.0-par.rho))/(1.0-par.rho))

def theta(theta0,theta1,N):
    return 1/(1+(np.exp(-(theta0+theta1*N))))

def child_birth(par):
    # 1. settings
    age_min = 25 # age at which the model starts
    age_fer = 45 # maximum age of fertility
    T =  par.Tr  # maximum periods from age_min needed to solve the model
    num_n = 3    # maximum number of children

    # 2. Child arrival probabilities
    # i. allocate memory
    shape = (num_n+1,T+1)
    p = np.nan + np.zeros(shape)

    # ii. load calibrations
    birth_df = pd.read_excel('cali_birth_prob.xls') 
    birth_df = birth_df.groupby(['nkids','age']).mean().reset_index()

    # iii. Pick out relevant based on age and number of children
    age_grid = np.array([age for age in range(age_min,age_min+T+1)])
    for n in range(num_n+1):
        for iage,age in enumerate(age_grid):
            if (age>age_fer) or (n==num_n):
                p[n,iage] = 0.0
            else:
                p[n,iage] = birth_df.loc[ (birth_df['age']==age) & (birth_df['nkids']==n) ,['birth']].to_numpy()    
   
    return p

##########################################################################
def simulate (par,sol,n=4):

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
    sim.m[0,:] = par.sim_M_ini
    sim.p[0,:] = 0.0

    if n<4:
        c1 = sol.c1[:,n,:]
        c2 = sol.c2[:,n,:]
    else:
        c1 = np.mean(sol.c1,1)
        c2 = np.mean(sol.c2,1)

    # Simulation 
    for t in range(par.simT):
        sim.c1[t,:] = tools.interp_linear_1d(par.grid_M,c1[t,:],sim.m[t,:])
        sim.c2[t,:] = tools.interp_linear_1d(par.grid_M,c2[t,:],sim.m[t,:])
        sim.a[t,:] = sim.m[t,:] - sim.c1[t,:] - sim.c2[t,:]

        if t < par.simT-1:
            if t+1 > par.Tr: #after pension
                sim.m[t+1,:] = par.R*sim.a[t,:]/(par.G_ret)+par.nu
                sim.p[t+1,:] = np.log(par.G_ret)+sim.p[t,:]
                sim.y[t+1,:] = sim.p[t+1,:]
            else:       #before pension
                sim.m[t+1,:] = par.R*sim.a[t,:]/(par.G*sim.psi[t+1,:])+sim.xi[t+1,:]
                sim.p[t+1,:] = np.log(par.G)+sim.p[t,:]+np.log(sim.psi[t+1,:])
                sim.y[t+1,:] = sim.p[t+1,:]+np.log(sim.xi[t+1,:])
    
    #Renormalize 
    sim.P = np.exp(sim.p)
    sim.Y = np.exp(sim.y)
    sim.M = sim.m*sim.P
    sim.C1 = sim.c1*sim.P
    sim.C2 = sim.c2*sim.P
    sim.A = sim.a*sim.P
    return sim
    ##########################################################################