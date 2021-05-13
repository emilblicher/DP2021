# import packages used
import numpy as np
import tools_Exercise_1_5 as tools
import scipy.optimize as optimize

def solve_consumption(par):
    #Unpack par
    T = par.T
    grid_M = par.grid_M
    num_M = len(grid_M)

    # Initalize solution class and allocate memory
    class sol: pass
    shape=(T,num_M)
    sol.C = np.nan+np.zeros(shape)
    sol.C1 = np.nan+np.zeros(shape)
    sol.C2 = np.nan+np.zeros(shape)
    sol.V = np.nan+np.zeros(shape)

    # Last period, cosume what is left:
    sol.C[T-1,:] = grid_M.copy()
    sol.C1[T-1,:] = 0.5*sol.C[T-1,:].copy()
    sol.C2[T-1,:] = 0.5*sol.C[T-1,:].copy()
    sol.V[T-1,:] = util(sol.C1[T-1,:],sol.C2[T-1,:],par)

    # Before last period
    
    # Loop over periods
    for t in range(T-2, -1, -1): 
       
        #Initalize
        M_next = grid_M
        V_next = sol.V[t+1,:]
       
        #loop over states
        for im,m in enumerate(grid_M):   # enumerate automaticcaly unpack m
            
            # call the optimizer
            bounds = ((0,m+1.0e-4),(0,m+1.0e-4))
            obj_fun = lambda x: - value_of_choice(x,m,M_next,V_next,par)
            x0 = np.array([0.1,0.1]) # define initial values
            res = optimize.minimize(obj_fun, x0, bounds=bounds, method='SLSQP')

            sol.V[t,im] = -res.fun
            sol.C1[t,im] = res.x[0]
            sol.C2[t,im] = res.x[1]
            
        
    return sol

def value_of_choice(x, m,M_next,V_next,par):

    #"unpack" c1
    if type(x) == np.ndarray: # vector-type: depends on the type of solver used
        c1 = x[0] 
        c2 = x[1]
    else:
        c = x
 

    
    #Expected Value next period given states and choice
    EV_next = 0.0 #Initialize
    for s,eps in enumerate(par.eps):
         
        M_plus = par.R*(m - c1 - c2) + eps
        V_plus = tools.interp_linear_1d_scalar(M_next,V_next,M_plus) 

        # weight on the shock 
        w = par.eps_w[s]

        EV_next +=w*V_plus 
  

    # Value of choice
    V_guess = util(c1,c2,par)+par.beta*EV_next

    return V_guess


def util(c1,c2,par):
    return theta(par.theta0,par.theta1,par.N)*(c1**(1.0-par.rho))/(1.0-par.rho) + (1-theta(par.theta0,par.theta1,par.N))*(c2**(1.0-par.rho))/(1.0-par.rho)

def theta(theta0,theta1,N):
    return 1/(1+(np.exp(-(theta0+theta1*N))))