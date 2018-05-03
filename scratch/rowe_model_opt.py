from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from numpy import pi,abs,exp,log
from scipy import optimize

class Rowe2004Model():

    def __init__(self,freqs=None, alpha = 88.,gamma = 71.8, t_0 = 79.2,
                 G_ee = 3.8,G_ei = -8.,G_ese = 10.8, G_esre = -5.7,
                 G_srs = -0.34, p_0 = 2.94,
                 r_e=80.):
        
        # Initialize parameters
        self.freqs = freqs
        
        # Table 1
        self.sigma_e = 3.3
        self.theta_e = 15.
        self.alpha = 40.
        self.beta_over_alpha = 4.0
        self.beta = self.beta_over_alpha * self.alpha
        
        
        self.r_e = 80.
        self.r_i = self.r_r = self.r_s = None
       
        self.va = 8.
        self.t0 = 84.
        self.k0 = 30.
        self.v_ee = self.v_es = self.v_se = 1.2
        self.v_ei = -1.8
        self.v_re = 0.4
        self.v_rs = 0.2
        self.v_sr = -0.8

        self.v_sn_phi_n = 1.0

        self.gamma_e = 130.
        self.gamma_i = self.gamma_r = self.gamma_s = 10E4

        self.Q_a = 250.

        self.G_ee = 6.
        self.G_ei = -7.
        self.G_srs = -0.4
        self.G_esre = -3. 
        self.G_ese = 5.  # G_ese = G_es * G_se

        self.G_es = -0.03
        self.G_se = self.G_ese / self.G_es
        
   
        self.G_rs = 0.1  # from Abeysuria 2015
        self.G_sr = self.G_srs / self.G_rs 

        self.G_re = 0.2  # from Abeysuria 2015
        
        self.G_sn = 1. # JG random choice
        self.G_ie = 1. # JG random choice
        
        self.lx = 0.5 # from Abeysuria. Unit = m.
        self.ly = 0.5 # from Abeysuria. Unit = m.
        
        self.phi_n = np.random.rand(1,len(self.freqs))[0]
        
        self.freq_min = 5.
        self.freq_max = 100
        
        self.fmax = 50. # near eqn. 14 Rowe
        
        
        # Table 1 bounds
        self.bound_sigma_e = [3.,8.]
        self.bound_theta_e = [10.,25.]
        self.bound_alpha = [35.,150.]
        self.bound_beta_over_alpha = [1.,8.]
        self.bound_beta = self.beta_over_alpha * self.alpha
        
        
        self.bound_r_e = [60.,100.]
        self.bound_r_i = self.r_r = self.r_s = [0.,0.1] # ~0.1
       
        self.bound_va = [5.,10.]
        self.bound_t0 = [60.,100.]
        self.bound_k0 = [10.,50.]
        self.bound_v_ee = self.bound_v_es = self.bound_v_se = [0.05,10.]
        self.bound_v_ei = [-10.,-0.05]
        self.bound_v_re = [0.05,10.]
        self.bound_v_rs = [0.05,10.]
        self.bound_v_sr = [-10.,-0.05]

        self.bound_v_sn_phi_n = [0.05,10.]

        self.bound_gamma_e = [50.,200.]
        # Approx 10E4
        self.bound_gamma_i = self.bound_gamma_r = self.bound_gamma_s = [10E4 - 10E3, 10E4 + 10E3]

        self.bound_Q_a = [100.,1000.]

        self.bound_G_ee = [1.,20.]
        self.bound_G_ei = [-20.,-1.]
        self.bound_G_srs = [-2.5,-0.01]
        self.bound_G_esre = [-10.,-0.5] 
        self.bound_G_ese = [0.1,10.]  # G_ese = G_es * G_se
        
        
    def compute_L(self,omega):
        # Rowe et al. Eq 
    
        a,b = self.alpha,self.beta
    
        # omega = 2 pi f is angular frequency
        # f is freq in Hz
        L = (1. - 1j*omega / a)**-1 * (1 - 1j*omega / b ) **-1
        
        return L


    def compute_T(self,omega):
        # Rowe et al. Eq. 9
        
        G_sn,G_sr,G_rs,t0 = self.G_sn,self.G_sr,self.G_rs,self.t0
    
        L = self.compute_L(omega)
    
        T = ( L*G_sn * exp(1j*omega *t0 / 2.)  )  / ( 1 - L*G_sr * L * G_rs)
    
        return T


    def compute_S(self,omega):
        
        G_se,G_sr,G_re,G_rs,t0 = self.G_se,self.G_sr,self.G_re,self.G_rs,self.t0
    
        # Rowe et al. Eq. 10
    
        L = self.compute_L(omega)
    
        S = ( (L * G_se + L * G_sr * L * G_re ) * exp(1j*omega*t0 / 2.) ) / ( 1. - L * G_sr * L * G_rs)
    
        return S


    def compute_q2r2(self,omega):
        
        G_ee,G_es,G_ei,gamma_e = self.G_ee,self.G_es,self.G_ei,self.gamma_e
        
        # Eq 12
        L = self.compute_L(omega)
        S = self.compute_S(omega)
    
        q2r2 = ( 1 - ((1j * omega) / gamma_e))**2    - ( (G_ee * L + G_es * L * S ) / (1 - G_ei * L) )
    
        return q2r2


    def compute_P0(self):
        
        # Eq 13
        
        phi_n,G_es,G_sn,r_e = self.phi_n,self.G_es,self.G_sn,self.r_e
        
        P0 = (( pi * sum(phi_n**2) ) / r_e**2 ) * G_es * G_sn

        return P0


    def compute_k2r2(self,m,n):
    
        r_e,lx,ly = self.r_e,self.lx,self.ly
    
        term1 = (2. * pi ** m * r_e / lx)**2
        term2 = (2. * pi * n * r_e / ly)**2
    
        return term1 + term2

    
    def compute_P_EEG(self, omega):
        '''
        Computes the P_EEG of a single frequency omega.
        '''
        
        G_sn,G_ie,lx,ly,r_e = self.G_sn,self.G_ie,self.lx,self.ly,self.r_e
        k0 = self.k0
        
        P0 = self.compute_P0()
        L = self.compute_L(omega)
        T = self.compute_T(omega)
        q2r2 = self.compute_q2r2(omega)
        
        fmax = self.fmax

        term1  = P0 * abs(  ((L * T) / G_sn) / ( 1- G_ie * L ) )** 2 * (2 * pi)**2 / (lx*ly)
            
        term2 = 0
        # Can be summed over |m|, |n| < fmax / 2
        for m in np.arange(-fmax,fmax):
            for n in  np.arange(-fmax,fmax):
                
                k2r2 = self.compute_k2r2(m,n)
                k2 = k2r2/(r_e**2.)
                
                term2+= (exp((-k2**2)/(k0**2)) ) /  abs(k2 * r_e**2. + q2r2) **2                          
                                         
        return term1 * term2
    
    
    def compute_vector_P_EEG(self):
        '''
        Computes the P_EEG for every freq in self.freq. Used for optimization.
        '''
        
        G_sn,G_ie,lx,ly,r_e = self.G_sn,self.G_ie,self.lx,self.ly,self.r_e
        k0 = self.k0
        
        # Vectorized functions
        P0 = self.compute_P0()
        L_fun = np.vectorize(self.compute_L)
        T_fun = np.vectorize(self.compute_T)
        q2r2_fun = np.vectorize(self.compute_q2r2)
        
        # Arrays
        L = L_fun(self.freqs)
        T = T_fun(self.freqs)
        q2r2 = q2r2_fun(self.freqs)
        
        fmax = self.fmax
        term1 = P0 * abs(  ((L * T) / G_sn) / ( 1- G_ie * L ) )** 2 * (2 * pi)**2 / (lx*ly)
        
        term2 = np.zeros(len(self.freqs))
        
        for m in np.arange(-fmax,fmax):
            for n in  np.arange(-fmax,fmax):
                
                k2r2 = self.compute_k2r2(m,n)
                k2 = k2r2/(r_e**2.)
                
                term2+= (exp((-k2**2)/(k0**2)) ) /  abs(k2 * r_e**2. + q2r2) **2  
                
        return term1 * term2
    
        
    # For optimization
    def update_compute_P_EEG(self, values, param_list):
        '''
        Given a vector of frequencies, parameter values and their 
        corresponding parameters (of same length), updates the parameter values 
        and returns the PPG at each frequency.
        '''
        
        N = min(len(values), len(param_list))
        for k in range(N):
            setattr(self, param_list[k], values[k])
        
        return self.compute_vector_P_EEG()
    
    
class RoweOptimization():
    '''
    Optimizing the Rowe Model onto a training set.
    '''
    
    def __init__(self, train=[]):
        
        self.train = train
        
        # Get frequencies
        self.freqs = np.array([train[k][0] for k in range(len(train))])
        self.output = np.array([train[k][1] for k in range(len(train))])
        self.rowe = Rowe2004Model(freqs=self.freqs)
        
        self.variance = get_var_weights(self.freqs)
        
        
    def optimize(self, param_list, tol=None):
        '''
        Fit the model by adjusting the listed parameters (given in strings)
        '''
    
        # Define the function w.r.t. the parameters. The vector P has the same
        # length as params, with 1-1 coordinate correspondance.
        EEG_fun = lambda P: self.rowe.update_compute_P_EEG(P, param_list)
        
        # Consider taking logorithmic difference (see paper). Make tolerance
        # 50.
        # ERR_fun = lambda P: sum((EEG_fun(P) - self.output)**2) / 2
        ERR_fun = lambda P: sum((log(abs(EEG_fun(P))) - log(abs(self.output)))**2 / self.variance)
        
        
        # Get initial parameter values
        P0 = []
        for j in range(len(param_list)):
            P0.append(getattr(self.rowe, param_list[j]))
        
        P0 = np.array(P0)
    
        # Obtain the bounds for the optimization procedure w.r.t. the selected
        # parameters.
        bounds_list = []
        for k in range(len(param_list)):
            
            bound_attr_str = 'bound_' + param_list[k]
            # Check if model has the bound attribute.
            if not hasattr(self.rowe, bound_attr_str):
                bounds_list.append((None,None))
            
            else:
                bounds_list.append(tuple(getattr(self.rowe, bound_attr_str)))
        
        bounds_tuple = tuple(bounds_list)
        
        # Initiate the optimization
        result = optimize.minimize(ERR_fun, P0, bounds=bounds_list, tol=tol)
        
        return result
        

# SUPPLEMENTARY FUNCTIONS
def get_var_weights(freqs):
    '''
    Returns the variance weightings for an array of frequencies.
    '''
    
    var_weights = []
    for k in range(len(freqs)):
        
        freq = freqs[k]
        weight = 1
        
        # Assign weight
        if freq <= 30:
            weight = 4**2
            
        elif freq < 60 and freq > 30:
            weight = 10**2
            
        elif freq >= 60:
            weight = 4**2
        
        var_weights.append(weight)
    
    return np.array(var_weights)
        
        
if __name__ == '__main__':
    
    task = 'optimize'
    
    if task == 'optimize':
        train_data = [(1,2), (3,4)]
        rowe_opt = RoweOptimization(train=train_data)
        result = rowe_opt.optimize(['sigma_e', 'theta_e'])
    
    elif task == 'graph':
        freqs = np.linspace(0.001,100, num=50)
        mod = Rowe2004Model(freqs=freqs)
        
        EEG = mod.compute_vector_P_EEG()
        df_EEG = pd.DataFrame(np.squeeze(EEG))
        
        df_EEG.abs().plot(logx=True,logy=True)        

