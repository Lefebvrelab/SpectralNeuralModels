from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from numpy import pi,abs,exp,log,log10
from scipy import optimize

class Rowe2015Model():
    
    def __init__(self):     
        
        # Constants
        self.gamma_e = 116 # s^-1
        self.r_e = 86 # mm
        self.Q_max = 340 # s^-1
        self.theta = 12.9 # mV
        self.sigma = 3.8 #mV
        self.phi_n = 10**-5 # s^-1
        self.k0 = 10
        
        self.G_rs = 0.1  # from Abeysuria 2015
        self.G_re = 0.2  # from Abeysuria 2015
        self.G_sn = 1 # Random <3 John
        
        self.l_x = self.l_y = 0.5
        
        self.fmax = 50
        self.freq_min = 5.
        self.freq_max = 100. 
        
        # Variable parameters
        self.G_ee = 5.4
        self.G_ei = -7.
        self.G_ese = 5.6
        self.G_esre = -2.8
        self.G_srs = -0.6
        
        self.alpha = 75 #s^-1
        self.beta = 75*3.8 #s^-1
        self.t0 = 84 # ms
        self.A_EMG = 0.5E-12 #s^-1
        self.f_EMG = 40 # Hz
        
        # Variable bounds
        self.bound_G_ee = [0., 20.]
        self.bound_G_ei = [-40., 0.]
        self.bound_G_ese = [0., 40.]
        self.bound_G_esre = [-40., 0.]
        self.bound_G_srs = [-14., -0.1]
        
        self.bound_alpha = [10., 100.]
        self.bound_beta = [100., 800.]
        self.bound_t0 = [75., 140.]
        self.bound_A_EMG = [0., 1E-12]
        self.bound_f_EMG = [10., 50.]
        
        
    def compute_L(self, omega):
        
        alpha, beta = self.alpha, self.beta
        L = (1 - 1j*omega/alpha)**-1 * (1 - 1j*omega/beta)**-1
        
        return L
    
    
    def compute_q2r2(self, omega):
        
        gamma_e = self.gamma_e
        G_ei, G_ee = self.G_ei, self.G_ee
        G_ese, G_esre, G_srs = self.G_ese, self.G_esre, self.G_srs
        t0 = self.t0
        
        L = self.compute_L(omega)
        
        term1 = (1 - 1j*omega / gamma_e)**2
        coeff2 = (1 - G_ei * L)**-1
        
        term2_1 = L * G_ee 
        term2_2 = (L**2 * G_ese + L**3 * G_esre) * exp(1j*omega*t0) / (1 - L**2 * G_srs)
        term2 = term2_1 + term2_2
        
        q2r2 = term1 - coeff2 * term2
        
        return q2r2
    
    
    def compute_k2r2(self, m, n):
        
        k_x = 2*pi*m / self.l_x
        k_y = 2*pi*n / self.l_y
        
        k2r2 = (k_x**2 + k_y**2)*self.r_e**2
        
        return k2r2
    
    
    def compute_P_EEG(self, omega):
        
        G_ei, G_ee = self.G_ei, self.G_ee
        G_ese, G_esre, G_srs = self.G_ese, self.G_esre, self.G_srs
        t0 = self.t0
        r_e = self.r_e
        k0 = self.k0
        
        phi_n = self.phi_n
        
        # Other Gs
        G_sr = G_srs / self.G_rs
        G_es = G_esre / (G_sr * self.G_re)
        G_sn = self.G_sn
        
        L = self.compute_L(omega)
        q2r2 = self.compute_q2r2(omega)
        
        term1 = G_es * G_sn * phi_n * L**2 * exp(1j*omega*t0/2)
        term2 = (1 - G_srs * L**2) * (1 - G_ei * L)
        
        term3 = 0
        k_x = 2 * pi / self.l_x
        k_y = 2 * pi / self.l_y
        fmax = self.fmax
        for m in np.arange(-fmax,fmax):
            for n in  np.arange(-fmax,fmax):
                
                k2r2 = self.compute_k2r2(m,n)
                k2 = k2r2 / r_e
                Fk = exp(-k2 / k0**2)
                term3 += abs(k2r2 + q2r2)**-2 * Fk * k_x * k_y
        
        P_EEG = abs(term1)**2 * abs(term2)**2 * term3 
        return P_EEG
    
    
    def compute_P(self, omega):
        '''
        Compute the power spectrum.
        '''
        
        A_EMG, f_EMG = self.A_EMG, self.f_EMG
        
        mod_omega = omega / (2 * pi * f_EMG)
        P_EMG = A_EMG * (mod_omega)**2 / (1 + mod_omega**2)**2
        
        P_EEG = self.compute_P_EEG(omega)
        return P_EEG + P_EMG
    
    
    def update_and_compute_P(self, values, param_list, omega):
        
        N = min(len(values), len(param_list))
        for k in range(N):
            setattr(self, param_list[k], values[k])
        
        return self.compute_P(omega)     
    
    
class RoweOptimization():
    '''
    Optimizing the Rowe Model onto a training set. The key parameters to adjust
    are as follows:
    - G_ee
    - G_ei
    - G_ese
    - G_esre
    - G_srs
    - alpha
    - beta
    - t0
    - A_EMG
    - f_EMG
    '''
    
    def __init__(self, train=[]):
        self.train = train
        
        # Get frequencies
        self.freqs = np.array([train[k][0] for k in range(len(train))])
        self.output = np.array([train[k][1] for k in range(len(train))])
        self.mod = Rowe2015Model()
    
    
    def optimize(self, param_list, tol=None):
        '''
        Fits the model using the listed parameters.
        '''
        
        # Define the function w.r.t. the parameters. The vector P has the same
        # length as params, with 1-1 coordinate correspondance.
        EEG_fun = lambda P: self.mod.update_and_compute_P(P, param_list, self.freqs)  
        chi_fun = lambda P: sum(((EEG_fun(P) - self.output) / self.output)**2)
        
        # Get initial parameter values
        P0 = []
        for j in range(len(param_list)):
            P0.append(getattr(self.mod, param_list[j]))
        
        P0 = np.array(P0)
    
        # Obtain the bounds for the optimization procedure w.r.t. the selected
        # parameters.
        bounds_list = []
        for k in range(len(param_list)):
            
            bound_attr_str = 'bound_' + param_list[k]
            # Check if model has the bound attribute.
            if not hasattr(self.mod, bound_attr_str):
                bounds_list.append((None,None))
            
            else:
                bounds_list.append(tuple(getattr(self.mod, bound_attr_str)))
        
        bounds_tuple = tuple(bounds_list)
        
        # Initiate the optimization
        result = optimize.minimize(chi_fun, P0, bounds=bounds_list, tol=tol)
        
        return result        
    

if __name__ == '__main__':
    
    task = 'graph'
    
    if task == 'optimize':
        # Get training data
        text_file = np.loadtxt('EEG_data.csv', skiprows=1, delimiter=',')
        freqs = text_file[1:,0]
        powers = text_file[1:,1]
        
        N = min(len(freqs), len(powers))
        train_data = [(freqs[k], powers[k]) for k in range(N)]
        rowe_opt = RoweOptimization(train=train_data)
        
        param_list = ['G_ee',
                      'G_ei',              
                      'G_ese',
                      'G_esre',
                      'G_srs',
                      'alpha',
                      'beta',
                      't0',
                      'A_EMG'
                      ]
                      
        result = rowe_opt.optimize(param_list, tol=5)
        
        model_powers = rowe_opt.mod.compute_P(freqs)
        plt.plot(freqs, log10(powers), 'r--',
                 freqs, log10(model_powers))
        plt.show()
    
    elif task == 'graph':
        freqs = np.linspace(0.2,100, num=50)
        mod = Rowe2015Model()
        
        EEG.G_srs = -0.01
        EEG = mod.compute_P(freqs)
        df_EEG = pd.DataFrame(np.squeeze(EEG))
        
        df_EEG.abs().plot(logx=True,logy=True)   
        
            
            
        