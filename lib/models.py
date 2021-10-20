import numpy as np

# Implementation of exact leaky integrate-and-fire neuron
class LIF:
    def __init__(self, v_init, v_reset, w, tau_s, tau_m, dt=1.):
        self.v_init = v_init
        self.v0 = v_init
        self.v_reset = v_reset
        self.w = np.asarray(w)
        self.tau_s = tau_s
        self.tau_m = tau_m
        self.dt = dt
        
        self.t = 0
        self.history = np.asarray([])
    
    def __call__(self, x):
        if not isinstance(x, np.ndarray):
            x = np.asarray(x, dtype=np.bool)
        
        if len(self.history) == 0:
            self.history = x[:, np.newaxis]
        else:
            self.history = np.hstack((self.history, x[:, np.newaxis]))
        
        return self._step()
    
    def _step(self):
        spiked = False
        synaptic_inputs = 0.
            
        for i, w in enumerate(self.w):
            t_ks = np.where(self.history[i, :] == True)[0] * self.dt
            self._delta_t = self.t - t_ks
            synaptic_inputs += w * np.sum(np.exp(-self._delta_t / self.tau_m) - np.exp(-self._delta_t / self.tau_s))
        dynamics = self.v0 * np.exp(-self.t / self.tau_m) + synaptic_inputs
        
        self.t += self.dt
        
        if dynamics > 1:
            dynamics = 1.
            self.v0 = self.v_reset
            spiked = True
            self.history = np.asarray([])
            self.t = 0
        
        return dynamics, spiked


# Implementation of exact leaky resonate-and-fire neuron
class LRF:
    def __init__(self, z_init, z_reset, w, b, omega, dt=1):
        self.z_init = z_init
        self.z_reset = z_reset
        self.z0 = z_init
        self.w = np.asarray(w, dtype=complex)
        self.a = b+omega*1j
        self.dt = dt/1000
        
        self.t = 0
        self.history = np.asarray([])
    
    def __call__(self, x):
        if not isinstance(x, np.ndarray):
            x = np.asarray(x, dtype=np.bool)
        
        if len(self.history) == 0:
            self.history = x[:, np.newaxis]
        else:
            self.history = np.hstack((self.history, x[:, np.newaxis]))
        
        return self._step()
    
    def generate(self, xs):
        if not isinstance(xs, np.ndarray):
            xs = np.asarray(xs, dtype=np.bool)
        
        self.t = 0
        self.z0 = self.z_init
        
        dynamics = np.zeros(xs.shape[-1], dtype=np.complex)
        spikes = np.zeros(xs.shape[-1])
        
        for i in range(xs.shape[-1]):
            self.history = xs[:, i-int(round(self.t/self.dt)):i+1]
            dynamics[i], spikes[i] = self._step()
        
        return dynamics, spikes
    
    def _step(self):
        spiked = False
        synaptic_inputs = 0
        for i, w in enumerate(self.w):
            t_ks = np.where(self.history[i, :] == True)[0]*self.dt
            self._delta_t = self.t-t_ks
            synaptic_inputs += w*np.sum(np.exp(self.a*self._delta_t))
        dynamics = self.z0*np.exp(self.a*self.t) + synaptic_inputs
        
        self.t += self.dt
        
        if np.imag(dynamics) > 1:
            dynamics = np.real(dynamics)+1j
            self.z0 = self.z_reset
            spiked = True
            self.history = np.asarray([])
            self.t = 0
        
        return dynamics, spiked

    
    def separatrix(self):
        b = np.real(self.a)
        omega = np.imag(self.a)
        m = -b / omega
        
        separatrix = LRF(m+0.99999j, 0+0j, [0+0j], -b, -omega, 0.1)
        time_steps = np.zeros((1, int((2*np.pi*omega)/0.1)))
        separatrix, spikes = separatrix.generate(time_steps)
        separatrix = separatrix[0:np.where(spikes==1)[0][0]]
        return separatrix

