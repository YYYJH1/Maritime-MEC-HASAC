import numpy as np
from typing import Tuple, Optional


class CommunicationModel:
    
    def __init__(self, env_params: dict = None):
        params = {
            'zeta_L': 2.3,
            'zeta_NL': 34.0,
            'alpha_env': 5.0188,
            'beta_env': 0.3511,
            'phi_c': 2e9,
            'C0': 3e8,
            'h_m': 0.0,
            'h_u': 30.0,
            'h_v': 0.0,
            'B0': 1e6,
            'B': 20e6,
            'L': 2,
            'P_m': 0.5,
            'P_uv': 5.0,
            'N_G': 10**(-14.4),
            'G0': 10**(-5),
        }
        
        if env_params:
            for key in params:
                if key in env_params:
                    params[key] = float(env_params[key])
        
        for key, value in params.items():
            setattr(self, key, float(value))
    
    def get_elevation_angle(self, l_m, l_u) -> float:
        l_m = np.array(l_m, dtype=np.float64)
        l_u = np.array(l_u, dtype=np.float64)
        if len(l_m) == 2:
            l_m = np.array([l_m[0], l_m[1], self.h_m])
        if len(l_u) == 2:
            l_u = np.array([l_u[0], l_u[1], self.h_u])
        
        horizontal_dist = np.linalg.norm(l_m[:2] - l_u[:2])
        height_diff = abs(l_m[2] - l_u[2])
        
        if horizontal_dist < 1e-6:
            return np.pi / 2
        
        gamma = np.arctan(height_diff / horizontal_dist)
        return gamma
    
    def compute_m2u_path_loss(self, l_m, l_u) -> float:
        l_m = np.array(l_m, dtype=np.float64)
        l_u = np.array(l_u, dtype=np.float64)
        
        if len(l_m) == 2:
            l_m_3d = np.array([l_m[0], l_m[1], self.h_m])
        else:
            l_m_3d = l_m
            
        if len(l_u) == 2:
            l_u_3d = np.array([l_u[0], l_u[1], self.h_u])
        else:
            l_u_3d = l_u
        
        distance = np.linalg.norm(l_m_3d - l_u_3d)
        if distance < 1.0:
            distance = 1.0
        
        gamma_rad = self.get_elevation_angle(l_m, l_u)
        gamma_deg = np.degrees(gamma_rad)
        
        P_LoS = 1 / (1 + self.alpha_env * np.exp(
            -self.beta_env * (gamma_deg - self.alpha_env)
        ))
        
        fspl = 20 * np.log10(4 * np.pi * distance * self.phi_c / self.C0)
        
        path_loss_db = P_LoS * self.zeta_L + (1 - P_LoS) * self.zeta_NL + fspl
        
        return path_loss_db
    
    def compute_m2u_channel_gain(self, l_m: np.ndarray, l_u: np.ndarray) -> float:
        path_loss_db = self.compute_m2u_path_loss(l_m, l_u)
        return 10 ** (-path_loss_db / 10)
    
    def compute_m2u_rate(self, l_m: np.ndarray, l_u: np.ndarray, 
                         o_ij: int = 1) -> float:
        if o_ij == 0:
            return 0.0
        
        channel_gain = self.compute_m2u_channel_gain(l_m, l_u)
        snr = self.P_m * channel_gain / self.N_G
        rate = self.B0 * np.log2(1 + snr)
        
        return rate
    
    def compute_u2v_channel_gain(self, l_u, l_v) -> float:
        l_u = np.array(l_u, dtype=np.float64)
        l_v = np.array(l_v, dtype=np.float64)
        
        if len(l_u) == 2:
            l_u_3d = np.array([l_u[0], l_u[1], self.h_u])
        else:
            l_u_3d = l_u
            
        if len(l_v) == 2:
            l_v_3d = np.array([l_v[0], l_v[1], self.h_v])
        else:
            l_v_3d = l_v
        
        distance = np.linalg.norm(l_u_3d - l_v_3d)
        if distance < 1.0:
            distance = 1.0
        
        G_jk = self.G0 * (distance ** (-2))
        
        return G_jk
    
    def compute_u2v_rate(self, l_u: np.ndarray, l_v: np.ndarray, 
                         s_jk: int = 1, num_sharing_uavs: int = 1) -> float:
        if s_jk == 0:
            return 0.0
        
        G_jk = self.compute_u2v_channel_gain(l_u, l_v)
        snr = self.P_uv * G_jk / self.N_G
        rate = (self.L * self.B / num_sharing_uavs) * np.log2(1 + snr)
        
        return rate
    
    def compute_transmission_delay(self, data_size: float, rate: float) -> float:
        if rate <= 0:
            return float('inf')
        return data_size / rate
    
    def compute_uav_processing_delay(self, task_size: float, cycles_per_bit: float,
                                      f_ij: float) -> float:
        if f_ij <= 0:
            return float('inf')
        return task_size * cycles_per_bit / f_ij
    
    def compute_total_uav_delay(self, l_m: np.ndarray, l_u: np.ndarray,
                                 task_size: float, cycles_per_bit: float,
                                 f_ij: float) -> float:
        rate_m2u = self.compute_m2u_rate(l_m, l_u, 1)
        trans_delay = self.compute_transmission_delay(task_size, rate_m2u)
        proc_delay = self.compute_uav_processing_delay(task_size, cycles_per_bit, f_ij)
        
        return trans_delay + proc_delay
    
    def compute_total_vessel_delay(self, l_m: np.ndarray, l_u: np.ndarray,
                                    l_v: np.ndarray, task_size: float,
                                    cycles_per_bit: float, f_jk: float) -> float:
        rate_m2u = self.compute_m2u_rate(l_m, l_u, 1)
        delay_m2u = self.compute_transmission_delay(task_size, rate_m2u)
        
        rate_u2v = self.compute_u2v_rate(l_u, l_v, 1)
        delay_u2v = self.compute_transmission_delay(task_size, rate_u2v)
        
        if f_jk <= 0:
            delay_proc = float('inf')
        else:
            delay_proc = task_size * cycles_per_bit / f_jk
        
        return delay_m2u + delay_u2v + delay_proc


class QueueManager:
    
    def __init__(self, n_miots: int, n_uavs: int, n_vessels: int, 
                 tau: float = 1.0):
        self.n_miots = n_miots
        self.n_uavs = n_uavs
        self.n_vessels = n_vessels
        self.tau = tau
        
        self.Q_m = np.zeros(n_miots)
        self.Q_u = np.zeros(n_uavs)
        self.Q_v = np.zeros(n_vessels)
        
        self.prev_lyapunov = 0.0
    
    def reset(self):
        self.Q_m = np.zeros(self.n_miots)
        self.Q_u = np.zeros(self.n_uavs)
        self.Q_v = np.zeros(self.n_vessels)
        self.prev_lyapunov = 0.0
    
    def update_miot_queue(self, i: int, R_m2u: float, A_i: float) -> float:
        departure = self.tau * R_m2u
        self.Q_m[i] = max(self.Q_m[i] - departure + A_i, 0)
        return self.Q_m[i]
    
    def update_uav_queue(self, j: int, R_m2u_in: float, R_u2v_out: float,
                         f_u_process: float) -> float:
        arrival = self.tau * R_m2u_in
        departure = self.tau * (R_u2v_out + f_u_process)
        self.Q_u[j] = max(self.Q_u[j] - departure + arrival, 0)
        return self.Q_u[j]
    
    def update_vessel_queue(self, k: int, R_u2v_in: float, 
                            f_v_process: float) -> float:
        arrival = self.tau * R_u2v_in
        departure = self.tau * f_v_process
        self.Q_v[k] = max(self.Q_v[k] - departure + arrival, 0)
        return self.Q_v[k]
    
    def compute_lyapunov_function(self) -> float:
        L = 0.5 * (np.sum(self.Q_m ** 2) + 
                   np.sum(self.Q_u ** 2) + 
                   np.sum(self.Q_v ** 2))
        return L
    
    def compute_lyapunov_drift(self) -> float:
        current_L = self.compute_lyapunov_function()
        drift = current_L - self.prev_lyapunov
        self.prev_lyapunov = current_L
        return drift
    
    def compute_drift_plus_penalty(self, V: float, Phi_t: float) -> float:
        drift = self.compute_lyapunov_drift()
        return drift + V * Phi_t
    
    def get_queue_state(self) -> dict:
        return {
            'Q_m': self.Q_m.copy(),
            'Q_u': self.Q_u.copy(),
            'Q_v': self.Q_v.copy()
        }
    
    def is_stable(self, threshold: float = 1e6) -> bool:
        return (np.all(self.Q_m < threshold) and 
                np.all(self.Q_u < threshold) and 
                np.all(self.Q_v < threshold))
