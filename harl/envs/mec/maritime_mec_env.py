import numpy as np
from gym.spaces import Box, Discrete, MultiDiscrete
import json
from typing import Dict, List, Tuple, Optional
import os

from harl.envs.mec.communication_model import CommunicationModel, QueueManager


class Task:
    
    def __init__(self, device_id: int, data_size: float, 
                 required_computation: float, time_slot: int):
        self.device_id = device_id
        self.data_size = float(data_size)
        self.required_computation = float(required_computation)
        self.remaining_computation = float(required_computation)
        
        self.time_slot = time_slot
        self.arrival_time = time_slot
        self.completion_time = -1
        self.is_completed = False
        
        self.processing_location = None
        self.assigned_uav = None
        self.assigned_vessel = None
        
        self.transmission_delay = 0.0
        self.computation_delay = 0.0
        self.total_delay = 0.0
        
        self.status = "created"


class MaritimeMECEnv:
    
    def __init__(self, env_args: dict):
        self.env_args = env_args
        
        if "task_file" in env_args and os.path.exists(env_args["task_file"]):
            with open(env_args["task_file"], 'r') as f:
                self.task_data = json.load(f)
        else:
            self.task_data = self._generate_default_tasks(env_args)
        
        self.n_miots = int(env_args.get("num_iot_devices", 10))
        self.n_uavs = int(env_args.get("num_uavs", 6))
        self.n_vessels = int(env_args.get("num_vessels", 2))
        self.n_agents = self.n_uavs + self.n_vessels
        
        self.map_size = float(env_args.get("map_size", 1000))
        self.num_time_slots = int(env_args.get("num_time_slots", 30))
        self.tau = float(env_args.get("tau", 1.0))
        
        self.f_uav_max = float(env_args.get("uav_computing_power", 1e9))
        self.f_vessel_max = float(env_args.get("vessel_computing_power", 1e10))
        self.c_i = float(env_args.get("cycles_per_bit", 270))
        
        self.lambda_arrival = float(env_args.get("lambda_arrival", 15))
        
        self.V = float(env_args.get("lyapunov_v", 10.0))
        
        self.comm_model = CommunicationModel(env_args)
        
        self.queue_manager = QueueManager(self.n_miots, self.n_uavs, 
                                          self.n_vessels, self.tau)
        
        self._setup_spaces()
        
        self.current_time_slot = 0
        self.miots = []
        self.uavs = []
        self.vessels = []
        self.tasks = []
        self.completed_tasks = []
        
        self.statistics = self._init_statistics()
        
        self.reset()
    
    def _generate_default_tasks(self, env_args: dict) -> dict:
        n_devices = env_args.get("num_iot_devices", 10)
        n_slots = env_args.get("num_time_slots", 30)
        lambda_param = env_args.get("lambda_arrival", 15)
        
        tasks = []
        for t in range(n_slots):
            slot_tasks = []
            if t % 3 == 0:
                for i in range(n_devices):
                    data_size = max(1, np.random.poisson(lambda_param)) * 1e6
                    slot_tasks.append({
                        "device_id": i,
                        "time_slot": t,
                        "data_size": data_size,
                        "computation": data_size * env_args.get("cycles_per_bit", 270)
                    })
            tasks.append(slot_tasks)
        
        return {
            "num_devices": n_devices,
            "num_time_slots": n_slots,
            "tasks": tasks
        }
    
    def _setup_spaces(self):
        obs_dim = (
            self.n_miots * 2 +
            self.n_uavs * 2 +
            self.n_vessels * 2 +
            self.n_miots +
            self.n_uavs +
            self.n_vessels +
            self.n_miots * 2
        )
        
        action_dim = max(self.n_miots * 2, self.n_uavs * 2)
        
        self.observation_space = []
        self.action_space = []
        self.share_observation_space = []
        
        share_obs_dim = obs_dim
        
        for _ in range(self.n_agents):
            self.observation_space.append(Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ))
            self.action_space.append(Box(
                low=0.0, high=1.0, shape=(action_dim,), dtype=np.float32
            ))
            self.share_observation_space.append(Box(
                low=-np.inf, high=np.inf, shape=(share_obs_dim,), dtype=np.float32
            ))
    
    def _init_statistics(self) -> dict:
        return {
            'completed_tasks': 0,
            'total_tasks': 0,
            'avg_completion_time': 0.0,
            'avg_delay': 0.0,
            'total_computation': 0.0,
            'uav_computation': 0.0,
            'vessel_computation': 0.0,
            'total_transmission': 0.0,
            'edge_computing_percentage': 0.0,
            'queue_stability': True,
            'lyapunov_drift': 0.0
        }
    
    def reset(self) -> Tuple[List[np.ndarray], np.ndarray, None]:
        self.current_time_slot = 0
        self.tasks = []
        self.completed_tasks = []
        self.statistics = self._init_statistics()
        
        self.queue_manager.reset()
        
        self.miots = []
        center = np.array([self.map_size / 2, self.map_size / 2])
        miot_radius = self.map_size / 6
        
        for i in range(self.n_miots):
            if "device_positions" in self.task_data and i < len(self.task_data["device_positions"]):
                position = np.array(self.task_data["device_positions"][i])
            else:
                angle = 2 * np.pi * i / self.n_miots
                base_pos = center + miot_radius * np.array([np.cos(angle), np.sin(angle)])
                offset = np.random.uniform(-50, 50, size=2)
                position = np.clip(base_pos + offset, 0, self.map_size)
            
            self.miots.append({
                'id': i,
                'position': position,
                'current_task': None,
                'task_queue': []
            })
        
        self.uavs = []
        uav_radius = self.map_size / 5
        for j in range(self.n_uavs):
            angle = 2 * np.pi * j / self.n_uavs + np.pi / self.n_uavs
            position = center + uav_radius * np.array([np.cos(angle), np.sin(angle)])
            
            self.uavs.append({
                'id': j,
                'position': position,
                'f_max': self.f_uav_max,
                'allocated_tasks': [],
                'task_queue': [],
                'completed_tasks': []
            })
        
        self.vessels = []
        vessel_radius = self.map_size / 10
        for k in range(self.n_vessels):
            if self.n_vessels == 1:
                position = center.copy()
            else:
                angle = 2 * np.pi * k / self.n_vessels + np.pi / 4
                position = center + vessel_radius * np.array([np.cos(angle), np.sin(angle)])
            
            self.vessels.append({
                'id': k,
                'position': position,
                'f_max': self.f_vessel_max,
                'allocated_tasks': [],
                'task_queue': [],
                'completed_tasks': []
            })
        
        self._load_tasks_for_time_slot()
        
        self.statistics['total_tasks'] = sum(
            len(slot_tasks) for slot_tasks in self.task_data.get("tasks", [])
        )
        
        obs = self._get_observations()
        state = self._get_state()
        
        return obs, state, None
    
    def _load_tasks_for_time_slot(self):
        if self.current_time_slot >= len(self.task_data.get("tasks", [])):
            return
        
        slot_tasks = self.task_data["tasks"][self.current_time_slot]
        for task_data in slot_tasks:
            device_id = task_data.get("device_id", 0)
            if device_id >= self.n_miots:
                continue
            
            data_size = task_data.get("data_size", 1e6)
            computation = task_data.get("computation", data_size * self.c_i)
            
            task = Task(
                device_id=device_id,
                data_size=data_size,
                required_computation=computation,
                time_slot=self.current_time_slot
            )
            
            if self.miots[device_id]['current_task'] is None:
                self.miots[device_id]['current_task'] = task
            else:
                self.miots[device_id]['task_queue'].append(task)
            
            self.tasks.append(task)
    
    def _get_observations(self) -> List[np.ndarray]:
        observations = []
        for _ in range(self.n_agents):
            obs = self._get_unified_observation()
            observations.append(obs)
        return observations
    
    def _get_unified_observation(self) -> np.ndarray:
        obs = []
        
        for miot in self.miots:
            obs.extend(miot['position'] / self.map_size)
        
        for uav in self.uavs:
            obs.extend(uav['position'] / self.map_size)
        
        for vessel in self.vessels:
            obs.extend(vessel['position'] / self.map_size)
        
        for i in range(self.n_miots):
            obs.append(self.queue_manager.Q_m[i] / 1e9)
        
        for j in range(self.n_uavs):
            obs.append(self.queue_manager.Q_u[j] / 1e9)
        
        for k in range(self.n_vessels):
            obs.append(self.queue_manager.Q_v[k] / 1e9)
        
        for miot in self.miots:
            if miot['current_task'] is not None:
                obs.append(miot['current_task'].data_size / 1e8)
                obs.append(miot['current_task'].required_computation / 1e12)
            else:
                obs.extend([0, 0])
        
        return np.array(obs, dtype=np.float32)
    
    def _get_state(self) -> np.ndarray:
        state = []
        
        for miot in self.miots:
            state.extend(miot['position'] / self.map_size)
        for uav in self.uavs:
            state.extend(uav['position'] / self.map_size)
        for vessel in self.vessels:
            state.extend(vessel['position'] / self.map_size)
        
        state.extend(self.queue_manager.Q_m / 1e9)
        state.extend(self.queue_manager.Q_u / 1e9)
        state.extend(self.queue_manager.Q_v / 1e9)
        
        for miot in self.miots:
            if miot['current_task'] is not None:
                state.append(miot['current_task'].data_size / 1e8)
                state.append(miot['current_task'].required_computation / 1e12)
            else:
                state.extend([0, 0])
        
        state = np.array(state, dtype=np.float32)
        
        share_obs = [state.copy() for _ in range(self.n_agents)]
        return np.array(share_obs)
    
    def step(self, actions: np.ndarray) -> Tuple:
        uav_actions = actions[:self.n_uavs]
        vessel_actions = actions[self.n_uavs:]
        
        self._process_uav_actions(uav_actions)
        
        self._process_vessel_actions(vessel_actions)
        
        self._process_computation()
        
        self._update_queues()
        
        rewards = self._compute_rewards()
        
        self.current_time_slot += 1
        
        if self.current_time_slot < self.num_time_slots:
            self._load_tasks_for_time_slot()
        
        self._update_statistics()
        
        obs = self._get_observations()
        state = self._get_state()
        
        dones = self._check_done()
        
        formatted_rewards = [np.array([r], dtype=np.float32) for r in rewards]
        
        infos = [self.statistics.copy() for _ in range(self.n_agents)]
        
        return obs, state, formatted_rewards, dones, infos, None
    
    def _process_uav_actions(self, uav_actions: np.ndarray):
        for j, uav in enumerate(self.uavs):
            action = uav_actions[j]
            
            for i in range(self.n_miots):
                collect_prob = action[i * 2]
                resource_ratio = action[i * 2 + 1]
                
                miot = self.miots[i]
                
                if collect_prob > 0.5 and miot['current_task'] is not None:
                    task = miot['current_task']
                    
                    rate = self.comm_model.compute_m2u_rate(
                        miot['position'], uav['position'], 1
                    )
                    task.transmission_delay = task.data_size / rate if rate > 0 else float('inf')
                    
                    task.assigned_uav = j
                    task.processing_location = f"uav_{j}"
                    task.status = "transmitting"
                    
                    uav['task_queue'].append(task)
                    miot['current_task'] = None
                    
                    if miot['task_queue']:
                        miot['current_task'] = miot['task_queue'].pop(0)
    
    def _process_vessel_actions(self, vessel_actions: np.ndarray):
        for k, vessel in enumerate(self.vessels):
            action = vessel_actions[k]
            
            for j in range(self.n_uavs):
                accept_prob = action[j * 2]
                resource_ratio = action[j * 2 + 1]
                
                uav = self.uavs[j]
                
                if accept_prob > 0.5 and uav['allocated_tasks']:
                    tasks_to_relay = []
                    for task in uav['allocated_tasks']:
                        if task.remaining_computation > 0.5 * task.required_computation:
                            tasks_to_relay.append(task)
                    
                    for task in tasks_to_relay[:1]:
                        rate = self.comm_model.compute_u2v_rate(
                            uav['position'], vessel['position'], 1
                        )
                        task.transmission_delay += task.data_size / rate if rate > 0 else 0
                        
                        task.assigned_vessel = k
                        task.processing_location = f"vessel_{k}"
                        
                        uav['allocated_tasks'].remove(task)
                        vessel['task_queue'].append(task)
    
    def _process_computation(self):
        for j, uav in enumerate(self.uavs):
            while uav['task_queue']:
                task = uav['task_queue'].pop(0)
                task.status = "computing"
                uav['allocated_tasks'].append(task)
            
            if uav['allocated_tasks']:
                n_tasks = len(uav['allocated_tasks'])
                f_per_task = uav['f_max'] / n_tasks
                
                for task in list(uav['allocated_tasks']):
                    progress = f_per_task * self.tau
                    task.remaining_computation -= progress
                    
                    self.statistics['uav_computation'] += min(progress, task.remaining_computation + progress)
                    
                    if task.remaining_computation <= 0:
                        task.is_completed = True
                        task.completion_time = self.current_time_slot
                        task.total_delay = task.completion_time - task.arrival_time
                        task.status = "completed"
                        
                        uav['allocated_tasks'].remove(task)
                        uav['completed_tasks'].append(task)
                        self.completed_tasks.append(task)
                        self.statistics['completed_tasks'] += 1
        
        for k, vessel in enumerate(self.vessels):
            while vessel['task_queue']:
                task = vessel['task_queue'].pop(0)
                task.status = "computing"
                vessel['allocated_tasks'].append(task)
            
            if vessel['allocated_tasks']:
                n_tasks = len(vessel['allocated_tasks'])
                f_per_task = vessel['f_max'] / n_tasks
                
                for task in list(vessel['allocated_tasks']):
                    progress = f_per_task * self.tau
                    task.remaining_computation -= progress
                    
                    self.statistics['vessel_computation'] += min(progress, task.remaining_computation + progress)
                    
                    if task.remaining_computation <= 0:
                        task.is_completed = True
                        task.completion_time = self.current_time_slot
                        task.total_delay = task.completion_time - task.arrival_time
                        task.status = "completed"
                        
                        vessel['allocated_tasks'].remove(task)
                        vessel['completed_tasks'].append(task)
                        self.completed_tasks.append(task)
                        self.statistics['completed_tasks'] += 1
    
    def _update_queues(self):
        for i, miot in enumerate(self.miots):
            R_m2u = 0
            for uav in self.uavs:
                R_m2u += self.comm_model.compute_m2u_rate(
                    miot['position'], uav['position'], 1
                )
            
            A_i = 0
            if miot['current_task'] is not None:
                A_i = miot['current_task'].data_size
            
            self.queue_manager.update_miot_queue(i, R_m2u, A_i)
        
        for j, uav in enumerate(self.uavs):
            R_m2u_in = sum(
                self.comm_model.compute_m2u_rate(miot['position'], uav['position'], 1)
                for miot in self.miots
            )
            
            R_u2v_out = sum(
                self.comm_model.compute_u2v_rate(uav['position'], vessel['position'], 1)
                for vessel in self.vessels
            )
            
            f_u_process = len(uav['allocated_tasks']) * uav['f_max'] / max(1, len(uav['allocated_tasks']))
            
            self.queue_manager.update_uav_queue(j, R_m2u_in, R_u2v_out, f_u_process)
        
        for k, vessel in enumerate(self.vessels):
            R_u2v_in = sum(
                self.comm_model.compute_u2v_rate(uav['position'], vessel['position'], 1)
                for uav in self.uavs
            )
            
            f_v_process = len(vessel['allocated_tasks']) * vessel['f_max'] / max(1, len(vessel['allocated_tasks']))
            
            self.queue_manager.update_vessel_queue(k, R_u2v_in, f_v_process)
    
    def _compute_rewards(self) -> List[float]:
        Phi_t = 0
        for task in self.completed_tasks:
            if task.completion_time == self.current_time_slot:
                Phi_t += task.total_delay
        
        drift = self.queue_manager.compute_lyapunov_drift()
        self.statistics['lyapunov_drift'] = drift
        
        cost = drift + self.V * Phi_t
        
        if self.queue_manager.is_stable():
            stability_bonus = 1.0
        else:
            stability_bonus = -10.0
        
        completion_bonus = self.statistics['completed_tasks'] * 0.1
        
        reward = -cost + stability_bonus + completion_bonus
        
        rewards = [reward / self.n_agents for _ in range(self.n_agents)]
        
        return rewards
    
    def _update_statistics(self):
        if self.completed_tasks:
            delays = [t.total_delay for t in self.completed_tasks]
            self.statistics['avg_completion_time'] = np.mean(delays)
            self.statistics['avg_delay'] = np.mean(delays)
        
        total_comp = self.statistics['uav_computation'] + self.statistics['vessel_computation']
        self.statistics['total_computation'] = total_comp
        
        if self.statistics['completed_tasks'] > 0:
            edge_tasks = sum(1 for t in self.completed_tasks 
                           if t.processing_location and 
                           (t.processing_location.startswith('uav') or 
                            t.processing_location.startswith('vessel')))
            self.statistics['edge_computing_percentage'] = (
                edge_tasks / self.statistics['completed_tasks'] * 100
            )
        
        self.statistics['queue_stability'] = self.queue_manager.is_stable()
    
    def _check_done(self) -> List[bool]:
        is_done = self.current_time_slot >= self.num_time_slots
        return [is_done] * self.n_agents
    
    def seed(self, seed: int = None):
        if seed is not None:
            np.random.seed(seed)
        return [seed]
    
    def render(self):
        pass
    
    def close(self):
        pass
