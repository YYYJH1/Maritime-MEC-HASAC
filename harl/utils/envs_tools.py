import os
import random
import numpy as np
import torch
from harl.envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv


def check(value):
    output = torch.from_numpy(value) if isinstance(value, np.ndarray) else value
    return output


def get_shape_from_obs_space(obs_space):
    if obs_space.__class__.__name__ == "Box":
        obs_shape = obs_space.shape
    elif obs_space.__class__.__name__ == "list":
        obs_shape = obs_space
    else:
        raise NotImplementedError
    return obs_shape


def get_shape_from_act_space(act_space):
    if act_space.__class__.__name__ == "Discrete":
        act_shape = 1
    elif act_space.__class__.__name__ == "MultiDiscrete":
        act_shape = act_space.shape[0]
    elif act_space.__class__.__name__ == "Box":
        act_shape = act_space.shape[0]
    elif act_space.__class__.__name__ == "MultiBinary":
        act_shape = act_space.shape[0]
    return act_shape


def make_train_env(env_name, seed, n_threads, env_args):
    def get_env_fn(rank):
        def init_env():
            if env_name == "maritime_mec":
                from harl.envs.mec.maritime_mec_env import MaritimeMECEnv
                env = MaritimeMECEnv(env_args)
            else:
                print("Can not support the " + env_name + " environment.")
                raise NotImplementedError
            env.seed(seed + rank * 1000)
            return env

        return init_env

    if n_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(n_threads)])


def make_eval_env(env_name, seed, n_threads, env_args):
    def get_env_fn(rank):
        def init_env():
            if env_name == "maritime_mec":
                from harl.envs.mec.maritime_mec_env import MaritimeMECEnv
                env = MaritimeMECEnv(env_args)
            else:
                print("Can not support the " + env_name + " environment.")
                raise NotImplementedError
            env.seed(seed * 50000 + rank * 10000)
            return env

        return init_env

    if n_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(n_threads)])


def make_render_env(env_name, seed, env_args):
    manual_render = True
    manual_expand_dims = True
    manual_delay = True
    env_num = 1
    
    if env_name == "maritime_mec":
        from harl.envs.mec.maritime_mec_env import MaritimeMECEnv
        env = MaritimeMECEnv(env_args)
        env.seed(seed * 60000)
    else:
        print("Can not support the " + env_name + " environment.")
        raise NotImplementedError
    
    return env, manual_render, manual_expand_dims, manual_delay, env_num


def set_seed(args):
    if not args["seed_specify"]:
        args["seed"] = np.random.randint(1000, 10000)
    random.seed(args["seed"])
    np.random.seed(args["seed"])
    os.environ["PYTHONHASHSEED"] = str(args["seed"])
    torch.manual_seed(args["seed"])
    torch.cuda.manual_seed(args["seed"])
    torch.cuda.manual_seed_all(args["seed"])


def get_num_agents(env, env_args, envs):
    if env == "maritime_mec":
        return envs.n_agents
    else:
        raise NotImplementedError(f"Environment {env} not supported")
