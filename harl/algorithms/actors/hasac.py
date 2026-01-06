import numpy as np
import torch
import torch.nn as nn
from harl.models.policy_models.stochastic_policy import StochasticPolicy
from harl.utils.envs_tools import check
from harl.algorithms.actors.off_policy_base import OffPolicyBase


class HASAC(OffPolicyBase):
    
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        self.args = args
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.polyak = args["tau"]
        self.lr = args["lr"]
        self.obs_space = obs_space
        self.act_space = act_space
        
        self.actor = StochasticPolicy(args, obs_space, act_space, device)
        self.target_actor = StochasticPolicy(args, obs_space, act_space, device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.lr,
            eps=args["opti_eps"],
            weight_decay=args["weight_decay"],
        )
        
        for param in self.target_actor.parameters():
            param.requires_grad = False
    
    def get_actions(self, obs, available_actions=None, stochastic=True):
        obs = check(obs).to(**self.tpdv)
        batch_size = obs.shape[0]
        rnn_states = torch.zeros(batch_size, 1, 1).to(**self.tpdv)
        masks = torch.ones(batch_size, 1).to(**self.tpdv)
        
        deterministic = not stochastic
        actions, _, _ = self.actor(obs, rnn_states, masks, available_actions, deterministic)
        return actions
    
    def get_actions_with_logprobs(self, obs, available_actions=None):
        obs = check(obs).to(**self.tpdv)
        batch_size = obs.shape[0]
        rnn_states = torch.zeros(batch_size, 1, 1).to(**self.tpdv)
        masks = torch.ones(batch_size, 1).to(**self.tpdv)
        
        actions, log_probs, _ = self.actor(obs, rnn_states, masks, available_actions, False)
        return actions, log_probs
    
    def get_target_actions(self, obs, available_actions=None):
        obs = check(obs).to(**self.tpdv)
        batch_size = obs.shape[0]
        rnn_states = torch.zeros(batch_size, 1, 1).to(**self.tpdv)
        masks = torch.ones(batch_size, 1).to(**self.tpdv)
        
        actions, _, _ = self.target_actor(obs, rnn_states, masks, available_actions, False)
        return actions
        
    def soft_update(self):
        for target_param, param in zip(
            self.target_actor.parameters(), self.actor.parameters()
        ):
            target_param.data.copy_(
                target_param.data * (1.0 - self.polyak) + param.data * self.polyak
            )
    
    def prep_training(self):
        self.actor.train()
        self.target_actor.train()
    
    def prep_rollout(self):
        self.actor.eval()
        self.target_actor.eval()
    
    def turn_on_grad(self):
        for p in self.actor.parameters():
            p.requires_grad = True

    def turn_off_grad(self):
        for p in self.actor.parameters():
            p.requires_grad = False
    
    def save(self, save_dir, agent_id):
        torch.save(
            self.actor.state_dict(), 
            str(save_dir) + f"/actor_agent{agent_id}.pt"
        )
        torch.save(
            self.target_actor.state_dict(),
            str(save_dir) + f"/target_actor_agent{agent_id}.pt",
        )

    def restore(self, model_dir, agent_id):
        actor_state_dict = torch.load(
            str(model_dir) + f"/actor_agent{agent_id}.pt"
        )
        self.actor.load_state_dict(actor_state_dict)
        target_actor_state_dict = torch.load(
            str(model_dir) + f"/target_actor_agent{agent_id}.pt"
        )
        self.target_actor.load_state_dict(target_actor_state_dict)
