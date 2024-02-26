import numpy as np
import torch

class AsArray:
    """ 
       Converts lists of interactions to ndarray.
    """
    def __call__(self, trajectory):
      # Modify trajectory inplace. 
      for k, v in filter(lambda kv: kv[0] != "state",
                         trajectory.items()):
        trajectory[k] = np.asarray(v)


class NormalizeAdvantages:
    """ Normalizes advantages to have zero mean and variance 1. """
    def __call__(self, trajectory):
        adv = trajectory["advantages"]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        trajectory["advantages"] = adv


class GAE:
    """ Generalized Advantage Estimator. """
    def __init__(self, policy, gamma=0.99, lambda_=0.95):
        self.policy = policy
        self.gamma = gamma
        self.lambda_ = lambda_
    
    def __call__(self, trajectory):
        value_target = policy.act(trajectory['state']['latest_observation'])['values'][0]
        env_steps = trajectory['state']['env_steps']
        rewards = torch.tensor(trajectory['rewards'], dtype=torch.float32)
        dones = torch.tensor(trajectory['resets'], dtype=torch.float32)
        is_not_done = 1 - dones
        trajectory['values'] = torch.tensor(trajectory['values'],dtype=torch.float32)
        trajectory['advantages'] = []
        trajectory['value_targets'] = []
        gae = 0
        for step in reversed(range(env_steps)):
            if step==env_steps - 1:
                delta = rewards[step] + self.gamma*value_target*is_not_done[step] - trajectory['values'][step]
            else:
                delta = rewards[step] + self.gamma*trajectory['values'][step + 1]*is_not_done[step] -\
                        trajectory['values'][step]
            
            gae = delta + self.gamma*self.lambda_*is_not_done[step]*gae
            trajectory['advantages'].insert(0, gae)
            trajectory['value_targets'].insert(0, gae + trajectory['values'][step])
        trajectory['advantages'] = torch.tensor(trajectory['advantages'], dtype=torch.float32)
        trajectory['value_targets'] = torch.tensor(trajectory['value_targets'], dtype=torch.float32)