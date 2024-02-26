import torch
import numpy as np

class Policy:
    def __init__(self, model):
        self.model = model
    
    def act(self, inputs, training=False):
        inputs = torch.tensor(inputs, dtype=torch.float32)
        (mus, sigmas), values = self.model(inputs)
        dist = torch.distributions.MultivariateNormal(mus, torch.diag_embed(sigmas, 0))
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        if training:
            return {'distribution': dist,
                    'values': values}
        else:
            return {'actions': actions.detach().numpy(),
                    'log_probs': log_probs.detach().numpy(),
                    'values': values.detach().numpy()}