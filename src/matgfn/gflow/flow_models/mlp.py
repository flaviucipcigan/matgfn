import torch
import torch.nn as nn
from torch.distributions import Categorical


class MLP(nn.Module):
    def __init__(self, dimensions, num_embed=64, num_hid=256):
        super().__init__()
        
        self.dimensions = dimensions
        self.n_actions = dimensions + 1

        self.logZ = nn.Parameter(torch.ones(1))
        self.mlp = nn.Sequential(nn.Linear(self.dimensions, num_embed), \
                                nn.LeakyReLU(),
                                nn.Linear(num_embed, num_hid), 
                                nn.LeakyReLU(),
                                nn.Linear(num_hid, self.n_actions * 2))

    def sample_action(self, obs, info):
        x = torch.tensor(obs).float()
        logits = self.mlp(x)

        # Separate them into forward and backwards
        forward_logits, backward_logits = logits[:self.n_actions], \
            logits[self.n_actions:]

        # Find out the allowed forward and backward actions
        allowed_forward_actions = info["allowed_forward_actions"]
        allowed_backward_actions = info["allowed_backward_actions"]

        boolean_forward_mask = [x not in allowed_forward_actions 
                                for x in range(0, self.n_actions)]
        boolean_backward_mask = [x not in allowed_backward_actions 
                                 for x in range(0, self.n_actions)]

        logits_forward_mask = torch.zeros(self.n_actions)
        logits_backward_mask = torch.zeros(self.n_actions)

        logits_forward_mask[boolean_forward_mask] = -10000
        logits_backward_mask[boolean_backward_mask] = -10000

        # Mask logits
        masked_forward_logits = forward_logits + logits_forward_mask
        masked_backward_logits = backward_logits + logits_backward_mask

        # Sampler
        forward_flow_categorical = Categorical(logits=masked_forward_logits)
        backward_flow_categorical = Categorical(logits=masked_backward_logits)
        
        # Action
        action = forward_flow_categorical.sample()
        
        # Return
        return int(action), forward_flow_categorical.logits, \
            backward_flow_categorical.logits