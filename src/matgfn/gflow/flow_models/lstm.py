# Torch
import torch
import torch.nn as nn
from torch.distributions import Categorical


class LSTM(nn.Module):
    def __init__(self, token_vocabulary, n_actions, embedding_size=32, 
                 lstm_hidden_size=16, lstm_num_layers=2,linear_hidden_size=8):
        super().__init__()

        self.num_tokens = len(token_vocabulary)
        self.n_actions = n_actions
        self.n_flow_logits = self.n_actions * 2

        self.logZ = nn.Parameter(torch.ones(1))
        self.initial_state_flow = nn.Parameter(torch.zeros(self.n_flow_logits))
        self.embed = nn.Embedding(num_embeddings = self.num_tokens,
                                  embedding_dim = embedding_size)

        self.lstm = nn.LSTM(input_size = embedding_size,
                            hidden_size = lstm_hidden_size,
                            num_layers = lstm_num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_size, linear_hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(linear_hidden_size, self.n_flow_logits)
        )

    def sample_action(self, obs, info):
        # Use LSTM to compute the logits
        if len(obs) == 0:
            logits = self.initial_state_flow
        else:
            embeddings = self.embed(torch.tensor(obs))
            lstm_output, (h_n, c_n) = self.lstm(embeddings)
            logits = self.classifier(lstm_output[-1])

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