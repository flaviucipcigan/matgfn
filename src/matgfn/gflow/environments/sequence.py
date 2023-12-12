# Farama Gymnasium
import gymnasium as gym

class SequenceEnvironment(gym.Env):
    def __init__(self, token_vocabulary, mask, termination_token, reward_function, max_sequence_length, min_sequence_length, render_function = None):
        super().__init__()

        # Sequence definition
        assert termination_token not in token_vocabulary

        self.metadata["termination_token"] = termination_token
        self.metadata["token_vocabulary"] = token_vocabulary + [self.metadata["termination_token"]]

        self.metadata["min_sequence_length"] = min_sequence_length
        self.metadata["max_sequence_length"] = max_sequence_length
        self._reward_function = reward_function
        self._render_function = render_function
        
        self.mask = mask

        # Spaces
        # Add the termination token explcitly
        # [x1, x2, x3, TER] is a different state than [x1, x2, x3]
        self.observation_space = gym.spaces.Sequence(gym.spaces.Discrete(len(token_vocabulary) + 1))

        # An action for each token plus a termination action
        self.action_space = gym.spaces.Discrete(len(token_vocabulary) + 1)

    # Reset
    def reset(self, seed=None):

        # Seed self.np_random
        super().reset(seed=seed)

        self._sequence = []
        self._terminated = False
        
        obs = self._get_obs()
        info = self._get_info()
    
        return obs, info
    

    # Utility functions
    def _get_obs(self):
        token_vocabulary = self.metadata["token_vocabulary"]
        return [token_vocabulary.index(token) for token in self._sequence]

    def _get_info(self):
        sequence = self._sequence

        if self._terminated:
            assert self._sequence[-1] == self.metadata["termination_token"]
            # No forward actions
            allowed_forward_actions = []

            # Only backward action is un-Terminate
            allowed_backward_actions = [self.action_space.n - 1]

        if not self._terminated:
            # If smaller than the minimum sequence length, allow all actions except termination
            if (len(sequence) < self.metadata["min_sequence_length"]):
                allowed_forward_actions = self.mask.allowed_forward_actions(sequence)
            # Otherwise allow all actions if below max sequence length
            elif (len(sequence) >= self.metadata["min_sequence_length"]) and (len(sequence) < self.metadata["max_sequence_length"]):
                allowed_forward_actions = self.mask.allowed_forward_actions(sequence)
                allowed_forward_actions.append(self.action_space.n - 1)
            else: # Only alow termination action
                allowed_forward_actions = [self.action_space.n - 1]

            # Only allowed backward action is "remove previous token"
            if len(sequence) == 0:
                allowed_backward_actions = []
            else:
                allowed_backward_actions = [self.metadata["token_vocabulary"].index(sequence[-1])]

        return {"sequence": self._sequence, 
                "allowed_forward_actions": allowed_forward_actions,
                "allowed_backward_actions": allowed_backward_actions}

    def _get_reward(self):
        if not self._terminated:
            return None
        else:
            return self._reward_function(self._sequence)

    # Step
    def step(self, action):
        if not self._terminated:
            if action == self.action_space.n - 1:
                self._terminated = True
                self._sequence.append(self.metadata["termination_token"])
            else:
                token = self.metadata["token_vocabulary"][action]
                self._sequence.append(token)

        trunchated = False # Environment is never trunchated

        reward = self._get_reward()

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, self._terminated, trunchated, info

    # Render
    def render(self):
        if self._render_function:
            return self._render_function(self._sequence)
        else:
            print("".join(self._sequence))
