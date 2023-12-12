
# Torch
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.optim import Adam

# Utils
from tqdm import tqdm

class TrajectoryBalanceGFlowNet(nn.Module):
    def __init__(self, env, flow_model):
        super().__init__()

        # gym.Env wih reset(), update(action), action_space, observation_space
        self.env = env

        # (obs, info) -> (action, forward_logits, backward_logits)
        self.flow_model = flow_model

    def trajectory_balance_loss(self, trajectory, reward):
        sum_log_forward_flow = 0
        sum_log_backward_flow = 0

        for i in range(0, len(trajectory)): 
            (_, info, action, log_forward_flow, _) = trajectory[i]
            assert action in info["allowed_forward_actions"]
            sum_log_forward_flow  += log_forward_flow[action]

            if i+1 < len(trajectory):
                (_, info, _, _, log_backward_flow) = trajectory[i+1]
                assert action in info["allowed_backward_actions"]
                sum_log_backward_flow += log_backward_flow[action]

        reward_as_tensor = torch.tensor(reward).float()
        loss = (self.flow_model.logZ + sum_log_forward_flow - 
                torch.log(reward_as_tensor).clip(-20) - sum_log_backward_flow).pow(2)
        
        return loss

    def sample(self, grad=True):
        obs, info = self.env.reset()
        terminated = False        
        trajectory = []

        # Enable / disable gradients depending on training or sampling
        torch.set_grad_enabled(grad)

        # Sample using Flow model

        while not terminated:
            action, log_forward_flow, log_backward_flow = self.flow_model.sample_action(obs, info)
            assert action in self.env.action_space
            assert action in info["allowed_forward_actions"]
            
            trajectory.append((obs, info, action, log_forward_flow, log_backward_flow))
            obs, reward, terminated, _, info = self.env.step(action)

        # Return
        loss = self.trajectory_balance_loss(trajectory, reward)
        return obs, info, reward, loss

    def fit(self, learning_rate, num_episodes, minibatch_size):
        self.train(True)
        opt = Adam(self.parameters(), lr=learning_rate)

        observations = []
        infos = []
        rewards = []
        losses = []
        logZs = []

        minibatch_loss = 0

        for episode_id in (p := tqdm(range(1, num_episodes+1))):
            obs, info, reward, loss=self.sample()
                
            observations.append(obs)
            infos.append(info)
            rewards.append(reward)
            losses.append(float(loss))
            logZs.append(float(self.flow_model.logZ))

            minibatch_loss += loss

            if episode_id % minibatch_size == 0:
                p.set_description(f"{minibatch_loss.item():.3f}, {float(self.flow_model.logZ)}")

                # Backpropagate
                minibatch_loss.backward()
                opt.step()
                opt.zero_grad()

                # Gather statistics
                # losses.append(minibatch_loss.item())
                # logZs.append(float(self.flow_model.logZ))
            
                # Reset loss
                minibatch_loss = 0

        return observations, infos, rewards, losses, logZs
