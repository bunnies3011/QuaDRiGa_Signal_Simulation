# ddpg_agent.py
import numpy as np
import torch
import torch.nn.functional as F

from networks import Actor, Critic


class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action=1.0):

        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

        self.max_action = max_action
        self.gamma = 0.99
        self.tau = 0.005

    # -------------------------------
    def select_action(self, s, noise=0.1):
        s = torch.FloatTensor(s.reshape(1, -1))
        a = self.actor(s).detach().numpy()[0]
        a = a + noise * np.random.randn(*a.shape)
        return np.clip(a, -self.max_action, self.max_action)

    # -------------------------------
    def train(self, replay, batch):
        s, a, r, s2, d = replay.sample(batch)

        s = torch.FloatTensor(s)
        a = torch.FloatTensor(a)
        r = torch.FloatTensor(r)
        s2 = torch.FloatTensor(s2)
        d = torch.FloatTensor(d)

        # ---- Critic update ----
        with torch.no_grad():
            a2 = self.actor_target(s2)
            target_Q = self.critic_target(s2, a2)
            target_Q = r + self.gamma * (1 - d) * target_Q

        current_Q = self.critic(s, a)
        critic_loss = F.mse_loss(current_Q, target_Q)

        self.opt_critic.zero_grad()
        critic_loss.backward()
        self.opt_critic.step()

        # ---- Actor update ----
        actor_loss = -self.critic(s, self.actor(s)).mean()

        self.opt_actor.zero_grad()
        actor_loss.backward()
        self.opt_actor.step()

        # ---- Soft update ----
        for param, target in zip(self.critic.parameters(), self.critic_target.parameters()):
            target.data = self.tau * param.data + (1 - self.tau) * target.data

        for param, target in zip(self.actor.parameters(), self.actor_target.parameters()):
            target.data = self.tau * param.data + (1 - self.tau) * target.data
