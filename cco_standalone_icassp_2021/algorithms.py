from typing import Tuple

import numpy as np
import torch
from problem_formulation import (
    CCORasterBlanketFormulation,
)
from simulated_rsrp import SimulatedRSRP


class CCOAlgorithm:
    def __init__(
        self,
        simulated_rsrp: SimulatedRSRP,
        problem_formulation: CCORasterBlanketFormulation,
        **kwargs,
    ):
        self.simulated_rsrp = simulated_rsrp
        self.problem_formulation = problem_formulation

        # Get configuration range for downtilts and powers
        (
            self.downtilt_range,
            self.power_range,
        ) = self.simulated_rsrp.get_configuration_range()

        # Get the number of total sectors
        _, self.num_sectors = self.simulated_rsrp.get_configuration_shape()

    def step(self) -> Tuple[
        Tuple[np.ndarray, np.ndarray],
        float,
        Tuple[float, float],
    ]:
        """Stub for one step of the algorithm.

            Return tuple::
            1. configuration : nested tuple of optimal tilts and optimal powers
            2. reward : weigted combination of metrics
            3. metrics : tuple of dual objectives : under-coverage and over-coverage
        """
        return [None, None], None, [0.0, 0.0]


class RandomSelection(CCOAlgorithm):
    def __init__(
        self,
        simulated_rsrp: SimulatedRSRP,
        problem_formulation: CCORasterBlanketFormulation,
        **kwargs,
    ):
        super().__init__(
            simulated_rsrp=simulated_rsrp, problem_formulation=problem_formulation
        )

    def step(self) -> Tuple[Tuple[np.ndarray, np.ndarray], float, Tuple[float, float]]:
        # Random select powers and downtilts

        downtilts_for_sectors = np.random.uniform(
            self.downtilt_range[0], self.downtilt_range[1], self.num_sectors
        )
        power_for_sectors = np.random.uniform(
            self.power_range[0], self.power_range[1], self.num_sectors
        )

        # power_for_sectors = [max_tx_power_dBm] * num_sectors
        configuration = (downtilts_for_sectors, power_for_sectors)
        # Get the rsrp and interferences powermap
        (
            rsrp_powermap,
            interference_powermap,
            _,
        ) = self.simulated_rsrp.get_RSRP_and_interference_powermap(configuration)

        # According to the problem formulation, calculate the reward
        reward = self.problem_formulation.get_objective_value(
            rsrp_powermap, interference_powermap
        )

        # Get the metrics
        metrics = self.problem_formulation.get_weak_over_coverage_area_percentages(
            rsrp_powermap, interference_powermap
        )
        return configuration, reward, metrics

# ddpg_cco_singlefile.py
"""
Single-file DDPG (paper-style) for CCO.
Reference paper (uploaded): /mnt/data/10.1109@ICASSP39728.2021.9414155.pdf

Expectations:
- CCOAlgorithm, SimulatedRSRP, CCORasterBlanketFormulation available in scope.
- This file defines DDPGAlgorithm (inherits CCOAlgorithm).
- main() at bottom demonstrates usage (Random vs DDPG).
"""

# ddpg_cco_stable.py
# DDPG (stable) for CCO with state = interleaved 30 dims [d1,p1,...,d15,p15]
# Reference / paper (local copy): /mnt/data/10.1109@ICASSP39728.2021.9414155.pdf
#
# Assumes:
#   - CCOAlgorithm base class available in scope
#   - SimulatedRSRP and CCORasterBlanketFormulation available
# Paste this file into your repo and import/use the DDPGAlgorithm class.

import os
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple

# -------------------------------
# Replay buffer (simple)
# -------------------------------
class ReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, max_size: int = 20000):
        self.max_size = int(max_size)
        self.ptr = 0
        self.size = 0
        self.state = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.next_state = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((self.max_size, action_dim), dtype=np.float32)
        self.reward = np.zeros((self.max_size, 1), dtype=np.float32)
        self.done = np.zeros((self.max_size, 1), dtype=np.float32)

    def add(self, s, a, r, s2, done=0.0):
        self.state[self.ptr] = s
        self.action[self.ptr] = a
        self.reward[self.ptr] = r
        self.next_state[self.ptr] = s2
        self.done[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int):
        assert self.size > 0, "Replay buffer empty"
        idx = np.random.randint(0, self.size, size=int(batch_size))
        return (
            torch.from_numpy(self.state[idx]).float(),
            torch.from_numpy(self.action[idx]).float(),
            torch.from_numpy(self.reward[idx]).float(),
            torch.from_numpy(self.next_state[idx]).float(),
            torch.from_numpy(self.done[idx]).float(),
        )

    def clear(self):
        self.ptr = 0
        self.size = 0

# -------------------------------
# Networks (with LayerNorm for stability)
# -------------------------------
class ActorNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden)
        self.ln1 = nn.LayerNorm(hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.ln2 = nn.LayerNorm(hidden)
        self.fc3 = nn.Linear(hidden, action_dim)
        self.act = nn.Tanh()  # normalized action in [-1,1]

        # init small weights for stability
        nn.init.xavier_uniform_(self.fc1.weight, gain=1.0)
        nn.init.xavier_uniform_(self.fc2.weight, gain=1.0)
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)

    def forward(self, x):
        x = torch.relu(self.ln1(self.fc1(x)))
        x = torch.relu(self.ln2(self.fc2(x)))
        x = self.act(self.fc3(x))
        return x


class CriticNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden)
        self.ln1 = nn.LayerNorm(hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.ln2 = nn.LayerNorm(hidden)
        self.fc3 = nn.Linear(hidden, 1)

        nn.init.xavier_uniform_(self.fc1.weight, gain=1.0)
        nn.init.xavier_uniform_(self.fc2.weight, gain=1.0)
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        x = torch.relu(self.ln1(self.fc1(x)))
        x = torch.relu(self.ln2(self.fc2(x)))
        return self.fc3(x)


# -------------------------------
# Stable DDPGAlgorithm (inherits CCOAlgorithm)
# -------------------------------
class DDPGAlgorithm(CCOAlgorithm):
    """
    Stable DDPG for CCO where state = interleaved [d1,p1,d2,p2,...].
    Key stabilizations:
      - state normalization per-dimension before feeding to networks
      - actor update frequency control (policy_delay)
      - gradient clipping
      - automatic replay flush / random restart on plateau
      - reward scaling option
    """

    def __init__(
        self,
        simulated_rsrp,
        problem_formulation,
        gamma: float = 0.0,                 # immediate reward works well for static envs
        tau: float = 0.005,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        buffer_size: int = 20000,
        batch_size: int = 64,
        noise_std_init: float = 0.6,        # more exploration initially
        noise_decay: float = 0.9997,
        min_noise_std: float = 0.05,
        max_delta_tilt: float = 3.0,
        max_delta_power: float = 3.0,
        policy_delay: int = 2,              # update actor once every policy_delay critic updates
        grad_clip: float = 1.0,
        reward_scaling: float = 1.0,        # divide objective by this (set auto later)
        device: str = None,
        hidden: int = 256,
        restart_patience: int = 3000,       # if no improvement in this many steps -> random restart
        min_improve: float = 1e-4,          # minimal improvement to reset patience
        **kwargs
    ):
        super().__init__(simulated_rsrp=simulated_rsrp, problem_formulation=problem_formulation, **kwargs)

        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

        # dims
        self.num_sectors = self.num_sectors
        self.state_dim = 2 * self.num_sectors
        self.action_dim = 2 * self.num_sectors

        # ranges & scale
        self.dt_min, self.dt_max = self.downtilt_range
        self.p_min, self.p_max = self.power_range
        # compute mid and spans for normalization
        self.dt_mid = 0.5 * (self.dt_min + self.dt_max)
        self.dt_span = 0.5 * (self.dt_max - self.dt_min) + 1e-8
        self.p_mid = 0.5 * (self.p_min + self.p_max)
        self.p_span = 0.5 * (self.p_max - self.p_min) + 1e-8

        # hyperparams
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.batch_size = int(batch_size)
        self.grad_clip = float(grad_clip)
        self.policy_delay = int(policy_delay)
        self.reward_scaling = float(reward_scaling)

        # exploration
        self.noise_std = float(noise_std_init)
        self.noise_decay = float(noise_decay)
        self.min_noise_std = float(min_noise_std)

        # deltas
        self.max_delta_tilt = float(max_delta_tilt)
        self.max_delta_power = float(max_delta_power)

        # networks
        self.actor = ActorNet(self.state_dim, self.action_dim, hidden=hidden).to(self.device)
        self.actor_target = ActorNet(self.state_dim, self.action_dim, hidden=hidden).to(self.device)
        self.critic = CriticNet(self.state_dim, self.action_dim, hidden=hidden).to(self.device)
        self.critic_target = CriticNet(self.state_dim, self.action_dim, hidden=hidden).to(self.device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # replay
        self.replay = ReplayBuffer(self.state_dim, self.action_dim, max_size=buffer_size)

        # counters & logging
        self.total_steps = 0
        self.critic_updates = 0
        self.best_objective = float("inf")
        self.best_configuration = None

        # restart logic
        self.restart_patience = int(restart_patience)
        self.min_improve = float(min_improve)
        self.last_improve_step = 0

        # initialize state from a (possibly deterministic) baseline or random
        dt_init, p_init = self._sample_random_configuration()
        self.state = self._configuration_to_state(dt_init, p_init)

        # auto set reward scaling if not provided (estimate by one forward)
        try:
            conf = self._state_to_configuration(self.state)
            rsrp, interf, _ = self.simulated_rsrp.get_RSRP_and_interference_powermap(conf)
            obj = float(self.problem_formulation.get_objective_value(rsrp, interf))
            # Avoid divide-by-zero
            self.reward_scaling = max(self.reward_scaling, obj if obj > 0 else 1.0)
        except Exception:
            # leave given reward_scaling
            pass

    # -----------------------
    # Helpers: mapping & normalize
    # -----------------------
    def _sample_random_configuration(self) -> Tuple[np.ndarray, np.ndarray]:
        downtilts = np.random.uniform(self.dt_min, self.dt_max, size=self.num_sectors).astype(np.float32)
        powers = np.random.uniform(self.p_min, self.p_max, size=self.num_sectors).astype(np.float32)
        return downtilts, powers

    def _configuration_to_state(self, downtilts: np.ndarray, powers: np.ndarray) -> np.ndarray:
        s = np.empty(2 * self.num_sectors, dtype=np.float32)
        s[0::2] = downtilts
        s[1::2] = powers
        return s

    def _state_to_configuration(self, s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return s[0::2].copy(), s[1::2].copy()

    def _clip_state(self, s: np.ndarray) -> np.ndarray:
        s2 = s.copy()
        s2[0::2] = np.clip(s2[0::2], self.dt_min, self.dt_max)
        s2[1::2] = np.clip(s2[1::2], self.p_min, self.p_max)
        return s2.astype(np.float32)

    def _normalize_state(self, s: np.ndarray) -> np.ndarray:
        """
        Normalize interleaved state to roughly [-1,1] per-dimension:
          tilt_norm = (d - dt_mid) / dt_span
          power_norm = (p - p_mid) / p_span
        """
        s2 = s.copy().astype(np.float32)
        s2[0::2] = (s2[0::2] - self.dt_mid) / self.dt_span
        s2[1::2] = (s2[1::2] - self.p_mid) / self.p_span
        return s2

    def _denormalize_action(self, a_norm: np.ndarray) -> np.ndarray:
        """
        a_norm is action in [-1,1] interleaved; map to physical delta:
          delta_{tilt} = a_norm_even * max_delta_tilt
          delta_{power} = a_norm_odd * max_delta_power
        """
        delta = np.zeros_like(a_norm, dtype=np.float32)
        delta[0::2] = a_norm[0::2] * self.max_delta_tilt
        delta[1::2] = a_norm[1::2] * self.max_delta_power
        return delta

    def _soft_update(self, src: nn.Module, tgt: nn.Module):
        for p_src, p_tgt in zip(src.parameters(), tgt.parameters()):
            p_tgt.data.copy_(self.tau * p_src.data + (1.0 - self.tau) * p_tgt.data)

    # -----------------------
    # training internal
    # -----------------------
    def _train_step(self):
        if self.replay.size < self.batch_size:
            return None, None  # nothing to report

        s_b, a_b, r_b, s2_b, done_b = self.replay.sample(self.batch_size)
        s_b = s_b.to(self.device)
        a_b = a_b.to(self.device)
        r_b = r_b.to(self.device)
        s2_b = s2_b.to(self.device)
        done_b = done_b.to(self.device)

        # Critic update
        with torch.no_grad():
            a2 = self.actor_target(s2_b)
            q2 = self.critic_target(s2_b, a2)
            y = r_b + self.gamma * (1.0 - done_b) * q2

        q = self.critic(s_b, a_b)
        loss_critic = nn.MSELoss()(q, y)

        self.critic_opt.zero_grad()
        loss_critic.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.critic_opt.step()

        # Actor update (delayed)
        loss_actor = None
        if (self.critic_updates % self.policy_delay) == 0:
            a_pred = self.actor(s_b)
            loss_actor = -self.critic(s_b, a_pred).mean()
            self.actor_opt.zero_grad()
            loss_actor.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
            self.actor_opt.step()

            # soft update actor target
            self._soft_update(self.actor, self.actor_target)

        # soft update critic target always
        self._soft_update(self.critic, self.critic_target)

        self.critic_updates += 1

        return loss_critic.item(), (loss_actor.item() if loss_actor is not None else None)

    # -----------------------
    # main step (public)
    # -----------------------
    def step(self) -> Tuple[Tuple[np.ndarray, np.ndarray], float, Tuple[float, float]]:
        """
        One DDPG step:
          - normalize state
          - actor outputs normalized action [-1,1]
          - add gaussian noise
          - denormalize to delta, apply and clip
          - simulate RF (get objective, under, over)
          - store (s, a_norm, r_scaled, s_next)
          - train one step
          - restart if no improvement for long
        """
        self.total_steps += 1

        s = self.state.copy()  # physical units
        s_norm = self._normalize_state(s)
        s_t = torch.from_numpy(s_norm).float().unsqueeze(0).to(self.device)

        # actor policy -> normalized action in [-1,1]
        with torch.no_grad():
            a_norm = self.actor(s_t).cpu().numpy().flatten()

        # add action noise
        noise = np.random.normal(0.0, self.noise_std, size=self.action_dim).astype(np.float32)
        a_noisy = np.clip(a_norm + noise, -1.0, 1.0)

        # debug logs: monitor action spread and state change
        if self.total_steps % 100 == 0:
            print(f"[DBG] step={self.total_steps} | action mean={a_noisy.mean():.4f} std={a_noisy.std():.4f}")

        # convert to delta and apply
        delta = self._denormalize_action(a_noisy)
        s_next = s + delta
        s_next = self._clip_state(s_next)

        # compute physical configuration and simulate RF
        configuration = self._state_to_configuration(s_next)
        rsrp_map, interf_map, _ = self.simulated_rsrp.get_RSRP_and_interference_powermap(configuration)

        under, over = self.problem_formulation.get_weak_over_coverage_area_percentages(rsrp_map, interf_map)
        objective = float(self.problem_formulation.get_objective_value(rsrp_map, interf_map))

        # scaled reward for RL (negative objective, normalized)
        # dividing by reward_scaling stabilizes scale of Q-values
        reward_rl = -objective / max(1.0, self.reward_scaling)

        # store transition (store a_noisy as the policy input)
        self.replay.add(s.astype(np.float32), a_noisy.astype(np.float32), np.array([reward_rl], dtype=np.float32), s_next.astype(np.float32), 0.0)

        # train
        train_report = self._train_step()
        if train_report is not None:
            loss_critic, loss_actor = train_report
            if self.total_steps % 200 == 0:
                print(f"[TRAIN] step={self.total_steps} critic_loss={loss_critic:.6f} actor_loss={loss_actor}")

        # update noise schedule (decay)
        self.noise_std = max(self.noise_std * self.noise_decay, self.min_noise_std)

        # compute mean absolute state change for debug
        if self.total_steps % 100 == 0:
            mean_change = np.mean(np.abs(s_next - s))
            print(f"[DBG] mean_state_change={mean_change:.6f} | objective={objective:.4f} | under={under*100:.2f}% over={over*100:.2f}%")

        # restart logic: if no improvement for long, random restart + optionally flush buffer
        if objective + self.min_improve < self.best_objective:
            # improvement found
            self.best_objective = objective
            self.best_configuration = configuration
            self.last_improve_step = self.total_steps
            # optional: shrink exploration a bit
            self.noise_std = max(self.noise_std * 0.95, self.min_noise_std)
        else:
            # no improvement
            if (self.total_steps - self.last_improve_step) > self.restart_patience:
                # random restart: new initial configuration
                print(f"[RESTART] No improvement for {self.restart_patience} steps. Random restart & flush replay.")
                dt, p = self._sample_random_configuration()
                self.state = self._configuration_to_state(dt, p)
                # flush to force exploration from new region
                self.replay.clear()
                # increase noise to encourage exploration
                self.noise_std = max(self.noise_std, 0.8)
                self.last_improve_step = self.total_steps
                return configuration, objective, (under, over)

        # finalize state
        self.state = s_next

        return configuration, objective, (under, over)

    # -----------------------
    # save/load helpers
    # -----------------------
    def save(self, dirpath: str):
        os.makedirs(dirpath, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(dirpath, "actor.pth"))
        torch.save(self.critic.state_dict(), os.path.join(dirpath, "critic.pth"))

    def load(self, dirpath: str):
        self.actor.load_state_dict(torch.load(os.path.join(dirpath, "actor.pth"), map_location=self.device))
        self.critic.load_state_dict(torch.load(os.path.join(dirpath, "critic.pth"), map_location=self.device))

# -------------------------------
# Example quick-run (use only if simulator & problem_formulation available)
# -------------------------------
def quick_run_example(config_path="cco_noop.json", ddpg_steps=10000):
    import json
    from simulated_rsrp import SimulatedRSRP
    from problem_formulation import CCORasterBlanketFormulation

    with open(config_path, "r") as f:
        cfg = json.load(f)

    sim_path = cfg["simulated_rsrp"]["path"]
    power_range = tuple(cfg["simulated_rsrp"]["power_range"])
    simulated_rsrp = SimulatedRSRP.construct_from_npz_files(sim_path, power_range)

    pf_params = cfg["problem_formulation"]["parameters"]
    pf = CCORasterBlanketFormulation(**pf_params)

    ddpg = DDPGAlgorithm(
        simulated_rsrp=simulated_rsrp,
        problem_formulation=pf,
        gamma=0.0,             # immediate reward often better for this static task
        tau=0.005,
        actor_lr=1e-4,
        critic_lr=1e-3,
        buffer_size=20000,
        batch_size=64,
        noise_std_init=0.8,
        noise_decay=0.9997,
        min_noise_std=0.05,
        max_delta_tilt=3.0,
        max_delta_power=3.0,
        policy_delay=2,
        grad_clip=1.0,
        reward_scaling=1.0,    # auto-adjusted at init
        restart_patience=3000,
        min_improve=1e-4
    )

    best_obj = float("inf")
    start = time.time()
    for t in range(1, ddpg_steps + 1):
        conf, obj, (u, o) = ddpg.step()
        if obj < best_obj:
            best_obj = obj
            best_conf = conf
        if t % 200 == 0:
            print(f"[MAIN] step={t} obj={obj:.2f} | under={u*100:.2f}% | over={o*100:.2f}% | noise={ddpg.noise_std:.4f}")
    elapsed = time.time() - start
    print(f"Done {ddpg_steps} steps. Best obj={best_obj:.4f} in {elapsed:.1f}s")
    ddpg.save("models/ddpg_stable_run")
    return ddpg, best_conf, best_obj

# If executed as script, attempt quick run (will fail gracefully if modules missing)
if __name__ == "__main__":
    try:
        quick_run_example(ddpg_steps=30000)
    except Exception as e:
        print("Quick run failed (missing simulator or config). Exception:", e)
        print("Import DDPGAlgorithm class and run from your training script.")
