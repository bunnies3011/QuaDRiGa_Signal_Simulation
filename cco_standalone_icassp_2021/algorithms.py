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

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim 

from typing import Tuple

# Giả sử CCOAlgorithm, RandomSelection, SimulatedRSRP, CCORasterBlanketFormulation
# đã được định nghĩa ở trên trong cùng file.


# ==========================
#  Replay Buffer đơn giản
# ==========================
class ReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, max_size: int = 5000):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.done = np.zeros((max_size, 1), dtype=np.float32)

    def add(self, s, a, r, s2, done):
        self.state[self.ptr] = s
        self.action[self.ptr] = a
        self.reward[self.ptr] = r
        self.next_state[self.ptr] = s2
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.from_numpy(self.state[idx]),
            torch.from_numpy(self.action[idx]),
            torch.from_numpy(self.reward[idx]),
            torch.from_numpy(self.next_state[idx]),
            torch.from_numpy(self.done[idx]),
        )


# ==========================
#  Actor / Critic Networks
# ==========================
class Actor(nn.Module):
    """Actor: map state -> action (chuẩn hóa [-1, 1])"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),  # output trong [-1, 1]
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class Critic(nn.Module):
    """Critic: Q(s, a)"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


# ==========================
#  DDPGAlgorithm
# ==========================
class DDPGAlgorithm(CCOAlgorithm):
    """
    DDPG cho bài CCO, kế thừa CCOAlgorithm.

    - State s_t  = [under_t, over_t] (theo phần trăm, dạng fraction 0..1)
    - Action a_t = vector (2 * num_sectors,) trong [-1,1] -> map về (tilt, power)
    - Reward RL dùng để train = -objective (RL muốn MAX, mình muốn MIN objective)
    - Giá trị reward trả ra từ step() = objective (như RandomSelection) cho dễ so sánh.
    """

    def __init__(
        self,
        simulated_rsrp: SimulatedRSRP,
        problem_formulation: CCORasterBlanketFormulation,
        state_dim: int = 2,        # [under, over]
        gamma: float = 0.99,
        tau: float = 0.005,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        buffer_size: int = 5000,
        batch_size: int = 64,
        noise_std_init: float = 1.0,       # theo paper: 1.0
        noise_decay: float = 0.9996,       # theo paper: 0.9996
        min_noise_std: float = 0.05,
        **kwargs,
    ):
        super().__init__(
            simulated_rsrp=simulated_rsrp,
            problem_formulation=problem_formulation,
            **kwargs,
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.state_dim = state_dim
        self.action_dim = 2 * self.num_sectors  # (tilt, power) cho mỗi sector

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        # Noise cho exploration
        self.noise_std = noise_std_init
        self.noise_decay = noise_decay
        self.min_noise_std = min_noise_std

        # Actor & Critic + Target networks
        self.actor = Actor(self.state_dim, self.action_dim).to(self.device)
        self.actor_target = Actor(self.state_dim, self.action_dim).to(self.device)
        self.critic = Critic(self.state_dim, self.action_dim).to(self.device)
        self.critic_target = Critic(self.state_dim, self.action_dim).to(self.device)

        # Copy trọng số ban đầu
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            self.state_dim, self.action_dim, max_size=buffer_size
        )

        # Đếm step để decay noise
        self.total_steps = 0

        # Khởi tạo state ban đầu: random 1 cấu hình để có under/over
        init_config = self._sample_random_configuration()
        rsrp_map, interf_map, _ = self.simulated_rsrp.get_RSRP_and_interference_powermap(
            init_config
        )
        under, over = self.problem_formulation.get_weak_over_coverage_area_percentages(
            rsrp_map, interf_map
        )
        self.state = np.array([under, over], dtype=np.float32)

    # --------------------------
    #  Random config helper
    # --------------------------
    def _sample_random_configuration(self) -> Tuple[np.ndarray, np.ndarray]:
        downtilts = np.random.uniform(
            self.downtilt_range[0], self.downtilt_range[1], self.num_sectors
        )
        powers = np.random.uniform(
            self.power_range[0], self.power_range[1], self.num_sectors
        )
        return (downtilts, powers)

    # --------------------------
    #  Map action [-1,1] -> (downtilts, powers)
    # --------------------------
    def _action_to_configuration(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        action: vector (2 * num_sectors,) trong [-1, 1]
        output: (downtilts, powers) trong range thực tế
        """
        assert action.shape[0] == self.action_dim

        action_2d = action.reshape(2, self.num_sectors)
        tilt_norm = action_2d[0]   # [-1,1]
        power_norm = action_2d[1]  # [-1,1]

        dt_min, dt_max = self.downtilt_range
        p_min, p_max = self.power_range

        downtilts = dt_min + (tilt_norm + 1.0) * 0.5 * (dt_max - dt_min)
        powers    = p_min + (power_norm + 1.0) * 0.5 * (p_max - p_min)

        return downtilts, powers

    # --------------------------
    #  Soft update target networks
    # --------------------------
    def _soft_update(self, net: nn.Module, target_net: nn.Module):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

    # --------------------------
    #  Train DDPG 1 bước
    # --------------------------
    def _train_ddpg(self):
        if self.replay_buffer.size < self.batch_size:
            return

        state, action, reward, next_state, done = self.replay_buffer.sample(
            self.batch_size
        )

        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)
        done = done.to(self.device)

        # --------- Cập nhật Critic ---------
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            target_q = self.critic_target(next_state, next_action)
            target_q = reward + self.gamma * (1.0 - done) * target_q

        current_q = self.critic(state, action)
        critic_loss = nn.MSELoss()(current_q, target_q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # --------- Cập nhật Actor ---------
        actor_actions = self.actor(state)
        actor_loss = -self.critic(state, actor_actions).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # --------- Soft update target nets ---------
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

    # --------------------------
    #  Hàm step() chính
    # --------------------------
    def step(self) -> Tuple[Tuple[np.ndarray, np.ndarray], float, Tuple[float, float]]:
        """
        Một bước DDPG:

          - Input: dùng state nội bộ self.state = [under, over] (từ bước trước)
          - Actor sinh action + thêm noise
          - Map action -> configuration (downtilt, power)
          - Gọi simulated_rsrp để lấy rsrp_map & interference
          - Tính objective (>=0), reward_RL = -objective, metrics = (under, over)
          - Lưu (s, a, r_RL, s') vào replay buffer và train DDPG 1 bước
          - Cập nhật self.state = s'
          - Return:
              configuration,
              reward = objective (cho giống RandomSelection),
              metrics = (under, over)
        """
        self.total_steps += 1

        # ---- 1. State hiện tại ----
        s = self.state  # numpy (2,)
        s_tensor = torch.from_numpy(s).float().unsqueeze(0).to(self.device)

        # ---- 2. Actor chọn action ----
        with torch.no_grad():
            a_tensor = self.actor(s_tensor)
        a = a_tensor.cpu().numpy().flatten()  # [-1,1]

        # ---- 3. Thêm noise Gaussian để exploration ----
        noise = np.random.normal(0.0, self.noise_std, size=self.action_dim)
        a = a + noise
        a = np.clip(a, -1.0, 1.0)

        # Decay noise
        self.noise_std = max(self.noise_std * self.noise_decay, self.min_noise_std)

        # ---- 4. Map action -> cấu hình thực ----
        configuration = self._action_to_configuration(a)

        # ---- 5. Simulate RF ----
        rsrp_map, interf_map, _ = self.simulated_rsrp.get_RSRP_and_interference_powermap(
            configuration
        )

        # metrics: under%, over% (dạng fraction 0..1)
        under, over = self.problem_formulation.get_weak_over_coverage_area_percentages(
            rsrp_map, interf_map
        )

        # objective (>=0), reward_RL = -objective
        objective = self.problem_formulation.get_objective_value(
            rsrp_map, interf_map
        )
        reward_RL = -float(objective)

        # state mới
        s2 = np.array([under, over], dtype=np.float32)

        # ---- 6. Lưu vào buffer & train ----
        done = 0.0  # không có terminal rõ ràng
        self.replay_buffer.add(s, a, reward_RL, s2, done)
        self._train_ddpg()

        # ---- 7. Cập nhật state nội bộ ----
        self.state = s2

        # ---- 8. Trả về theo interface CCOAlgorithm ----
        metrics = (under, over)
        # reward trả ra để so với RandomSelection: dùng objective (giống paper)
        return configuration, float(objective), metrics
