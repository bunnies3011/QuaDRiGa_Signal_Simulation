# cco_env.py
import numpy as np
import json

from simulated_rsrp import SimulatedRSRP
from problem_formulation import CCORasterBlanketFormulation


class CCOEnv:
    """
    Môi trường RL tối ưu Coverage/Capacity theo đúng paper ICASSP 2021:
    
    - State = [under_percentage, over_percentage]
    - Action = 30-dimensional vector (downtilt_i, power_i)
    - Reward = - objective_value
        objective_value = λ * weak_coverage_score + (1-λ) * over_coverage_score
    """

    def __init__(self, config_path="cco_noop.json"):
        # Load config
        with open(config_path, "r") as f:
            config = json.load(f)

        # --- Simulator (y như visualize_rsrp.py) ---
        self.sim = SimulatedRSRP.construct_from_npz_files(
            config["simulated_rsrp"]["path"],
            tuple(config["simulated_rsrp"]["power_range"])
        )

        # --- Problem Formulation class ---
        pf_params = config["problem_formulation"]["parameters"]
        self.pf = CCORasterBlanketFormulation(**pf_params)

        # Extract λ (lambda_weight)
        self.lambda_weight = pf_params["lambda_weight"]

        # Sector information
        self.downtilt_range, self.power_range = self.sim.get_configuration_range()
        _, self.num_sectors = self.sim.get_configuration_shape()

        # --- RL Dimensions (according to paper) ---
        self.action_dim = self.num_sectors * 2  # (downtilt, power)
        self.state_dim = 2                      # [under_pct, over_pct]

    # -------------------------------------------------------
    def _scale_action(self, a_norm):
        """ Map [-1,1]^30 → physical antenna configuration """
        a_real = np.zeros_like(a_norm)
        for i in range(self.num_sectors):
            di = a_norm[2*i]
            pi = a_norm[2*i+1]

            # scale downtilt
            a_real[2*i] = self.downtilt_range[0] + (di + 1) / 2 * (
                self.downtilt_range[1] - self.downtilt_range[0]
            )

            # scale power
            a_real[2*i+1] = self.power_range[0] + (pi + 1) / 2 * (
                self.power_range[1] - self.power_range[0]
            )

        return a_real

    def _real_config_from_action(self, a_real):
        """Convert 30-dim vector → (downtilt vector, power vector)"""
        d = a_real[0::2]
        p = a_real[1::2]
        return (d, p)

    # -------------------------------------------------------
    def _compute_state_and_reward(self, rsrp_map, interf_map):
        """
        Compute:
        - under_percentage
        - over_percentage
        - reward = - objective_value
        """
        # Under & over percentages
        under_pct, over_pct = self.pf.get_weak_over_coverage_area_percentages(
            rsrp_map, interf_map
        )

        # Compute combined objective (paper's reward function)
        objective_value = self.pf.get_objective_value(rsrp_map, interf_map)

        # RL reward = negative objective (since RL maximizes reward)
        reward = -objective_value

        state = np.array([under_pct, over_pct], dtype=np.float32)
        return state, reward

    # -------------------------------------------------------
    def step(self, action_norm):
        """
        Input action_norm ∈ [-1,1]^30
        """
        a_real = self._scale_action(action_norm)
        config = self._real_config_from_action(a_real)

        # Simulator output
        rsrp_map, interf_map, _ = self.sim.get_RSRP_and_interference_powermap(config)

        next_state, reward = self._compute_state_and_reward(rsrp_map, interf_map)

        done = False  # Episodic termination not used in ICASSP
        return next_state, reward, done

    # -------------------------------------------------------
    def reset(self):
        """Random initial configuration"""
        d0 = np.random.uniform(*self.downtilt_range, size=self.num_sectors)
        p0 = np.random.uniform(*self.power_range, size=self.num_sectors)
        config = (d0, p0)

        rsrp_map, interf_map, _ = self.sim.get_RSRP_and_interference_powermap(config)

        state, _ = self._compute_state_and_reward(rsrp_map, interf_map)
        return state
