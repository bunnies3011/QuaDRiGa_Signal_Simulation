# main_train.py
from cco_env import CCOEnv
from ddpg_agent import DDPGAgent
from replay_buffer import ReplayBuffer
import numpy as np


TOTAL_STEPS = 30001       # đúng theo paper
BATCH = 64
BUFFER = 5000             # paper dùng ~5000
NOISE = 0.2               # noise exploration
LOG_INTERVAL = 200        # in log mỗi 200 bước (cho đỡ spam)


def main():
    env = CCOEnv("cco_noop.json")    

    agent = DDPGAgent(env.state_dim, env.action_dim, max_action=1.0)
    replay = ReplayBuffer(BUFFER, env.state_dim, env.action_dim)

    # State ban đầu
    s = env.reset()     # s = [under_pct, over_pct]

    for t in range(TOTAL_STEPS):

        # ---- 1. Actor chọn action ----
        a = agent.select_action(s, noise=NOISE)

        # ---- 2. Môi trường trả về state mới và reward ----
        s2, r, done = env.step(a)

        under_pct = float(s2[0])
        over_pct  = float(s2[1])

        # ---- 3. Lưu vào replay buffer ----
        replay.add(s, a, r, s2, done)

        # ---- 4. Train DDPG khi buffer đủ ----
        if replay.size > BATCH:
            agent.train(replay, BATCH)

        # ---- 5. Cập nhật state ----
        s = s2

        # ---- 6. Logging ----
        if t % LOG_INTERVAL == 0:
            print(
                f"[Step {t:5d}] "
                f"reward={r:10.2f} | "
                f"under={under_pct*100:6.2f}% | "
                f"over={over_pct*100:6.2f}%"
            )


if __name__ == "__main__":
    main()
