# replay_buffer.py
import numpy as np

class ReplayBuffer:
    def __init__(self, max_size, state_dim, action_dim):
        self.max = max_size
        self.ptr = 0
        self.size = 0

        self.s = np.zeros((max_size, state_dim), dtype=np.float32)
        self.a = np.zeros((max_size, action_dim), dtype=np.float32)
        self.r = np.zeros((max_size, 1), dtype=np.float32)
        self.s2 = np.zeros((max_size, state_dim), dtype=np.float32)
        self.done = np.zeros((max_size, 1), dtype=np.float32)

    def add(self, s, a, r, s2, d):
        self.s[self.ptr] = s
        self.a[self.ptr] = a
        self.r[self.ptr] = r
        self.s2[self.ptr] = s2
        self.done[self.ptr] = d

        self.ptr = (self.ptr + 1) % self.max
        self.size = min(self.size + 1, self.max)

    def sample(self, batch):
        idx = np.random.randint(0, self.size, size=batch)
        return (
            self.s[idx],
            self.a[idx],
            self.r[idx],
            self.s2[idx],
            self.done[idx]
        )
