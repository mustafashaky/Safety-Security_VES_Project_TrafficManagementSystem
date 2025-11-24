# train_dqn_traffic_sim_3_smart.py
import random, numpy as np, torch, torch.nn as nn, torch.optim as optim
from rl_env_traffic_sim_3_smart import TMSRL, ACTIONS

class Net(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    def forward(self, x):
        return self.net(x)

def train(episodes=200):
    env = TMSRL(sim_time=120.0, dt=0.2, liar_prob=0.3, spawn_interval=2.5)
    s0 = env.reset()
    state_dim = s0.shape[0]
    action_dim = len(ACTIONS)

    q = Net(state_dim, action_dim)
    opt = optim.Adam(q.parameters(), lr=1e-3)
    gamma = 0.99
    eps = 1.0
    memory = []
    B = 256

    def select_action(s):
        nonlocal eps
        if random.random() < eps:
            return random.randrange(action_dim)
        with torch.no_grad():
            v = q(torch.tensor(s).float().unsqueeze(0))
            return int(v.argmax().item())

    for ep in range(episodes):
        s = env.reset()
        total = 0.0
        steps = 0
        while True:
            a = select_action(s)
            s2, r, done, info = env.step(a)
            memory.append((s, a, r, s2, done))
            s = s2
            total += r; steps += 1

            # learn
            if len(memory) >= B:
                batch = random.sample(memory, B)
                sb = torch.tensor([b[0] for b in batch]).float()
                ab = torch.tensor([b[1] for b in batch]).long()
                rb = torch.tensor([b[2] for b in batch]).float()
                s2b = torch.tensor([b[3] for b in batch]).float()
                db = torch.tensor([b[4] for b in batch]).float()

                q_pred = q(sb).gather(1, ab.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    q_next = q(s2b).max(1)[0]
                    target = rb + gamma * (1 - db) * q_next
                loss = (q_pred - target).pow(2).mean()
                opt.zero_grad(); loss.backward(); opt.step()

            if done:
                break

        eps = max(0.05, eps * 0.98)
        print(f"ep {ep:03d}  steps {steps:4d}  reward {total:7.2f}  eps {eps:.2f}")
    torch.save(q.state_dict(), "dqn_phase.pt")
    print("Saved model to dqn_phase.pt")

if __name__ == "__main__":
    train()