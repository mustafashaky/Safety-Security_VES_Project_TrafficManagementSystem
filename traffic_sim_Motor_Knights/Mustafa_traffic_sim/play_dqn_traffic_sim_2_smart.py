# play_dqn_traffic_sim_2_smart.py
import torch, numpy as np, pygame, sys
from pathlib import Path
import train_dqn_traffic_sim_2_smart as ts  # your simulator module
from rl_env_traffic_sim_2_smart import ACTIONS  # same action space as training

class Net(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, action_dim)
        )
    def forward(self, x): return self.net(x)

def get_state(tm, phase_remaining):
    # 12 queue lengths + phase one-hot (4) + remaining time (1)
    qlens = []
    for d in ts.DIRECTIONS:
        for m in ts.INTENTS:
            q = tm.queues[(d, m)]
            qlens.append(len([c for c in q if not c.finished and not c.crashed]))
    onehot = [0.0]*4
    onehot[tm.phase_index] = 1.0
    rem = [phase_remaining / 8.0]
    return np.array(qlens + onehot + rem, dtype=np.float32)

def main():
    # pygame setup (reuse your drawing)
    pygame.init()
    screen = pygame.display.set_mode((ts.SCREEN_WIDTH, ts.SCREEN_HEIGHT))
    pygame.display.set_caption("Play learned policy")
    clock = pygame.time.Clock()
    hud_font = pygame.font.SysFont("consolas", 14)
    car_font = pygame.font.SysFont("consolas", 12, bold=True)

    # sim state
    tm = ts.TrafficManager()
    tm._choose_new_phase = lambda: None  # disable auto greedy switch
    cars = []
    spawn_timers = {d: 0.0 for d in ts.DIRECTIONS}
    phase_remaining = 0.0

    # load policy
    s_dim = 12 + 4 + 1
    a_dim = len(ACTIONS)
    net = Net(s_dim, a_dim)
    net.load_state_dict(torch.load("dqn_phase.pt", map_location="cpu"))
    net.eval()

    running, crash = True, False
    while running:
        dt = clock.tick(ts.FPS) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: running = False
                elif event.key == pygame.K_1: cars.extend(ts.spawn_four_right_turns_honest())
                elif event.key == pygame.K_2: cars.extend(ts.spawn_four_with_liar())

        if not crash:
            # spawn
            for d in ts.DIRECTIONS:
                spawn_timers[d] += dt
                if spawn_timers[d] >= ts.SPAWN_INTERVAL:
                    active = [c for c in cars if c.direction == d and not c.finished and not c.crashed]
                    if len(active) < ts.MAX_CARS_PER_DIR:
                        cars.append(ts.spawn_car_for_direction(d))
                    spawn_timers[d] = 0.0

            # choose action when phase expires
            if phase_remaining <= 0.0:
                s = get_state(tm, phase_remaining)
                with torch.no_grad():
                    a_idx = int(net(torch.tensor(s).float().unsqueeze(0)).argmax().item())
                phase, dur = ACTIONS[a_idx]
                if phase != "KEEP":
                    tm.phase_index = phase
                    tm.current_phase = ts.PHASES[tm.phase_index]
                    tm.phase_timer = 0.0
                    phase_remaining = float(dur)
                else:
                    phase_remaining = 0.5  # brief keep

            # advance sim under chosen phase
            tm.phase_timer = 0.0  # prevent internal auto switch
            tm.update(dt)
            for car in cars: car.update(dt, tm, cars)
            crash = ts.check_collisions(cars)
            cars = [c for c in cars if not c.finished or c.crashed]
            phase_remaining = max(0.0, phase_remaining - dt)

        # draw
        screen.fill(ts.BG_COLOR)
        ts.draw_roads(screen)
        ts.draw_signals(screen, tm)
        for car in cars: car.draw(screen, car_font)
        ts.draw_hud(screen, hud_font, cars, tm, crash)
        pygame.display.flip()

    pygame.quit(); sys.exit()

if __name__ == "__main__":
    main()