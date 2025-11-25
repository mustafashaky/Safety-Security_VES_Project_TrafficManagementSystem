# play_dqn_traffic_sim_2_smart.py
import sys
from pathlib import Path

import numpy as np
import pygame
import torch

import traffic_sim_2_smart as ts
from rl_env_traffic_sim_2_smart import ACTIONS

MODEL_PATH = Path(r"D:/UF_Documents/UF_Coursework/Fall_2025/EEL5632_Safety_and_Security_of_Vehicular_Electronic_Systems/Project_EEL5632_Fall2025/traffic_sim_Motor_Knights/Mustafa_traffic_sim/dqn_phase.pt")

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
    pygame.init()
    screen = pygame.display.set_mode((ts.SCREEN_WIDTH, ts.SCREEN_HEIGHT))
    pygame.display.set_caption("Play learned policy (smart intent + penalties)")
    clock = pygame.time.Clock()
    hud_font = pygame.font.SysFont("consolas", 14)
    car_font = pygame.font.SysFont("consolas", 12, bold=True)

    # smart manager under agent control
    tm = ts.TrafficManager()
    tm._choose_new_phase = lambda: None  # agent controls phases
    tm.phase_index = 0
    tm.current_phase = ts.PHASES[0]
    tm.phase_timer = 0.0

    cars = []
    spawn_timers = {d: 0.0 for d in ts.DIRECTIONS}
    phase_remaining = 6.0  # start with NS straight+right

    # load policy
    s_dim = 12 + 4 + 1
    a_dim = len(ACTIONS)
    net = Net(s_dim, a_dim)
    net.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    net.eval()

    running, crash = True, False

    try:
        while running:
            dt = clock.tick(ts.FPS) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key in (pygame.K_1, pygame.K_KP1):
                        cars.extend(ts.spawn_four_right_turns_honest())
                    elif event.key in (pygame.K_2, pygame.K_KP2):
                        cars.extend(ts.spawn_four_with_liar())

            if not crash:
                # spawn per approach
                for d in ts.DIRECTIONS:
                    spawn_timers[d] += dt
                    if spawn_timers[d] >= ts.SPAWN_INTERVAL:
                        active = [c for c in cars if c.direction == d and not c.finished and not c.crashed]
                        if len(active) < ts.MAX_CARS_PER_DIR:
                            cars.append(ts.spawn_car_for_direction(d))
                        spawn_timers[d] = 0.0

                # agent chooses next phase when current expires
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

                # advance sim (front-of-lane release logic inside manager)
                tm.update(dt, cars)
                for car in cars:
                    car.update(dt, tm, cars)

                # free movements on crash to avoid deadlock
                crash = ts.check_collisions(cars, tm)

                # keep crashed visible; drop finished
                cars = [c for c in cars if not c.finished or c.crashed]
                phase_remaining = max(0.0, phase_remaining - dt)

            # draw
            screen.fill(ts.BG_COLOR)
            ts.draw_roads(screen)
            ts.draw_signals(screen, tm)
            for car in cars:
                car.draw(screen, car_font)
            ts.draw_hud(screen, hud_font, cars, tm, crash)
            pygame.display.flip()

    except KeyboardInterrupt:
        pass
    finally:
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    main()