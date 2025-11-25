# rl_env_traffic_sim_2_smart.py
# Gym-like wrapper around your smart intersection (headless; no pygame window)

import numpy as np
import random

from traffic_sim_2_smart import (
    TrafficManager, DIRECTIONS, INTENTS, PHASES,
    spawn_car_for_direction, check_collisions,
)

# Actions: choose next phase and how long to hold it; KEEP = do nothing this step
ACTION_PHASES = [0, 1, 2, 3]
ACTION_DURS   = [4.0, 6.0, 8.0]
ACTIONS = [(p, d) for p in ACTION_PHASES for d in ACTION_DURS] + [("KEEP", None)]


class TMSRL:
    def __init__(self, sim_time=120.0, dt=0.2, liar_prob=0.3,
                 spawn_interval=2.5, max_cars_per_dir=10):
        self.sim_time = sim_time
        self.dt = dt
        self.liar_prob = liar_prob
        self.spawn_interval = spawn_interval
        self.max_cars_per_dir = max_cars_per_dir

        self.tm = None
        self.cars = []
        self.spawn_timers = {d: 0.0 for d in DIRECTIONS}
        self.time = 0.0

        # The agent controls phases; we track remaining time for the chosen phase.
        self.phase_remaining = 0.0

        # Delay metrics
        self.wait_started = {}   # car_obj -> time when it began waiting
        self.wait_samples = []   # per-car wait durations

    def reset(self):
        self.tm = TrafficManager()
        # Disable internal greedy switching; agent decides phases.
        self.tm._choose_new_phase = lambda: None
        # Start with a valid phase active (NS straight+right) for 6 s.
        self.tm.phase_index = 0
        self.tm.current_phase = PHASES[0]
        self.tm.phase_timer = 0.0

        self.cars = []
        self.spawn_timers = {d: 0.0 for d in DIRECTIONS}
        self.time = 0.0
        self.phase_remaining = 6.0

        self.wait_started.clear()
        self.wait_samples.clear()
        return self._get_state()

    def step(self, action_idx):
        action = ACTIONS[action_idx]

        # Apply action (set a new phase/duration) or KEEP current
        if action[0] != "KEEP":
            self.tm.phase_index = action[0]
            self.tm.current_phase = PHASES[self.tm.phase_index]
            self.phase_remaining = float(action[1])
            self.tm.phase_timer = 0.0

        # Advance one simulation tick
        self._spawn()
        # Advance manager with cars (front-of-lane release logic needs them)
        self.tm.update(self.dt, self.cars)

        # Update cars and log waits
        for car in list(self.cars):
            prev = car.tms_state
            car.update(self.dt, self.tm, self.cars)

            if car.tms_state == "waiting" and car not in self.wait_started:
                self.wait_started[car] = self.tm.time

            if prev == "waiting" and car.tms_state == "in_intersection":
                start = self.wait_started.pop(car, self.tm.time)
                self.wait_samples.append(max(0.0, self.tm.time - start))

        # Pass tm so crashes free movements (no deadlock)
        crashed = check_collisions(self.cars, self.tm)

        # Housekeeping
        self.cars = [c for c in self.cars if not c.finished]
        self.time += self.dt
        if self.phase_remaining > 0.0:
            self.phase_remaining = max(0.0, self.phase_remaining - self.dt)

        done = (self.time >= self.sim_time) or crashed

        # Reward: minimize delay, mild fairness term, strong crash penalty
        avg_delay = self._avg_delay()
        fairness = self._fairness_penalty()
        reward = -avg_delay - 0.25 * fairness - (50.0 if crashed else 0.0)

        return self._get_state(), reward, done, {"crashed": crashed, "avg_delay": avg_delay}

    def _spawn(self):
        # Temporarily adjust liar probability in the simulator for spawns
        import traffic_sim_2_smart as ts
        old_lp = ts.LIAR_PROBABILITY
        ts.LIAR_PROBABILITY = self.liar_prob

        for d in DIRECTIONS:
            self.spawn_timers[d] += self.dt
            if self.spawn_timers[d] >= self.spawn_interval:
                active = [c for c in self.cars if c.direction == d and not c.finished]
                if len(active) < self.max_cars_per_dir:
                    self.cars.append(spawn_car_for_direction(d))
                self.spawn_timers[d] = 0.0

        ts.LIAR_PROBABILITY = old_lp

    def _avg_delay(self):
        if not self.wait_samples:
            return 0.0
        return float(np.mean(self.wait_samples[-200:]))  # recent average

    def _fairness_penalty(self):
        # Std deviation of queue lengths across movements (smoother queues are better)
        qlens = []
        for d in DIRECTIONS:
            for m in INTENTS:
                q = self.tm.queues[(d, m)]
                qlens.append(len([c for c in q if not c.finished and not c.crashed]))
        if not qlens:
            return 0.0
        arr = np.array(qlens, dtype=float)
        return float(arr.std())

    def _get_state(self):
        # 12 queue lengths + current phase one-hot (4) + remaining time (1)
        qlens = []
        for d in DIRECTIONS:
            for m in INTENTS:
                q = self.tm.queues[(d, m)]
                qlens.append(len([c for c in q if not c.finished and not c.crashed]))
        phase_onehot = [0.0] * 4
        phase_onehot[self.tm.phase_index] = 1.0
        rem = [self.phase_remaining / 8.0]  # normalize by max action duration
        return np.array(qlens + phase_onehot + rem, dtype=np.float32)


if __name__ == "__main__":
    # Quick sanity check (optional)
    env = TMSRL(sim_time=20.0, dt=0.2, liar_prob=0.3)
    s = env.reset()
    fixed_action = 0  # (phase 0, dur 4.0)
    for _ in range(int(env.sim_time / env.dt)):
        s, r, done, info = env.step(fixed_action)
        if done:
            break
    print("Headless check OK. Avg delay:", info.get("avg_delay", None))