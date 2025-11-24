# rl_env_traffic_sim_3_smart.py
# Gym-like wrapper around your intersection (headless; no pygame window)

import numpy as np
import random

from traffic_sim_3_smart import (
    TrafficManager, DIRECTIONS, INTENTS, PHASES,
    spawn_car_for_direction, check_collisions,
    LIAR_PROBABILITY
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

        # phase control owned by the env (we override the manager’s auto-switching)
        self.phase_remaining = 0.0

        # for delay metrics
        self.wait_started = {}   # car_id -> time
        self.wait_samples = []   # per-car wait times

    def reset(self):
        self.tm = TrafficManager()
        # disable auto-greedy phase changes; the agent will choose phases
        self.tm._choose_new_phase = lambda: None
        self.tm.phase_timer = 0.0

        self.cars = []
        self.spawn_timers = {d: 0.0 for d in DIRECTIONS}
        self.time = 0.0
        self.phase_remaining = 0.0
        self.wait_started.clear()
        self.wait_samples.clear()
        return self._get_state()

    def step(self, action_idx):
        action = ACTIONS[action_idx]

        # Apply action: either KEEP or set a new phase and duration
        if action[0] != "KEEP":
            self.tm.phase_index = action[0]
            self.tm.current_phase = PHASES[self.tm.phase_index]
            self.phase_remaining = action[1]
            self.tm.phase_timer = 0.0

        # Run one simulation tick of length dt
        self._spawn()
        # keep manager’s internal timer from forcing a change
        self.tm.phase_timer = 0.0
        self.tm.update(self.dt)

        # Update cars and record waiting-to-release events
        for car in list(self.cars):
            prev = car.tms_state
            car.update(self.dt, self.tm, self.cars)
            # record when a car first starts waiting at the stop line
            if car.tms_state == "waiting" and car not in self.wait_started:
                self.wait_started[car] = self.tm.time
            # when car transitions from waiting to in_intersection, log its wait
            if prev == "waiting" and car.tms_state == "in_intersection":
                start = self.wait_started.pop(car, self.tm.time)
                self.wait_samples.append(max(0.0, self.tm.time - start))

        crashed = check_collisions(self.cars)

        # remove finished cars (keep crashed visible if you want)
        self.cars = [c for c in self.cars if not c.finished]

        self.time += self.dt
        if self.phase_remaining > 0:
            self.phase_remaining = max(0.0, self.phase_remaining - self.dt)

        done = (self.time >= self.sim_time) or crashed

        # Reward: minimize average queueing delay; penalize crash; mild fairness penalty
        avg_delay = self._avg_delay()
        fairness = self._fairness_penalty()
        reward = -avg_delay - 0.25 * fairness - (50.0 if crashed else 0.0)

        return self._get_state(), reward, done, {"crashed": crashed, "avg_delay": avg_delay}

    def _spawn(self):
        # Temporarily adjust liar probability for spawns
        import traffic_sim_3_smart as ts
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
        # Std dev of queue lengths across movements
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
        phase_onehot = [0.0]*4
        phase_onehot[self.tm.phase_index] = 1.0
        rem = [self.phase_remaining / 8.0]  # normalize by max duration
        return np.array(qlens + phase_onehot + rem, dtype=np.float32)