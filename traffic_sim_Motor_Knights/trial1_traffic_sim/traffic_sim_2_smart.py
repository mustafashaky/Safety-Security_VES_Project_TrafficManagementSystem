# traffic_sim_2_smart.py
# pygame-ce 2.5.6 (SDL 2.32.10, Python 3.12.7 (base))
import math, random, sys, pygame

# ---------------- CONFIG ----------------
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800
FPS = 60
BG_COLOR = (20, 20, 20)
ROAD_COLOR = (60, 60, 60)
LANE_LINE_COLOR = (120, 120, 120)
INTERSECTION_COLOR = (80, 80, 80)
HONEST_COLOR = (0, 150, 255)
LIAR_COLOR = (220, 60, 60)
CRASH_COLOR = (255, 240, 0)
TEXT_COLOR = (230, 230, 230)
CAR_SIZE = 20
CENTER_X = SCREEN_WIDTH // 2
CENTER_Y = SCREEN_HEIGHT // 2
INTERSECTION_RADIUS = 70
EXIT_MARGIN = 30
PATH_STEP = 6.0
CAR_SPEED_PPS = 120.0
SPAWN_INTERVAL = 2.5
MAX_CARS_PER_DIR = 10
LIAR_PROBABILITY = 0.2
FOLLOW_DISTANCE = CAR_SIZE * 1.5
MIN_HEADWAY = 1.5
LIE_PENALTY_SEC = 3.0
DIRECTIONS = ["N", "E", "S", "W"]
INTENTS = ["straight", "right", "left"]
LANE_OFFSET = 35
PHASES = [
    {("N", "straight"), ("N", "right"), ("S", "straight"), ("S", "right")},
    {("E", "straight"), ("E", "right"), ("W", "straight"), ("W", "right")},
    {("N", "left"), ("S", "left")},
    {("E", "left"), ("W", "left")},
]
PHASE_NAMES = [
    "N-S straight+right",
    "E-W straight+right",
    "N-S left turns",
    "E-W left turns",
]
PHASE_DURATION = 6.0

# ---------------- GEOMETRY ----------------
def distance(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)

def generate_routes():
    routes = {d: {} for d in DIRECTIONS}
    SPAWN_OFFSET = 100
    ARC_STEPS = 20

    # Straights
    pts, idx = [], None
    x = CENTER_X; y = -SPAWN_OFFSET
    while y <= SCREEN_HEIGHT + SPAWN_OFFSET:
        if idx is None and distance(x, y, CENTER_X, CENTER_Y) <= INTERSECTION_RADIUS:
            idx = len(pts)
        pts.append((x, y)); y += PATH_STEP
    routes["N"]["straight"] = (pts, idx)

    pts, idx = [], None
    x = CENTER_X; y = SCREEN_HEIGHT + SPAWN_OFFSET
    while y >= -SPAWN_OFFSET:
        if idx is None and distance(x, y, CENTER_X, CENTER_Y) <= INTERSECTION_RADIUS:
            idx = len(pts)
        pts.append((x, y)); y -= PATH_STEP
    routes["S"]["straight"] = (pts, idx)

    pts, idx = [], None
    y = CENTER_Y; x = -SPAWN_OFFSET
    while x <= SCREEN_WIDTH + SPAWN_OFFSET:
        if idx is None and distance(x, y, CENTER_X, CENTER_Y) <= INTERSECTION_RADIUS:
            idx = len(pts)
        pts.append((x, y)); x += PATH_STEP
    routes["W"]["straight"] = (pts, idx)

    pts, idx = [], None
    y = CENTER_Y; x = SCREEN_WIDTH + SPAWN_OFFSET
    while x >= -SPAWN_OFFSET:
        if idx is None and distance(x, y, CENTER_X, CENTER_Y) <= INTERSECTION_RADIUS:
            idx = len(pts)
        pts.append((x, y)); x -= PATH_STEP
    routes["E"]["straight"] = (pts, idx)

    R = INTERSECTION_RADIUS
    def build_turn_route(start, approach, end):
        points, entry_index = [], None
        x, y = start
        if approach == "N":
            while y <= CENTER_Y - R:
                if entry_index is None and distance(x, y, CENTER_X, CENTER_Y) <= INTERSECTION_RADIUS:
                    entry_index = len(points)
                points.append((x, y)); y += PATH_STEP
        elif approach == "S":
            while y >= CENTER_Y + R:
                if entry_index is None and distance(x, y, CENTER_X, CENTER_Y) <= INTERSECTION_RADIUS:
                    entry_index = len(points)
                points.append((x, y)); y -= PATH_STEP
        elif approach == "W":
            while x <= CENTER_X - R:
                if entry_index is None and distance(x, y, CENTER_X, CENTER_Y) <= INTERSECTION_RADIUS:
                    entry_index = len(points)
                points.append((x, y)); x += PATH_STEP
        elif approach == "E":
            while x >= CENTER_X + R:
                if entry_index is None and distance(x, y, CENTER_X, CENTER_Y) <= INTERSECTION_RADIUS:
                    entry_index = len(points)
                points.append((x, y)); x -= PATH_STEP

        if approach == "N" and end == "W": ts, te = -math.pi/2, -math.pi
        elif approach == "N" and end == "E": ts, te = -math.pi/2, 0
        elif approach == "S" and end == "E": ts, te = math.pi/2, 0
        elif approach == "S" and end == "W": ts, te = math.pi/2, math.pi
        elif approach == "E" and end == "N": ts, te = 0, -math.pi/2
        elif approach == "E" and end == "S": ts, te = 0, math.pi/2
        elif approach == "W" and end == "S": ts, te = math.pi, math.pi/2
        elif approach == "W" and end == "N": ts, te = math.pi, -math.pi/2
        else: ts, te = 0, 0

        step = (te - ts) / 20
        for i in range(21):
            th = ts + step * i
            ax = CENTER_X + R * math.cos(th)
            ay = CENTER_Y + R * math.sin(th)
            if entry_index is None and distance(ax, ay, CENTER_X, CENTER_Y) <= INTERSECTION_RADIUS:
                entry_index = len(points)
            points.append((ax, ay))

        if end == "E":
            x_start = CENTER_X + R; y_const = CENTER_Y; x = x_start
            while x <= SCREEN_WIDTH + SPAWN_OFFSET: points.append((x, y_const)); x += PATH_STEP
        elif end == "W":
            x_start = CENTER_X - R; y_const = CENTER_Y; x = x_start
            while x >= -SPAWN_OFFSET: points.append((x, y_const)); x -= PATH_STEP
        elif end == "S":
            x_const = CENTER_X; y_start = CENTER_Y + R; y = y_start
            while y <= SCREEN_HEIGHT + SPAWN_OFFSET: points.append((x_const, y)); y += PATH_STEP
        elif end == "N":
            x_const = CENTER_X; y_start = CENTER_Y - R; y = y_start
            while y >= -SPAWN_OFFSET: points.append((x_const, y)); y -= PATH_STEP
        return points, entry_index

    routes["N"]["right"] = build_turn_route((CENTER_X, -SPAWN_OFFSET), "N", "W")
    routes["N"]["left"]  = build_turn_route((CENTER_X, -SPAWN_OFFSET), "N", "E")
    routes["S"]["right"] = build_turn_route((CENTER_X, SCREEN_HEIGHT + SPAWN_OFFSET), "S", "E")
    routes["S"]["left"]  = build_turn_route((CENTER_X, SCREEN_HEIGHT + SPAWN_OFFSET), "S", "W")
    routes["E"]["right"] = build_turn_route((SCREEN_WIDTH + SPAWN_OFFSET, CENTER_Y), "E", "N")
    routes["E"]["left"]  = build_turn_route((SCREEN_WIDTH + SPAWN_OFFSET, CENTER_Y), "E", "S")
    routes["W"]["right"] = build_turn_route((-SPAWN_OFFSET, CENTER_Y), "W", "S")
    routes["W"]["left"]  = build_turn_route((-SPAWN_OFFSET, CENTER_Y), "W", "N")
    return routes

ROUTES = generate_routes()

# ---------------- SMART INTENT ----------------
def infer_intent_from_route(car):
    end_x, end_y = car.base_route_points[-1]
    dest_axis = ('N' if end_y < CENTER_Y else 'S') if abs(end_x - CENTER_X) < 1e-3 else ('W' if end_x < CENTER_X else 'E')
    mapping = {
        ('N','S'): 'straight', ('N','W'): 'right', ('N','E'): 'left',
        ('S','N'): 'straight', ('S','E'): 'right', ('S','W'): 'left',
        ('E','W'): 'straight', ('E','N'): 'right', ('E','S'): 'left',
        ('W','E'): 'straight', ('W','S'): 'right', ('W','N'): 'left',
    }
    return mapping[(car.direction, dest_axis)]

# ---------------- MANAGER ----------------
class TrafficManager:
    def __init__(self):
        self.queues = {(d, m): [] for d in DIRECTIONS for m in INTENTS}
        self.phase_index = 0
        self.phase_timer = 0.0
        self.current_phase = PHASES[0]
        self.active_counts = {(d, m): 0 for d in DIRECTIONS for m in INTENTS}
        self.time = 0.0
        self.last_release_time = {(d, m): -1e9 for d in DIRECTIONS for m in INTENTS}

    def _is_front_of_lane(self, car, cars):
        cx, cy = car.pos
        for other in cars:
            if other is car or other.finished or other.crashed: continue
            if other.direction != car.direction: continue
            ox, oy = other.pos
            if car.direction == "N" and oy > cy:   return False
            if car.direction == "S" and oy < cy:   return False
            if car.direction == "W" and ox > cx:   return False
            if car.direction == "E" and ox < cx:   return False
        return True

    def register_waiting(self, car):
        car.detected_intent = infer_intent_from_route(car)
        liar_detected = (car.reported_intent != car.detected_intent)
        car.eligible_time = self.time + (LIE_PENALTY_SEC if liar_detected else 0.0)
        car.is_penalized = liar_detected
        key = (car.direction, car.detected_intent)
        if car not in self.queues[key]:
            self.queues[key].append(car)
        car.can_move = False
        car.tms_state = "waiting"

    def notify_exit(self, car):
        key = (car.direction, car.detected_intent or car.reported_intent)
        if self.active_counts[key] > 0:
            self.active_counts[key] -= 1
        car.in_intersection = False
        car.tms_state = "passed"

    def update(self, dt, cars):
        self.time += dt
        self.phase_timer += dt
        if self.phase_timer >= PHASE_DURATION:
            self._choose_new_phase()
            self.phase_timer = 0.0

        for key in self.current_phase:
            if self.active_counts[key] > 0: continue
            if self.time - self.last_release_time[key] < MIN_HEADWAY: continue

            q = self.queues[key]
            while q and (q[0].finished or q[0].crashed):
                q.pop(0)
            if not q: continue

            release_idx = -1
            for i, cand in enumerate(q):
                if cand.finished or cand.crashed: continue
                if self.time < getattr(cand, "eligible_time", 0.0): continue
                if self._is_front_of_lane(cand, cars):
                    release_idx = i
                    break

            if release_idx == -1:  # no eligible front-of-lane car right now
                continue

            car = q.pop(release_idx)
            if not car.can_move:
                car.can_move = True
                car.in_intersection = True
                car.tms_state = "in_intersection"
                self.active_counts[key] += 1
                self.last_release_time[key] = self.time

    def _choose_new_phase(self):
        best_idx, best_score = self.phase_index, -1
        for idx, phase in enumerate(PHASES):
            score = 0
            for key in phase:
                q = self.queues[key]
                score += len([c for c in q if not c.finished and not c.crashed])
            if score > best_score:
                best_score, best_idx = score, idx
        self.phase_index = best_idx
        self.current_phase = PHASES[self.phase_index]

    def get_signal_state(self, direction, intent):
        return "G" if (direction, intent) in self.current_phase else "R"

    def on_crash(self, car):
        key = (car.direction, getattr(car, "detected_intent", car.reported_intent))
        if self.active_counts.get(key, 0) > 0:
            self.active_counts[key] -= 1
        self.last_release_time[key] = self.time

# ---------------- CAR ----------------
class Car:
    def __init__(self, direction, true_intent, reported_intent, liar=False):
        self.direction = direction
        self.true_intent = true_intent
        self.reported_intent = reported_intent
        self.liar = liar
        self.color = LIAR_COLOR if liar else HONEST_COLOR
        self.base_route_points, self.entry_index = ROUTES[direction][true_intent]
        self.route_len = len(self.base_route_points)
        if direction == "N":   self.lane_shift = (-LANE_OFFSET, 0)
        elif direction == "S": self.lane_shift = (LANE_OFFSET, 0)
        elif direction == "W": self.lane_shift = (0, LANE_OFFSET)
        elif direction == "E": self.lane_shift = (0, -LANE_OFFSET)
        else:                  self.lane_shift = (0, 0)
        self.route_pos = 0.0
        self.speed_idx_per_sec = CAR_SPEED_PPS / PATH_STEP
        x0, y0 = self.base_route_points[0]
        self.pos = (x0 + self.lane_shift[0], y0 + self.lane_shift[1])
        self.can_move = True
        self.in_intersection = False
        self.has_queued = False
        self.tms_state = "approaching"
        self.finished = False
        self.crashed = False
        self.detected_intent = None
        self.eligible_time = 0.0
        self.is_penalized = False

    def _has_safe_gap(self, cand_x, cand_y, cars):
        for other in cars:
            if other is self or other.finished or other.crashed: continue
            if other.direction != self.direction: continue
            ox, oy = other.pos
            if self.direction == "N":
                if oy <= cand_y: continue
            elif self.direction == "S":
                if oy >= cand_y: continue
            elif self.direction == "W":
                if ox <= cand_x: continue
            elif self.direction == "E":
                if ox >= cand_x: continue
            if self.direction in ("N", "S"):
                if abs(ox - cand_x) > CAR_SIZE: continue
            else:
                if abs(oy - cand_y) > CAR_SIZE: continue
            if abs(ox - cand_x) < FOLLOW_DISTANCE and abs(oy - cand_y) < FOLLOW_DISTANCE:
                return False
        return True

    def update(self, dt, tm: TrafficManager, cars):
        if self.finished or self.crashed: return
        if self.in_intersection:
            cx, cy = self.pos
            d = distance(cx, cy, CENTER_X, CENTER_Y)
            if d > INTERSECTION_RADIUS + EXIT_MARGIN and self.route_pos > self.entry_index:
                tm.notify_exit(self)
        if not self.can_move: return

        raw_next = self.route_pos + self.speed_idx_per_sec * dt
        raw_next = min(raw_next, self.route_len - 1)

        if (not self.has_queued and self.entry_index is not None and
            self.route_pos < self.entry_index <= raw_next):
            cand_route_pos = float(self.entry_index)
        else:
            cand_route_pos = raw_next

        i0 = int(cand_route_pos)
        i1 = min(i0 + 1, self.route_len - 1)
        alpha = cand_route_pos - i0
        x0, y0 = self.base_route_points[i0]
        x1, y1 = self.base_route_points[i1]
        cand_x = x0 * (1 - alpha) + x1 * alpha + self.lane_shift[0]
        cand_y = y0 * (1 - alpha) + y1 * alpha + self.lane_shift[1]

        if not self._has_safe_gap(cand_x, cand_y, cars): return

        self.route_pos = cand_route_pos
        self.pos = (cand_x, cand_y)

        if (not self.has_queued and self.entry_index is not None and
            abs(self.route_pos - self.entry_index) < 1e-3):
            self.has_queued = True
            tm.register_waiting(self)
            return

        if self.route_pos >= self.route_len - 1:
            self.finished = True
            if self.in_intersection: tm.notify_exit(self)
            return

    def draw(self, surface, font):
        x, y = self.pos
        rect = pygame.Rect(0, 0, CAR_SIZE, CAR_SIZE); rect.center = (int(x), int(y))
        border_color = (255, 255, 255) if self.reported_intent == "straight" else ((0, 255, 0) if self.reported_intent == "right" else (255, 165, 0))
        fill_color = CRASH_COLOR if self.crashed else self.color
        pygame.draw.rect(surface, border_color, rect)
        inner = rect.inflate(-4, -4); pygame.draw.rect(surface, fill_color, inner)
        letter_map = {"straight": "S", "right": "R", "left": "L"}
        label = letter_map.get(self.true_intent, "?")
        text_surf = font.render(label, True, (0, 0, 0)); surface.blit(text_surf, text_surf.get_rect(center=rect.center))
        if self.tms_state == "waiting" and self.is_penalized and not self.crashed:
            pygame.draw.circle(surface, (255, 0, 0), (int(x), int(y - CAR_SIZE)), 3)

# ---------------- SPAWN ----------------
def spawn_car_for_direction(direction):
    true_intent = random.choice(INTENTS)
    liar = random.random() < LIAR_PROBABILITY
    reported_intent = ("right" if true_intent != "right" else "straight") if liar else true_intent
    return Car(direction, true_intent, reported_intent, liar=liar)

def spawn_four_right_turns_honest():
    return [Car(d, "right", "right", liar=False) for d in DIRECTIONS]

def spawn_four_with_liar():
    cars = []
    for d in DIRECTIONS:
        if d == "E": cars.append(Car(d, "straight", "right", liar=True))
        else:        cars.append(Car(d, "right", "right", liar=False))
    return cars

# ---------------- COLLISIONS ----------------
def check_collisions(cars, tm=None):
    crashed = False
    for i in range(len(cars)):
        a = cars[i]
        if a.finished or a.crashed: continue
        ax, ay = a.pos
        for j in range(i + 1, len(cars)):
            b = cars[j]
            if b.finished or b.crashed: continue
            bx, by = b.pos
            if abs(ax - bx) < CAR_SIZE and abs(ay - by) < CAR_SIZE:
                a.crashed = True; b.crashed = True; crashed = True
                if tm is not None:
                    tm.on_crash(a); tm.on_crash(b)
    return crashed

# ---------------- DRAW ----------------
def draw_roads(surface):
    road_width = 200
    h_rect = pygame.Rect(0, CENTER_Y - road_width // 2, SCREEN_WIDTH, road_width)
    pygame.draw.rect(surface, ROAD_COLOR, h_rect)
    v_rect = pygame.Rect(CENTER_X - road_width // 2, 0, road_width, SCREEN_HEIGHT)
    pygame.draw.rect(surface, ROAD_COLOR, v_rect)
    inter_rect = pygame.Rect(CENTER_X - INTERSECTION_RADIUS, CENTER_Y - INTERSECTION_RADIUS, INTERSECTION_RADIUS * 2, INTERSECTION_RADIUS * 2)
    pygame.draw.rect(surface, INTERSECTION_COLOR, inter_rect)
    pygame.draw.line(surface, LANE_LINE_COLOR, (CENTER_X, 0), (CENTER_X, SCREEN_HEIGHT), 2)
    pygame.draw.line(surface, LANE_LINE_COLOR, (0, CENTER_Y), (SCREEN_WIDTH, CENTER_Y), 2)
    pygame.draw.line(surface, LANE_LINE_COLOR, (CENTER_X - LANE_OFFSET, 0), (CENTER_X - LANE_OFFSET, SCREEN_HEIGHT), 1)
    pygame.draw.line(surface, LANE_LINE_COLOR, (CENTER_X + LANE_OFFSET, 0), (CENTER_X + LANE_OFFSET, SCREEN_HEIGHT), 1)
    pygame.draw.line(surface, LANE_LINE_COLOR, (0, CENTER_Y - LANE_OFFSET), (SCREEN_WIDTH, CENTER_Y - LANE_OFFSET), 1)
    pygame.draw.line(surface, LANE_LINE_COLOR, (0, CENTER_Y + LANE_OFFSET), (SCREEN_WIDTH, CENTER_Y + LANE_OFFSET), 1)

def draw_signals(surface, tm: TrafficManager):
    size = 10; margin = 4
    def color_for_state(s): return (0, 200, 0) if s == "G" else (200, 0, 0)
    x = CENTER_X - size - margin; y = CENTER_Y - INTERSECTION_RADIUS - 25
    for intent in ["straight", "right", "left"]:
        pygame.draw.rect(surface, color_for_state(tm.get_signal_state("N", intent)), pygame.Rect(x, y, size, size)); y += size + 2
    x = CENTER_X + margin; y = CENTER_Y + INTERSECTION_RADIUS + 5
    for intent in ["straight", "right", "left"]:
        pygame.draw.rect(surface, color_for_state(tm.get_signal_state("S", intent)), pygame.Rect(x, y, size, size)); y += size + 2
    x = CENTER_X - INTERSECTION_RADIUS - 25; y = CENTER_Y + margin
    for intent in ["straight", "right", "left"]:
        pygame.draw.rect(surface, color_for_state(tm.get_signal_state("W", intent)), pygame.Rect(x, y, size, size)); x += size + 2
    x = CENTER_X + INTERSECTION_RADIUS + 5; y = CENTER_Y - size - margin
    for intent in ["straight", "right", "left"]:
        pygame.draw.rect(surface, color_for_state(tm.get_signal_state("E", intent)), pygame.Rect(x, y, size, size)); x += size + 2

def draw_hud(surface, font, cars, tm: TrafficManager, crash_happened):
    honest = sum(1 for c in cars if not c.liar)
    liars = sum(1 for c in cars if c.liar)
    crashes = sum(1 for c in cars if c.crashed)
    lines = [
        f"Honest cars (blue): {honest}",
        f"Lying cars (red): {liars}",
        f"Crashes: {crashes}",
        f"Phase: {PHASE_NAMES[tm.phase_index]}",
        f"Headway per movement: {MIN_HEADWAY:.1f}s",
        f"Liar cooldown: {LIE_PENALTY_SEC:.1f}s",
        "Letter on car: TRUE intent (S/R/L).",
        "Box Fill: Blue = honest / Red = liar / Yellow = crashed.",
        "Border: Reported intent (white = S, green = R, orange = L).",
        "Press '1': spawn 4 honest right turns.",
        "Press '2': spawn 3 honest rights + 1 East liar.",
    ]
    if crash_happened: lines.append("CRASH detected: simulation paused.")
    y = 10
    for line in lines:
        surf = font.render(line, True, TEXT_COLOR); surface.blit(surf, (10, y)); y += surf.get_height() + 2

# ---------------- MAIN ----------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Smart intersection: intent inference + liar penalty + front-of-lane release")
    clock = pygame.time.Clock()
    hud_font = pygame.font.SysFont("consolas", 14)
    car_font = pygame.font.SysFont("consolas", 12, bold=True)
    tm = TrafficManager()
    cars = []
    crash_happened = False
    spawn_timers = {d: 0.0 for d in DIRECTIONS}

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: running = False
                elif event.key == pygame.K_1: cars.extend(spawn_four_right_turns_honest())
                elif event.key == pygame.K_2: cars.extend(spawn_four_with_liar())

        if not crash_happened:
            for d in DIRECTIONS:
                spawn_timers[d] += dt
                if spawn_timers[d] >= SPAWN_INTERVAL:
                    active_dir_cars = [c for c in cars if c.direction == d and not c.finished and not c.crashed]
                    if len(active_dir_cars) < MAX_CARS_PER_DIR:
                        cars.append(spawn_car_for_direction(d))
                    spawn_timers[d] = 0.0

            tm.update(dt, cars)

            for car in cars:
                car.update(dt, tm, cars)

            if check_collisions(cars, tm):
                crash_happened = True

            cars = [c for c in cars if not c.finished or c.crashed]

        screen.fill(BG_COLOR)
        draw_roads(screen); draw_signals(screen, tm)
        for car in cars: car.draw(screen, car_font)
        draw_hud(screen, hud_font, cars, tm, crash_happened)
        pygame.display.flip()

    pygame.quit(); sys.exit()

if __name__ == "__main__":
    main()