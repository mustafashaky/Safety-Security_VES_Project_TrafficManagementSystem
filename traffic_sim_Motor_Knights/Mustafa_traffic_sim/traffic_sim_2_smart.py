# pygame-ce 2.5.6 (SDL 2.32.10, Python 3.12.7 (base))
import math
import random
import sys
import pygame
# ============================================================
# CONFIG
# ============================================================
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800
FPS = 60
BG_COLOR = (20, 20, 20)
ROAD_COLOR = (60, 60, 60)
LANE_LINE_COLOR = (120, 120, 120)
INTERSECTION_COLOR = (80, 80, 80)
HONEST_COLOR = (0, 150, 255)   # blue
LIAR_COLOR = (220, 60, 60)     # red
CRASH_COLOR = (255, 240, 0)    # yellow
TEXT_COLOR = (230, 230, 230)
CAR_SIZE = 20  # slightly larger to fit letters
CENTER_X = SCREEN_WIDTH // 2
CENTER_Y = SCREEN_HEIGHT // 2
# Intersection modeled roughly as a circle; inside this radius is "intersection"
INTERSECTION_RADIUS = 70
EXIT_MARGIN = 30
# How finely we sample paths (distance between consecutive points)
PATH_STEP = 6.0  # in pixels
# Car speed: pixels per second
CAR_SPEED_PPS = 120.0
# Spawn logic: ensure continuous flow from all directions
SPAWN_INTERVAL = 2.5          # seconds between spawns per direction
MAX_CARS_PER_DIR = 10         # cap to avoid infinite buildup
LIAR_PROBABILITY = 0.2        # some cars lie about intent
# Minimum gap to maintain between cars in the same lane (to avoid rear-end crashes)
FOLLOW_DISTANCE = CAR_SIZE * 1.5
# Minimum time headway between *successive* cars of the same movement
MIN_HEADWAY = 1.5  # seconds
# Directions and intents
DIRECTIONS = ["N", "E", "S", "W"]
INTENTS = ["straight", "right", "left"]
# Lane offset from road center (for right-hand traffic)
LANE_OFFSET = 35
# Traffic signal phases
PHASES = [
    # Phase 0: N-S straight + right
    {("N", "straight"), ("N", "right"),
     ("S", "straight"), ("S", "right")},
    # Phase 1: E-W straight + right
    {("E", "straight"), ("E", "right"),
     ("W", "straight"), ("W", "right")},
    # Phase 2: N-S protected left turns
    {("N", "left"), ("S", "left")},
    # Phase 3: E-W protected left turns
    {("E", "left"), ("W", "left")},
]
PHASE_NAMES = [
    "N-S straight+right",
    "E-W straight+right",
    "N-S left turns",
    "E-W left turns",
]
PHASE_DURATION = 6.0  # seconds each phase before re-optimizing

# ============================================================
# PATH GENERATION (CENTERLINES)
# ============================================================

def distance(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)

def generate_routes():
    """
    Generate centerline routes and intersection entry indices
    for each direction + intent.
    IMPORTANT: We define geometry so that:
      - 'right' = driver's right turn (from the car's perspective)
      - 'left'  = driver's left turn
    That means:
      N right -> W,  N left -> E
      S right -> E,  S left -> W
      E right -> N,  E left -> S
      W right -> S,  W left -> N
    """
    routes = {d: {} for d in DIRECTIONS}
    SPAWN_OFFSET = 100
    ARC_STEPS = 20
    # STRAIGHT PATHS (centerlines)
    # N (top) -> S (bottom)
    points, entry_index = [], None
    x = CENTER_X
    y = -SPAWN_OFFSET
    while y <= SCREEN_HEIGHT + SPAWN_OFFSET:
        if entry_index is None and distance(x, y, CENTER_X, CENTER_Y) <= INTERSECTION_RADIUS:
            entry_index = len(points)
        points.append((x, y))
        y += PATH_STEP
    routes["N"]["straight"] = (points, entry_index)
    # S (bottom) -> N (top)
    points, entry_index = [], None
    x = CENTER_X
    y = SCREEN_HEIGHT + SPAWN_OFFSET
    while y >= -SPAWN_OFFSET:
        if entry_index is None and distance(x, y, CENTER_X, CENTER_Y) <= INTERSECTION_RADIUS:
            entry_index = len(points)
        points.append((x, y))
        y -= PATH_STEP
    routes["S"]["straight"] = (points, entry_index)
    # W (left) -> E (right)
    points, entry_index = [], None
    y = CENTER_Y
    x = -SPAWN_OFFSET
    while x <= SCREEN_WIDTH + SPAWN_OFFSET:
        if entry_index is None and distance(x, y, CENTER_X, CENTER_Y) <= INTERSECTION_RADIUS:
            entry_index = len(points)
        points.append((x, y))
        x += PATH_STEP
    routes["W"]["straight"] = (points, entry_index)
    # E (right) -> W (left)
    points, entry_index = [], None
    y = CENTER_Y
    x = SCREEN_WIDTH + SPAWN_OFFSET
    while x >= -SPAWN_OFFSET:
        if entry_index is None and distance(x, y, CENTER_X, CENTER_Y) <= INTERSECTION_RADIUS:
            entry_index = len(points)
        points.append((x, y))
        x -= PATH_STEP
    routes["E"]["straight"] = (points, entry_index)
    # Use a base radius
    R = INTERSECTION_RADIUS
    ARC_STEPS = 20
    # Helper to build a route: approach -> arc -> exit
    def build_turn_route(start, approach_dir, turn, end):
        """
        start: (x, y) start outside
        approach_dir: 'N','S','E','W' indicating direction of travel toward center
        turn: 'cw' or 'ccw' in world coordinates
        end: destination axis 'N','S','E','W' where we exit along that axis
        """
        points = []
        entry_index = None
        # Approach straight until reaching circle of radius R
        x, y = start
        if approach_dir == "N":   # coming from top, moving down
            while y <= CENTER_Y - R:
                if entry_index is None and distance(x, y, CENTER_X, CENTER_Y) <= INTERSECTION_RADIUS:
                    entry_index = len(points)
                points.append((x, y))
                y += PATH_STEP
        elif approach_dir == "S":  # coming from bottom, moving up
            while y >= CENTER_Y + R:
                if entry_index is None and distance(x, y, CENTER_X, CENTER_Y) <= INTERSECTION_RADIUS:
                    entry_index = len(points)
                points.append((x, y))
                y -= PATH_STEP
        elif approach_dir == "W":  # coming from left, moving right
            while x <= CENTER_X - R:
                if entry_index is None and distance(x, y, CENTER_X, CENTER_Y) <= INTERSECTION_RADIUS:
                    entry_index = len(points)
                points.append((x, y))
                x += PATH_STEP
        elif approach_dir == "E":  # coming from right, moving left
            while x >= CENTER_X + R:
                if entry_index is None and distance(x, y, CENTER_X, CENTER_Y) <= INTERSECTION_RADIUS:
                    entry_index = len(points)
                points.append((x, y))
                x -= PATH_STEP
        # Arc around center
        if approach_dir == "N" and end == "W":  # driver's right from N
            theta_start, theta_end = -math.pi / 2 - (0 if turn == "ccw" else 0), -math.pi if turn == "ccw" else 0
        elif approach_dir == "N" and end == "E":  # driver's left from N
            theta_start, theta_end = -math.pi / 2, 0
        elif approach_dir == "S" and end == "E":  # driver's right from S
            theta_start, theta_end = math.pi / 2, 0
        elif approach_dir == "S" and end == "W":  # driver's left from S
            theta_start, theta_end = math.pi / 2, math.pi
        elif approach_dir == "E" and end == "N":  # driver's right from E
            theta_start, theta_end = 0, -math.pi / 2
        elif approach_dir == "E" and end == "S":  # driver's left from E
            theta_start, theta_end = 0, math.pi / 2
        elif approach_dir == "W" and end == "S":  # driver's right from W
            theta_start, theta_end = math.pi, math.pi / 2
        elif approach_dir == "W" and end == "N":  # driver's left from W
            theta_start, theta_end = math.pi, -math.pi / 2
        else:
            theta_start, theta_end = 0, 0  # shouldn't happen
        # Ensure arc direction consistent
        if theta_start < theta_end:
            step = (theta_end - theta_start) / ARC_STEPS
        else:
            step = (theta_end - theta_start) / ARC_STEPS
        for i in range(ARC_STEPS + 1):
            theta = theta_start + step * i
            ax = CENTER_X + R * math.cos(theta)
            ay = CENTER_Y + R * math.sin(theta)
            if entry_index is None and distance(ax, ay, CENTER_X, CENTER_Y) <= INTERSECTION_RADIUS:
                entry_index = len(points)
            points.append((ax, ay))
        # Exit straight along destination axis
        if end == "E":
            x_start = CENTER_X + R
            y_const = CENTER_Y
            x = x_start
            while x <= SCREEN_WIDTH + SPAWN_OFFSET:
                points.append((x, y_const))
                x += PATH_STEP
        elif end == "W":
            x_start = CENTER_X - R
            y_const = CENTER_Y
            x = x_start
            while x >= -SPAWN_OFFSET:
                points.append((x, y_const))
                x -= PATH_STEP
        elif end == "S":
            x_const = CENTER_X
            y_start = CENTER_Y + R
            y = y_start
            while y <= SCREEN_HEIGHT + SPAWN_OFFSET:
                points.append((x_const, y))
                y += PATH_STEP
        elif end == "N":
            x_const = CENTER_X
            y_start = CENTER_Y - R
            y = y_start
            while y >= -SPAWN_OFFSET:
                points.append((x_const, y))
                y -= PATH_STEP
        return points, entry_index
    # Now define right/left based on DRIVER perspective
    # N: right->W, left->E
    routes["N"]["right"] = build_turn_route((CENTER_X, -SPAWN_OFFSET), "N", "ccw", "W")
    routes["N"]["left"] = build_turn_route((CENTER_X, -SPAWN_OFFSET), "N", "cw", "E")
    # S: right->E, left->W
    routes["S"]["right"] = build_turn_route((CENTER_X, SCREEN_HEIGHT + SPAWN_OFFSET), "S", "ccw", "E")
    routes["S"]["left"] = build_turn_route((CENTER_X, SCREEN_HEIGHT + SPAWN_OFFSET), "S", "cw", "W")
    # E: right->N, left->S
    routes["E"]["right"] = build_turn_route((SCREEN_WIDTH + SPAWN_OFFSET, CENTER_Y), "E", "ccw", "N")
    routes["E"]["left"] = build_turn_route((SCREEN_WIDTH + SPAWN_OFFSET, CENTER_Y), "E", "cw", "S")
    # W: right->S, left->N
    routes["W"]["right"] = build_turn_route((-SPAWN_OFFSET, CENTER_Y), "W", "ccw", "S")
    routes["W"]["left"] = build_turn_route((-SPAWN_OFFSET, CENTER_Y), "W", "cw", "N")
    return routes

ROUTES = generate_routes()

# ============================================================
# TRAFFIC MANAGER WITH SIGNALS
# ============================================================

class TrafficManager:
    """
    Manages signals assuming cars are truthful.
    At most ONE car per (direction, intent) can be in the intersection at a time,
    and successive cars from the same movement must be separated by MIN_HEADWAY.
    """
    def __init__(self):
        self.queues = {(d, m): [] for d in DIRECTIONS for m in INTENTS}
        self.phase_index = 0
        self.phase_timer = 0.0
        self.current_phase = PHASES[0]
        self.active_counts = {(d, m): 0 for d in DIRECTIONS for m in INTENTS}
        self.time = 0.0
        self.last_release_time = {(d, m): -1e9 for d in DIRECTIONS for m in INTENTS}
    def register_waiting(self, car):
        key = (car.direction, car.reported_intent)
        q = self.queues[key]
        if car not in q:
            q.append(car)
        car.can_move = False
        car.tms_state = "waiting"
    def notify_exit(self, car):
        key = (car.direction, car.reported_intent)
        if self.active_counts[key] > 0:
            self.active_counts[key] -= 1
        car.in_intersection = False
        car.tms_state = "passed"
    def update(self, dt):
        self.time += dt
        # Phase timer and switch
        self.phase_timer += dt
        if self.phase_timer >= PHASE_DURATION:
            self._choose_new_phase()
            self.phase_timer = 0.0
        # For each movement in current phase, maybe let one car go
        for key in self.current_phase:
            # Already a car in intersection for this movement?
            if self.active_counts[key] > 0:
                continue
            # Enforce time headway
            if self.time - self.last_release_time[key] < MIN_HEADWAY:
                continue
            q = self.queues[key]
            while q and (q[0].finished or q[0].crashed):
                q.pop(0)
            if q:
                car = q[0]
                if not car.can_move:
                    car.can_move = True
                    car.in_intersection = True
                    car.tms_state = "in_intersection"
                    self.active_counts[key] += 1
                    self.last_release_time[key] = self.time
                    q.pop(0)
    def _choose_new_phase(self):
        best_idx = self.phase_index
        best_score = -1
        for idx, phase in enumerate(PHASES):
            score = 0
            for key in phase:
                q = self.queues[key]
                score += len([c for c in q if not c.finished and not c.crashed])
            if score > best_score:
                best_score = score
                best_idx = idx
        self.phase_index = best_idx
        self.current_phase = PHASES[self.phase_index]
    def get_signal_state(self, direction, intent):
        return "G" if (direction, intent) in self.current_phase else "R"

# ============================================================
# CAR
# ============================================================

class Car:
    def __init__(self, direction, true_intent, reported_intent, liar=False):
        self.direction = direction
        self.true_intent = true_intent
        self.reported_intent = reported_intent
        self.liar = liar
        self.color = LIAR_COLOR if liar else HONEST_COLOR
        self.base_route_points, self.entry_index = ROUTES[direction][true_intent]
        self.route_len = len(self.base_route_points)
        if direction == "N":
            self.lane_shift = (-LANE_OFFSET, 0)
        elif direction == "S":
            self.lane_shift = (LANE_OFFSET, 0)
        elif direction == "W":
            self.lane_shift = (0, LANE_OFFSET)
        elif direction == "E":
            self.lane_shift = (0, -LANE_OFFSET)
        else:
            self.lane_shift = (0, 0)
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
    def _has_safe_gap(self, cand_x, cand_y, cars):
        for other in cars:
            if other is self or other.finished or other.crashed:
                continue
            if other.direction != self.direction:
                continue
            ox, oy = other.pos
            if self.direction == "N":
                if oy <= cand_y:
                    continue
            elif self.direction == "S":
                if oy >= cand_y:
                    continue
            elif self.direction == "W":
                if ox <= cand_x:
                    continue
            elif self.direction == "E":
                if ox >= cand_x:
                    continue
            if self.direction in ("N", "S"):
                if abs(ox - cand_x) > CAR_SIZE:
                    continue
            else:
                if abs(oy - cand_y) > CAR_SIZE:
                    continue
            if abs(ox - cand_x) < FOLLOW_DISTANCE and abs(oy - cand_y) < FOLLOW_DISTANCE:
                return False
        return True
    def update(self, dt, tm: TrafficManager, cars):
        if self.finished or self.crashed:
            return
        if self.in_intersection:
            cx, cy = self.pos
            d = distance(cx, cy, CENTER_X, CENTER_Y)
            if d > INTERSECTION_RADIUS + EXIT_MARGIN and self.route_pos > self.entry_index:
                tm.notify_exit(self)
        if not self.can_move:
            return
        raw_next = self.route_pos + self.speed_idx_per_sec * dt
        if raw_next > self.route_len - 1:
            raw_next = self.route_len - 1
        if (not self.has_queued and
                self.entry_index is not None and
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
        if not self._has_safe_gap(cand_x, cand_y, cars):
            return
        self.route_pos = cand_route_pos
        self.pos = (cand_x, cand_y)
        if (not self.has_queued and
                self.entry_index is not None and
                abs(self.route_pos - self.entry_index) < 1e-3):
            self.has_queued = True
            tm.register_waiting(self)
            return
        if self.route_pos >= self.route_len - 1:
            self.finished = True
            if self.in_intersection:
                tm.notify_exit(self)
            return
    def draw(self, surface, font):
        x, y = self.pos
        rect = pygame.Rect(0, 0, CAR_SIZE, CAR_SIZE)
        rect.center = (int(x), int(y))
        # Border color encodes REPORTED intent
        if self.reported_intent == "straight":
            border_color = (255, 255, 255)
        elif self.reported_intent == "right":
            border_color = (0, 255, 0)
        else:
            border_color = (255, 165, 0)
        if self.crashed:
            fill_color = CRASH_COLOR
        else:
            fill_color = self.color
        pygame.draw.rect(surface, border_color, rect)
        inner = rect.inflate(-4, -4)
        pygame.draw.rect(surface, fill_color, inner)
        letter_map = {"straight": "S", "right": "R", "left": "L"}
        label = letter_map.get(self.true_intent, "?")
        text_surf = font.render(label, True, (0, 0, 0))
        text_rect = text_surf.get_rect(center=rect.center)
        surface.blit(text_surf, text_rect)

# ============================================================
# SPAWNING
# ============================================================

def spawn_car_for_direction(direction):
    true_intent = random.choice(INTENTS)
    liar = random.random() < LIAR_PROBABILITY
    if liar:
        if true_intent != "right":
            reported_intent = "right"
        else:
            reported_intent = "straight"
    else:
        reported_intent = true_intent
    return Car(direction, true_intent, reported_intent, liar=liar)

def spawn_four_right_turns_honest():
    # 4 honest cars, all truly turning right from their perspective
    return [Car(d, "right", "right", liar=False) for d in DIRECTIONS]

def spawn_four_with_liar():
    cars = []
    for d in DIRECTIONS:
        if d == "E":
            cars.append(Car(d, "straight", "right", liar=True))
        else:
            cars.append(Car(d, "right", "right", liar=False))
    return cars

# ============================================================
# COLLISION DETECTION
# ============================================================

def check_collisions(cars):
    crashed = False
    for i in range(len(cars)):
        a = cars[i]
        if a.finished or a.crashed:
            continue
        ax, ay = a.pos
        for j in range(i + 1, len(cars)):
            b = cars[j]
            if b.finished or b.crashed:
                continue
            bx, by = b.pos
            if abs(ax - bx) < CAR_SIZE and abs(ay - by) < CAR_SIZE:
                a.crashed = True
                b.crashed = True
                crashed = True
    return crashed

# ============================================================
# DRAWING
# ============================================================

def draw_roads(surface):
    road_width = 200
    h_rect = pygame.Rect(0, CENTER_Y - road_width // 2, SCREEN_WIDTH, road_width)
    pygame.draw.rect(surface, ROAD_COLOR, h_rect)
    v_rect = pygame.Rect(CENTER_X - road_width // 2, 0, road_width, SCREEN_HEIGHT)
    pygame.draw.rect(surface, ROAD_COLOR, v_rect)
    inter_rect = pygame.Rect(
        CENTER_X - INTERSECTION_RADIUS,
        CENTER_Y - INTERSECTION_RADIUS,
        INTERSECTION_RADIUS * 2,
        INTERSECTION_RADIUS * 2,
    )
    pygame.draw.rect(surface, INTERSECTION_COLOR, inter_rect)
    pygame.draw.line(surface, LANE_LINE_COLOR, (CENTER_X, 0), (CENTER_X, SCREEN_HEIGHT), 2)
    pygame.draw.line(surface, LANE_LINE_COLOR, (0, CENTER_Y), (SCREEN_WIDTH, CENTER_Y), 2)
    pygame.draw.line(surface, LANE_LINE_COLOR, (CENTER_X - LANE_OFFSET, 0), (CENTER_X - LANE_OFFSET, SCREEN_HEIGHT), 1)
    pygame.draw.line(surface, LANE_LINE_COLOR, (CENTER_X + LANE_OFFSET, 0), (CENTER_X + LANE_OFFSET, SCREEN_HEIGHT), 1)
    pygame.draw.line(surface, LANE_LINE_COLOR, (0, CENTER_Y - LANE_OFFSET), (SCREEN_WIDTH, CENTER_Y - LANE_OFFSET), 1)
    pygame.draw.line(surface, LANE_LINE_COLOR, (0, CENTER_Y + LANE_OFFSET), (SCREEN_WIDTH, CENTER_Y + LANE_OFFSET), 1)

def draw_signals(surface, tm: TrafficManager):
    size = 10
    margin = 4
    def color_for_state(state):
        return (0, 200, 0) if state == "G" else (200, 0, 0)
    # North
    x = CENTER_X - size - margin
    y = CENTER_Y - INTERSECTION_RADIUS - 25
    for intent in ["straight", "right", "left"]:
        state = tm.get_signal_state("N", intent)
        rect = pygame.Rect(x, y, size, size)
        pygame.draw.rect(surface, color_for_state(state), rect)
        y += size + 2
    # South
    x = CENTER_X + margin
    y = CENTER_Y + INTERSECTION_RADIUS + 5
    for intent in ["straight", "right", "left"]:
        state = tm.get_signal_state("S", intent)
        rect = pygame.Rect(x, y, size, size)
        pygame.draw.rect(surface, color_for_state(state), rect)
        y += size + 2
    # West
    x = CENTER_X - INTERSECTION_RADIUS - 25
    y = CENTER_Y + margin
    for intent in ["straight", "right", "left"]:
        state = tm.get_signal_state("W", intent)
        rect = pygame.Rect(x, y, size, size)
        pygame.draw.rect(surface, color_for_state(state), rect)
        x += size + 2
    # East
    x = CENTER_X + INTERSECTION_RADIUS + 5
    y = CENTER_Y - size - margin
    for intent in ["straight", "right", "left"]:
        state = tm.get_signal_state("E", intent)
        rect = pygame.Rect(x, y, size, size)
        pygame.draw.rect(surface, color_for_state(state), rect)
        x += size + 2

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
        "Letter on car = TRUE intent (S/R/L).",
        "Box Fill = Outcome: Blue = honest, Red = liar, Yellow = crashed.",
        "Border = reported intent: white=S, green=R, orange=L.",
        "Press '1': 4 honest right-turn cars (should all pass safely).",
        "Press '2': 4 right-reporting cars (E lies -> possible crash).",
    ]
    if crash_happened:
        lines.append("CRASH detected: simulation paused.")
    y = 10
    for line in lines:
        surf = font.render(line, True, TEXT_COLOR)
        surface.blit(surf, (10, y))
        y += surf.get_height() + 2

# ============================================================
# MAIN LOOP
# ============================================================

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Traffic Management System – correct right/left geometry & lying cars")
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
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_1:
                    cars.extend(spawn_four_right_turns_honest())
                elif event.key == pygame.K_2:
                    cars.extend(spawn_four_with_liar())
        if not crash_happened:
            # Spawn continuous traffic
            for d in DIRECTIONS:
                spawn_timers[d] += dt
                if spawn_timers[d] >= SPAWN_INTERVAL:
                    active_dir_cars = [c for c in cars
                                       if c.direction == d and not c.finished and not c.crashed]
                    if len(active_dir_cars) < MAX_CARS_PER_DIR:
                        cars.append(spawn_car_for_direction(d))
                    spawn_timers[d] = 0.0
            tm.update(dt)
            for car in cars:
                car.update(dt, tm, cars)
            if check_collisions(cars):
                crash_happened = True
            cars = [c for c in cars if not c.finished or c.crashed]
        screen.fill(BG_COLOR)
        draw_roads(screen)
        draw_signals(screen, tm)
        for car in cars:
            car.draw(screen, car_font)
        draw_hud(screen, hud_font, cars, tm, crash_happened)
        pygame.display.flip()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()