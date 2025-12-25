import math
import os.path
import numpy as np
import pygame
import sys
import random
from enum import Enum
from agent import DQNAgent

# ==========================================
#        SECTION 1: CONFIGURATION
# ==========================================
WIDTH, HEIGHT = 1000, 1000  # Bigger window for bigger roads
FPS = 60
LANE_WIDTH = 40  # Slightly narrower lanes to fit 2 per side
TURN_RADIUS = LANE_WIDTH * 0.75
STOP_DIST = 40
SECONDS_PER_SIM_HOUR = 10
START_HOUR = 6

# --- TRAFFIC SCHEDULE ---
DEFAULT_RATE = 0.20

# Vehicles spawn from 8 approaches (4 directions × 2 lanes: Straight/Left)
SPAWN_LANE_INDICES = (0, 1, 2, 3, 4, 5, 6, 7)

# --- TRAFFIC SCHEDULE ---
# Keys: 0=N_Str, 1=N_Left, 2=S_Str, 3=S_Left, 4=E_Str, 5=E_Left, 6=W_Str, 7=W_Left
# Rates: 0.01 = Light, 0.03 = Medium, 0.06+ = Heavy Jam

HOURLY_TRAFFIC = {
    # --- MORNING RUSH (North/South Dominance) ---
    # Commuters driving straight into the city.
    7: {0: 0.08, 1: 0.02, 2: 0.08, 3: 0.02, 4: 0.01, 5: 0.01, 6: 0.01, 7: 0.01},
    8: {0: 0.10, 1: 0.03, 2: 0.10, 3: 0.03, 4: 0.01, 5: 0.01, 6: 0.01, 7: 0.01},
    9: {0: 0.06, 1: 0.02, 2: 0.06, 3: 0.02, 4: 0.02, 5: 0.02, 6: 0.02, 7: 0.02},

    # --- THE "LEFT TURN" CHALLENGE ---
    # Low traffic, but high probability of Left Turns.
    # Can the agent learn to trigger Phases 2 & 4 specifically?
    10: {0: 0.01, 1: 0.06, 2: 0.01, 3: 0.06, 4: 0.01, 5: 0.06, 6: 0.01, 7: 0.06},
    11: {0: 0.02, 1: 0.02, 2: 0.02, 3: 0.02, 4: 0.02, 5: 0.02, 6: 0.02, 7: 0.02},

    # --- LUNCH RUSH (Gridlock) ---
    # High pressure from ALL sides.
    12: {0: 0.06, 1: 0.06, 2: 0.06, 3: 0.06, 4: 0.06, 5: 0.06, 6: 0.06, 7: 0.06},
    13: {0: 0.04, 1: 0.04, 2: 0.04, 3: 0.04, 4: 0.04, 5: 0.04, 6: 0.04, 7: 0.04},

    # --- AFTERNOON LULL ---
    14: {0: 0.02, 1: 0.02, 2: 0.02, 3: 0.02, 4: 0.02, 5: 0.02, 6: 0.02, 7: 0.02},

    # --- EVENING RUSH (East/West Dominance) ---
    # Commuters leaving the city.
    16: {0: 0.01, 1: 0.01, 2: 0.01, 3: 0.01, 4: 0.08, 5: 0.03, 6: 0.08, 7: 0.03},
    17: {0: 0.01, 1: 0.01, 2: 0.01, 3: 0.01, 4: 0.12, 5: 0.04, 6: 0.12, 7: 0.04},
    18: {0: 0.02, 1: 0.02, 2: 0.02, 3: 0.02, 4: 0.06, 5: 0.02, 6: 0.06, 7: 0.02},
}

# --- COLORS ---
WHITE = (240, 240, 240)
GRAY = (50, 50, 50)
DARK_GRAY = (30, 30, 30)
YELLOW = (255, 200, 0)
RED = (255, 50, 50)
GREEN = (50, 200, 50)
BLACK = (0, 0, 0)
BLUE_UI = (20, 20, 60)
ORANGE_UI = (200, 100, 0)
CYAN = (0, 200, 200)  # Left turn lane color hint


class LightState(Enum):
    RED = 0
    RED_YELLOW = 1
    GREEN = 2
    GREEN_BLINK = 3
    YELLOW = 4


class Direction(Enum):
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3


class Turn(Enum):
    STRAIGHT = 0  # Forward or Right
    LEFT = 1  # Dedicated Left Turn


# ==========================================
#        SECTION 2: LOGIC CLASSES
# ==========================================

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.prev_error = 0
        self.integral = 0

    def update(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        return (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)


class Vehicle:
    def __init__(self, lane_index):
        # 8 Separate Lanes:
        # 0=N_Str, 1=N_Left, 2=S_Str, 3=S_Left, 4=E_Str, 5=E_Left, 6=W_Str, 7=W_Left
        self.lane_index = lane_index

        # Determine Direction (0-3) and Turn Intent based on the 8-way index
        direction_id = lane_index // 2  # 0,0, 1,1, 2,2, 3,3
        self.direction = Direction(direction_id)

        # Even = Straight, Odd = Left
        self.turn_intent = Turn.LEFT if (lane_index % 2 == 1) else Turn.STRAIGHT

        self.rect = pygame.Rect(0, 0, 25, 45)
        self.pos_x, self.pos_y = 0.0, 0.0
        self.velocity = 0.0
        self.acceleration = 0.0
        self.max_speed = random.uniform(5.5, 7.5)

        # Colors: Left turners are blue-ish, Straight are random
        self.color = (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200))
        if self.turn_intent == Turn.LEFT:
            self.color = (100, 100, 255)  # Distinct Blue for Left Turners

        self.pid = PIDController(kp=0.5, ki=0.0, kd=8.0)
        self.turning_complete = False
        self._set_spawn_position()

        # --- Left-turn arc animation state ---
        self.is_turning = False
        self.turn_theta = 0.0
        self.turn_theta_end = 0.0
        self.turn_center = (0.0, 0.0)
        self.turn_radius = float(TURN_RADIUS)
        self.turn_from_direction = self.direction
        self.turn_to_direction = self.direction
        self.turn_to_lane_index = self.lane_index


    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rect, border_radius=4)

        # Optional: show a small yellow dot for left-turn intent
        if self.turn_intent == Turn.LEFT:
            pygame.draw.circle(screen, (255, 255, 0), self.rect.center, 4)

    def _set_spawn_position(self):
        cx, cy = WIDTH // 2, HEIGHT // 2
        offset_straight = LANE_WIDTH * 1.5
        offset_left = LANE_WIDTH * 0.5

        # Offset depends on intent
        my_offset = offset_left if self.turn_intent == Turn.LEFT else offset_straight

        # IMPORTANT: set size first, then set center (pygame Rect keeps top-left when resizing)
        if self.direction in [Direction.NORTH, Direction.SOUTH]:
            self.rect.size = (25, 45)
        else:
            self.rect.size = (45, 25)

        half_w = self.rect.width // 2
        half_h = self.rect.height // 2

        # Spawn just inside the culling bounds (-50 / +50) so cars are not deleted immediately.
        if self.direction == Direction.NORTH:
            # From bottom going up
            self.rect.center = (cx + my_offset, HEIGHT + half_h - 1)
        elif self.direction == Direction.SOUTH:
            # From top going down
            self.rect.center = (cx - my_offset, -half_h + 1)
        elif self.direction == Direction.EAST:
            # From left going right
            self.rect.center = (-half_w + 1, cy + my_offset)
        elif self.direction == Direction.WEST:
            # From right going left
            self.rect.center = (WIDTH + half_w - 1, cy - my_offset)

        self.pos_x = float(self.rect.x)
        self.pos_y = float(self.rect.y)

    def is_waiting(self):
        return self.velocity < 0.5

    def get_stop_line_coord(self):
        cx, cy = WIDTH // 2, HEIGHT // 2
        box_size = LANE_WIDTH * 4
        if self.direction == Direction.NORTH:
            return cy + box_size / 2
        if self.direction == Direction.SOUTH:
            return cy - box_size / 2
        if self.direction == Direction.EAST:
            return cx - box_size / 2
        if self.direction == Direction.WEST:
            return cx + box_size / 2
        return 0

    def get_target_distance(self, lights, vehicles_ahead):
        light_id = self.direction.value
        light_obj = lights[light_id]
        my_signal_state = light_obj.state_left if self.turn_intent == Turn.LEFT else light_obj.state_straight

        stop_line = self.get_stop_line_coord()
        target_dist = 1000.0

        dist_to_light = 9999
        if self.direction == Direction.NORTH:
            dist_to_light = self.rect.y - stop_line
        elif self.direction == Direction.SOUTH:
            dist_to_light = stop_line - (self.rect.y + self.rect.height)
        elif self.direction == Direction.EAST:
            dist_to_light = stop_line - (self.rect.x + self.rect.width)
        elif self.direction == Direction.WEST:
            dist_to_light = self.rect.x - stop_line

        if my_signal_state not in [LightState.GREEN, LightState.GREEN_BLINK]:
            if dist_to_light > -10:
                target_dist = min(target_dist, dist_to_light)

        for other in vehicles_ahead:
            if other == self:
                continue
            if other.lane_index == self.lane_index:
                dist = 9999
                if self.direction == Direction.NORTH:
                    dist = self.rect.y - (other.rect.y + other.rect.height)
                elif self.direction == Direction.SOUTH:
                    dist = other.rect.y - (self.rect.y + self.rect.height)
                elif self.direction == Direction.EAST:
                    dist = other.rect.x - (self.rect.x + self.rect.width)
                elif self.direction == Direction.WEST:
                    dist = self.rect.x - (other.rect.x + other.rect.width)

                if 0 < dist < 800:
                    target_dist = min(target_dist, dist)

        return target_dist

    def update(self, lights, vehicles_ahead):
        dt = 1.0
        actual_gap = self.get_target_distance(lights, vehicles_ahead)
        desired_gap = STOP_DIST
        clipped_error = min(actual_gap - desired_gap, 200)
        adjustment = self.pid.update(clipped_error, dt)
        self.acceleration = adjustment - (self.velocity * 0.1)
        self.velocity += self.acceleration * 0.1
        self.velocity = max(0.0, min(self.velocity, self.max_speed))

        # If currently turning, keep following the turn arc
        if self.is_turning:
            self._advance_left_turn()
            return

        # Start a left turn only when we reached the turn trigger geometry
        if self.turn_intent == Turn.LEFT and not self.turning_complete:
            if self._should_begin_left_turn():
                self._begin_left_turn()
                return

        # Otherwise, normal forward movement
        self._move_forward()
        self.rect.x = int(self.pos_x)
        self.rect.y = int(self.pos_y)

    def _move_forward(self):
        if self.direction == Direction.NORTH:
            self.pos_y -= self.velocity
        elif self.direction == Direction.SOUTH:
            self.pos_y += self.velocity
        elif self.direction == Direction.EAST:
            self.pos_x += self.velocity
        elif self.direction == Direction.WEST:
            self.pos_x -= self.velocity

    def _set_rect_orientation(self, direction: Direction) -> None:
        center = self.rect.center
        if direction in [Direction.NORTH, Direction.SOUTH]:
            self.rect.size = (25, 45)
        else:
            self.rect.size = (45, 25)
        self.rect.center = center
        self.pos_x = float(self.rect.x)
        self.pos_y = float(self.rect.y)

    def _get_center_float(self) -> tuple[float, float]:
        return (self.pos_x + self.rect.width / 2.0, self.pos_y + self.rect.height / 2.0)

    def _set_center_float(self, x: float, y: float) -> None:
        self.rect.center = (int(x), int(y))
        self.pos_x = float(self.rect.x)
        self.pos_y = float(self.rect.y)

    def _left_turn_destination(self) -> tuple[Direction, int]:
        # Left turn mapping:
        # NORTH(up) -> WEST(left)
        # SOUTH(down) -> EAST(right)
        # EAST(right) -> NORTH(up)
        # WEST(left) -> SOUTH(down)
        if self.direction == Direction.NORTH:
            to_dir = Direction.WEST
        elif self.direction == Direction.SOUTH:
            to_dir = Direction.EAST
        elif self.direction == Direction.EAST:
            to_dir = Direction.NORTH
        else:
            to_dir = Direction.SOUTH

        # Destination "straight" lane index is always even: dir.value * 2
        to_lane_index = to_dir.value * 2
        return to_dir, to_lane_index

    def _should_begin_left_turn(self) -> bool:
        # Trigger point based on lane geometry so the arc lands exactly in the destination straight lane.
        cx, cy = WIDTH // 2, HEIGHT // 2
        offset_left = LANE_WIDTH * 0.5
        offset_straight = LANE_WIDTH * 1.5
        r = float(TURN_RADIUS)

        center_x, center_y = self._get_center_float()

        if self.direction == Direction.NORTH:
            # Approach x is right of centerline
            x_lane = cx + offset_left
            y_dest = cy - offset_straight  # Westbound straight lane (y above centerline)
            y_turn_start = y_dest - r
            return center_y <= y_turn_start and abs(center_x - x_lane) < LANE_WIDTH

        if self.direction == Direction.SOUTH:
            x_lane = cx - offset_left
            y_dest = cy + offset_straight  # Eastbound straight lane (y below centerline)
            y_turn_start = y_dest + r
            return center_y >= y_turn_start and abs(center_x - x_lane) < LANE_WIDTH

        if self.direction == Direction.EAST:
            y_lane = cy + offset_left
            x_dest = cx + offset_straight  # Northbound straight lane (x right of centerline)
            x_turn_start = x_dest + r
            return center_x >= x_turn_start and abs(center_y - y_lane) < LANE_WIDTH

        # WEST
        y_lane = cy - offset_left
        x_dest = cx - offset_straight  # Southbound straight lane (x left of centerline)
        x_turn_start = x_dest - r
        return center_x <= x_turn_start and abs(center_y - y_lane) < LANE_WIDTH

    def _begin_left_turn(self) -> None:
        cx, cy = WIDTH // 2, HEIGHT // 2
        offset_left = LANE_WIDTH * 0.5
        offset_straight = LANE_WIDTH * 1.5
        r = float(TURN_RADIUS)

        self.turn_from_direction = self.direction
        self.turn_to_direction, self.turn_to_lane_index = self._left_turn_destination()

        # Arc parameters are chosen so:
        # - start tangent matches approach direction
        # - end tangent matches destination direction
        # - end point lands on destination straight lane centerline

        if self.direction == Direction.NORTH:
            x_lane = cx + offset_left
            y_dest = cy - offset_straight
            center = (x_lane + r, y_dest - r)
            theta_start = math.pi
            theta_end = math.pi / 2.0

        elif self.direction == Direction.SOUTH:
            x_lane = cx - offset_left
            y_dest = cy + offset_straight
            center = (x_lane - r, y_dest + r)
            theta_start = 0.0
            theta_end = -math.pi / 2.0

        elif self.direction == Direction.EAST:
            y_lane = cy + offset_left
            x_dest = cx + offset_straight
            center = (x_dest + r, y_lane + r)
            theta_start = -math.pi / 2.0
            theta_end = -math.pi

        else:  # WEST
            y_lane = cy - offset_left
            x_dest = cx - offset_straight
            center = (x_dest - r, y_lane - r)
            theta_start = math.pi / 2.0
            theta_end = 0.0

        self.turn_center = (float(center[0]), float(center[1]))
        self.turn_radius = r
        self.turn_theta = float(theta_start)
        self.turn_theta_end = float(theta_end)
        self.is_turning = True

        # Snap to exact start point of the arc for clean visuals
        start_x = self.turn_center[0] + self.turn_radius * math.cos(self.turn_theta)
        start_y = self.turn_center[1] + self.turn_radius * math.sin(self.turn_theta)
        self._set_center_float(start_x, start_y)

    def _advance_left_turn(self) -> None:
        # Move along the arc with angular step proportional to speed (arc-length ~= velocity per tick)
        # dtheta = v / r
        if self.turn_radius <= 0.0:
            return

        step = float(self.velocity) / float(self.turn_radius)
        if step <= 0.0:
            return

        # Direction of theta movement
        sign = 1.0 if self.turn_theta_end > self.turn_theta else -1.0
        next_theta = self.turn_theta + sign * step

        # Clamp to end
        if sign > 0.0 and next_theta > self.turn_theta_end:
            next_theta = self.turn_theta_end
        if sign < 0.0 and next_theta < self.turn_theta_end:
            next_theta = self.turn_theta_end

        self.turn_theta = next_theta

        x = self.turn_center[0] + self.turn_radius * math.cos(self.turn_theta)
        y = self.turn_center[1] + self.turn_radius * math.sin(self.turn_theta)
        self._set_center_float(x, y)

        # Swap orientation halfway through the turn (best possible with axis-aligned Rect)
        total = abs(self.turn_theta_end - (self.turn_theta_end - (self.turn_theta_end - self.turn_theta)))  # no-op; kept explicit
        # Use normalized progress by comparing remaining distance
        denom = abs(self.turn_theta_end - self.turn_theta) + abs(self.turn_theta - self.turn_theta_end)
        # simpler: based on theta interpolation fraction
        # We'll compute using start/end snapshot:
        # (store start in local not possible) => approximate with midpoint check:
        mid = (self.turn_theta_end + (self.turn_theta_end - (self.turn_theta_end - self.turn_theta_end)))  # no-op
        # Instead: swap when close to half arc: use angle proximity to average.
        avg = (self.turn_theta_end + self.turn_theta) / 2.0

        # Practical: if direction changes axis, swap once we're closer to end than start
        if abs(self.turn_theta_end - self.turn_theta) < abs(self.turn_theta - (self.turn_theta_end + (self.turn_theta_end - self.turn_theta_end))):
            self._set_rect_orientation(self.turn_to_direction)
        else:
            self._set_rect_orientation(self.turn_from_direction)

        if self.turn_theta == self.turn_theta_end:
            # Finish: lock to destination lane + behavior
            self.is_turning = False
            self.turning_complete = True
            self.direction = self.turn_to_direction
            self.turn_intent = Turn.STRAIGHT
            self.lane_index = self.turn_to_lane_index
            self._set_rect_orientation(self.direction)


class TrafficLight:
    def __init__(self, lane_id):
        self.lane_id = lane_id
        self.state_straight = LightState.RED
        self.state_left = LightState.RED
        self.rect = pygame.Rect(0, 0, 0, 0)
        self._set_position()

    def _set_position(self):
        cx, cy = WIDTH // 2, HEIGHT // 2
        offset = LANE_WIDTH * 2 + 10
        if self.lane_id == 0:
            self.rect = pygame.Rect(cx + offset, cy + offset, 40, 80)
        elif self.lane_id == 1:
            self.rect = pygame.Rect(cx - offset - 40, cy - offset - 80, 40, 80)
        elif self.lane_id == 2:
            self.rect = pygame.Rect(cx - offset - 80, cy + offset, 80, 40)
        elif self.lane_id == 3:
            self.rect = pygame.Rect(cx + offset, cy - offset - 40, 80, 40)

    def set_state(self, straight, left):
        self.state_straight = straight
        self.state_left = left

    def draw(self, screen):
        pygame.draw.rect(screen, DARK_GRAY, self.rect, border_radius=4)

        def draw_head(x, y, state, vertical):
            r, y_c, g = (50, 0, 0), (50, 50, 0), (0, 50, 0)
            if state == LightState.RED:
                r = RED
            elif state == LightState.YELLOW:
                y_c = YELLOW
            elif state == LightState.GREEN:
                g = GREEN
            elif state == LightState.RED_YELLOW:
                r, y_c = RED, YELLOW

            if vertical:
                pygame.draw.circle(screen, r, (x, y), 5)
                pygame.draw.circle(screen, y_c, (x, y + 12), 5)
                pygame.draw.circle(screen, g, (x, y + 24), 5)
            else:
                pygame.draw.circle(screen, r, (x, y), 5)
                pygame.draw.circle(screen, y_c, (x + 12, y), 5)
                pygame.draw.circle(screen, g, (x + 24, y), 5)

        if self.lane_id in [0, 1]:
            draw_head(self.rect.centerx - 10, self.rect.y + 10, self.state_left, True)
            draw_head(self.rect.centerx + 10, self.rect.y + 10, self.state_straight, True)
        else:
            draw_head(self.rect.x + 10, self.rect.centery - 10, self.state_left, False)
            draw_head(self.rect.x + 10, self.rect.centery + 10, self.state_straight, False)


class Simulation:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Phase 3: Multi-Lane Intersection")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 24, bold=True)
        self.lights = {0: TrafficLight(0), 1: TrafficLight(1), 2: TrafficLight(2), 3: TrafficLight(3)}
        self.vehicles = []
        self.light_timer = 0
        self.light_state_idx = 0
        self.start_ticks = pygame.time.get_ticks()
        self.cars_passed = 0
        self.min_switch_time = 100

        # Format: (Duration, NS_Str, NS_Left, EW_Str, EW_Left)
        self.cycle = [
            (60, LightState.RED_YELLOW, LightState.RED, LightState.RED, LightState.RED),
            (300, LightState.GREEN, LightState.RED, LightState.RED, LightState.RED),
            (60, LightState.YELLOW, LightState.RED, LightState.RED, LightState.RED),
            (60, LightState.RED, LightState.RED, LightState.RED, LightState.RED),

            (60, LightState.RED, LightState.RED_YELLOW, LightState.RED, LightState.RED),
            (200, LightState.RED, LightState.GREEN, LightState.RED, LightState.RED),
            (60, LightState.RED, LightState.YELLOW, LightState.RED, LightState.RED),
            (60, LightState.RED, LightState.RED, LightState.RED, LightState.RED),

            (60, LightState.RED, LightState.RED, LightState.RED_YELLOW, LightState.RED),
            (300, LightState.RED, LightState.RED, LightState.GREEN, LightState.RED),
            (60, LightState.RED, LightState.RED, LightState.YELLOW, LightState.RED),
            (60, LightState.RED, LightState.RED, LightState.RED, LightState.RED),

            (60, LightState.RED, LightState.RED, LightState.RED, LightState.RED_YELLOW),
            (200, LightState.RED, LightState.RED, LightState.RED, LightState.GREEN),
            (60, LightState.RED, LightState.RED, LightState.RED, LightState.YELLOW),
            (60, LightState.RED, LightState.RED, LightState.RED, LightState.RED),
        ]

        self.waiting_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        self.agent = DQNAgent(state_size=11, action_size=4)

    def reset(self):
        self.vehicles = []
        self.light_timer = 0
        self.light_state_idx = 0
        self.cars_passed = 0
        self.waiting_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        self.start_ticks = pygame.time.get_ticks()

    def step_physics_lights(self):
        self.light_timer += 1
        duration, ns_s, ns_l, ew_s, ew_l = self.cycle[self.light_state_idx]

        self.lights[0].set_state(ns_s, ns_l)
        self.lights[1].set_state(ns_s, ns_l)
        self.lights[2].set_state(ew_s, ew_l)
        self.lights[3].set_state(ew_s, ew_l)

        is_green = self.light_state_idx in [1, 5, 9, 13]

        if not is_green and self.light_timer > duration:
            self.light_timer = 0
            self.light_state_idx = (self.light_state_idx + 1) % len(self.cycle)

    def update_lights_agent(self, action):
        if self.light_timer < self.min_switch_time:
            return
        if action == 0:
            if self.light_state_idx != 0 and self.light_state_idx != 1:
                self.light_timer = 0
                self.light_state_idx = 0
        elif action == 1:
            if self.light_state_idx != 4 and self.light_state_idx != 5:
                self.light_timer = 0
                self.light_state_idx = 4
        elif action == 2:
            if self.light_state_idx != 8 and self.light_state_idx != 9:
                self.light_timer = 0
                self.light_state_idx = 8
        elif action == 3:
            if self.light_state_idx != 12 and self.light_state_idx != 13:
                self.light_timer = 0
                self.light_state_idx = 12
        else:
            print("Invalid action or state for phase change.", file=sys.stderr)

    def spawn_vehicles(self):
        hour = (pygame.time.get_ticks() - self.start_ticks) // 1000 // (3600 // SECONDS_PER_SIM_HOUR) + START_HOUR
        current_schedule = HOURLY_TRAFFIC.get(hour % 24, {})

        # LOOP across all 8 lane options (4 directions × 2 lanes)
        for i in SPAWN_LANE_INDICES:
            rate = current_schedule.get(i, DEFAULT_RATE)

            if random.random() < rate:
                new_car = Vehicle(i)

                safe = True
                for car in self.vehicles:
                    if car.lane_index == i and car.rect.colliderect(new_car.rect):
                        safe = False
                        break

                if safe:
                    self.vehicles.append(new_car)

    def get_state(self):
        queues = [0] * 8
        for v in self.vehicles:
            if v.is_waiting():
                queues[v.lane_index] += 1

        state = [q / 10.0 for q in queues]

        idx = self.light_state_idx
        if 0 <= idx <= 3:
            phase = [0, 0]
        elif 4 <= idx <= 7:
            phase = [0, 1]
        elif 8 <= idx <= 11:
            phase = [1, 0]
        else:
            phase = [1, 1]

        timer = [self.light_timer / 100.0]
        return np.array(state + phase + timer)

    def calculate_reward(self):
        waiting = 0
        for v in self.vehicles:
            if v.is_waiting():
                waiting += 1
        return -waiting

    def draw_background(self):
        self.screen.fill(WHITE)
        cx, cy = WIDTH // 2, HEIGHT // 2
        hw = LANE_WIDTH * 2

        pygame.draw.rect(self.screen, GRAY, (cx - hw, 0, hw * 2, HEIGHT))
        pygame.draw.rect(self.screen, GRAY, (0, cy - hw, WIDTH, hw * 2))

        pygame.draw.line(self.screen, YELLOW, (cx, 0), (cx, HEIGHT), 2)
        pygame.draw.line(self.screen, YELLOW, (0, cy), (WIDTH, cy), 2)

        pygame.draw.line(self.screen, WHITE, (cx - LANE_WIDTH, 0), (cx - LANE_WIDTH, HEIGHT), 1)
        pygame.draw.line(self.screen, WHITE, (cx + LANE_WIDTH, 0), (cx + LANE_WIDTH, HEIGHT), 1)
        pygame.draw.line(self.screen, WHITE, (0, cy - LANE_WIDTH), (WIDTH, cy - LANE_WIDTH), 1)
        pygame.draw.line(self.screen, WHITE, (0, cy + LANE_WIDTH), (WIDTH, cy + LANE_WIDTH), 1)

    def run_use_brain(self, episodes: int = 1000):
        running, train_fast, best_score = True, False, 0
        for episode in range(episodes):
            if not running:
                break
            self.reset()
            current_step, max_steps = 0, 5000

            while current_step < max_steps:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                        train_fast = not train_fast

                state = self.get_state()
                action = self.agent.act(state)
                self.update_lights_agent(action)
                self.step_physics_lights()
                self.spawn_vehicles()

                active = []
                for v in self.vehicles:
                    v.update(self.lights, self.vehicles)
                    if -50 < v.rect.x < WIDTH + 50 and -50 < v.rect.y < HEIGHT + 50:
                        active.append(v)
                    else:
                        self.cars_passed += 1
                self.vehicles = active

                done = (current_step == max_steps - 1)
                self.agent.remember(state, action, self.calculate_reward(), self.get_state(), done)
                self.agent.replay()
                current_step += 1

                if not train_fast:
                    self.draw_background()

                    phase_names = ["NS Straight", "NS Left", "EW Straight", "EW Left"]
                    p_idx = 0
                    if 4 <= self.light_state_idx <= 7:
                        p_idx = 1
                    elif 8 <= self.light_state_idx <= 11:
                        p_idx = 2
                    elif 12 <= self.light_state_idx <= 15:
                        p_idx = 3

                    info_text = [
                        f"Phase: {phase_names[p_idx]}",
                        f"Timer: {self.light_timer}",
                        f"Total Cars: {self.cars_passed}",
                        f"Epsilon: {round(self.agent.epsilon, 2)}"
                    ]

                    for i, line in enumerate(info_text):
                        surf = self.font.render(line, True, BLACK)
                        self.screen.blit(surf, (10, 10 + i * 30))

                    for l in self.lights.values():
                        l.draw(self.screen)
                    for c in self.vehicles:
                        c.draw(self.screen)
                    pygame.display.flip()
                    self.clock.tick(FPS)

            print(f"Ep {episode + 1}: Score {self.cars_passed} | Eps {round(self.agent.epsilon, 2)}")
            if self.cars_passed > best_score:
                best_score = self.cars_passed
                self.agent.save("traffic_best_phase3.pth")
            if self.agent.epsilon > self.agent.epsilon_min:
                self.agent.epsilon *= 0.995


if __name__ == "__main__":
    sim = Simulation()
    brain_file = "traffic_best_phase3.pth"

    if os.path.exists(brain_file):
        print(f"Found brain file: {brain_file}")
        try:
            sim.agent.load(brain_file)
            print("Brain loaded successfully! Resuming training...")
            sim.agent.epsilon = 0.2
        except Exception as e:
            print(f"⚠️ Brain Mismatch detected: {e}")
            print("Starting a FRESH brain for Phase 3 structure (11 inputs).")
    else:
        print("No save file found. Starting fresh.")

    sim.run_use_brain()
