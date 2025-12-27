import math
import random

import pygame

from src.configuration import TURN_RADIUS, HEIGHT, WIDTH, LANE_WIDTH, STOP_DIST
from src.pid_controller import PIDController
from src.structs_enum import Direction, Turn, LightState


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
