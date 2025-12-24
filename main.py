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
# We define constants here so we can change the "rules" of the world
# in one place without breaking the code.

WIDTH, HEIGHT = 800, 800  # Window size in pixels
FPS = 60  # Frames Per Second (Game speed)
LANE_WIDTH = 60  # Width of the road lanes
STOP_DIST = 40  # Distance a car tries to keep from the car ahead
CAR_GAP = 70  # Minimum safety buffer

# --- TIME SETTINGS ---
# To simulate a full 24-hour day in a few minutes, we speed up time.
# 10 real seconds = 1 simulated hour.
SECONDS_PER_SIM_HOUR = 10
START_HOUR = 6  # The simulation clock starts at 06:00 AM

# --- TRAFFIC SCHEDULE ---
# This dictionary controls the flow of traffic.
# Keys = Hour of the day (0-23)
# Values = Spawn probability (0.0 to 1.0) for each lane ID (0-3).
# Higher number = More cars.
DEFAULT_RATE = 0.17

HOURLY_TRAFFIC = {
    # Morning Rush: Make it intense!
    7: {0: 0.15, 1: 0.15, 2: 0.05, 3: 0.05},
    8: {0: 0.15, 1: 0.15, 2: 0.05, 3: 0.05},

    # Evening Rush: Make it intense!
    17: {0: 0.05, 1: 0.05, 2: 0.15, 3: 0.15},
    18: {0: 0.05, 1: 0.05, 2: 0.15, 3: 0.15},
}

# --- COLORS (R, G, B) ---
WHITE = (240, 240, 240)
GRAY = (50, 50, 50)  # Road color
DARK_GRAY = (30, 30, 30)  # Traffic light housing
YELLOW = (255, 200, 0)
RED = (255, 50, 50)
GREEN = (50, 200, 50)
BLACK = (0, 0, 0)
BLUE_UI = (20, 20, 60)  # Background for the stats box
ORANGE_UI = (200, 100, 0)  # Highlight color for the clock


# --- ENUMS (Enumerations) ---
# Enums make code readable. Instead of checking "if state == 0",
# we check "if state == LightState.RED".
class LightState(Enum):
    RED = 0
    RED_YELLOW = 1  # Used in Europe/UK before Green
    GREEN = 2
    GREEN_BLINK = 3  # Warning that Red is coming
    YELLOW = 4


class Direction(Enum):
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3


# ==========================================
#        SECTION 2: LOGIC CLASSES
# ==========================================

class PIDController:
    """
    A mathematical controller used in engineering.
    It calculates how much 'Force' (acceleration) to apply to correct an 'Error' (distance).

    P (Proportional): "I am far behind, speed up."
    I (Integral): "I have been behind for a long time, speed up more."
    D (Derivative): "I am closing the gap too fast, slow down!"
    """

    def __init__(self, kp, ki, kd):
        self.kp = kp  # Tuning parameter for P
        self.ki = ki  # Tuning parameter for I
        self.kd = kd  # Tuning parameter for D
        self.prev_error = 0
        self.integral = 0

    def update(self, error, dt):
        # Calculate P term
        p_term = self.kp * error

        # Calculate I term (accumulate error over time)
        self.integral += error * dt
        i_term = self.ki * self.integral

        # Calculate D term (rate of change)
        derivative = (error - self.prev_error) / dt
        d_term = self.kd * derivative

        # Save error for next frame
        self.prev_error = error

        # Total output = sum of all terms
        return p_term + i_term + d_term


class Vehicle:
    """
    Represents a single car in the simulation.
    Handles its own physics, decision making (AI), and drawing.
    """

    def __init__(self, lane_id):
        self.lane_id = lane_id
        self.direction = Direction(lane_id)

        # Physical properties
        self.rect = pygame.Rect(0, 0, 30, 50)  # The "Hitbox" of the car
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.velocity = 5.0  # Current speed
        self.acceleration = 0.0  # Current force applied

        # Driver personality (some drivers are faster than others)
        self.max_speed = random.uniform(5.5, 7.5)
        self.color = (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200))

        # The Brain: Initialize the PID controller
        self.pid = PIDController(kp=0.5, ki=0.0, kd=8.0)

        # Place the car at the correct starting point
        self._set_spawn_position()

    def _set_spawn_position(self):
        """Calculates where the car appears based on its lane direction."""
        center_x, center_y = WIDTH // 2, HEIGHT // 2
        offset = LANE_WIDTH // 2

        if self.direction == Direction.NORTH:
            self.rect.center = (center_x + offset, HEIGHT + 50)  # Bottom, moving up
        elif self.direction == Direction.SOUTH:
            self.rect.center = (center_x - offset, -50)  # Top, moving down
        elif self.direction == Direction.EAST:
            self.rect.size = (50, 30)  # Rotate dimensions for horizontal
            self.rect.center = (-50, center_y + offset)  # Left, moving right
        elif self.direction == Direction.WEST:
            self.rect.size = (50, 30)
            self.rect.center = (WIDTH + 50, center_y - offset)  # Right, moving left

        # Store high-precision float coordinates for smooth physics
        self.pos_x = float(self.rect.x)
        self.pos_y = float(self.rect.y)

    def is_waiting(self):
        """Returns True if the car is effectively stopped (waiting in queue)."""
        return self.velocity < 0.5

    def get_target_distance(self, lights, vehicles_ahead):
        """
        The 'Vision' of the driver.
        Scans ahead to find the nearest obstacle:
        1. A red light stop line.
        2. The bumper of another car.
        """
        target_dist = 1000.0  # Default: Open road
        stop_line = self.get_stop_line_coord()

        # Get the traffic light controlling MY lane
        my_light = lights[self.lane_id]

        # Calculate distance to the intersection stop line
        dist_to_light = 9999
        if self.direction == Direction.NORTH:
            dist_to_light = self.rect.y - stop_line
        elif self.direction == Direction.SOUTH:
            dist_to_light = stop_line - (self.rect.y + self.rect.height)
        elif self.direction == Direction.EAST:
            dist_to_light = stop_line - (self.rect.x + self.rect.width)
        elif self.direction == Direction.WEST:
            dist_to_light = self.rect.x - stop_line

        # LOGIC: Should I stop for the light?
        # Only stop if light is NOT Green (or Blinking Green)
        # AND if we haven't already crossed the line (dist > -20)
        if my_light.state not in [LightState.GREEN, LightState.GREEN_BLINK]:
            if dist_to_light > -20:
                target_dist = min(target_dist, dist_to_light)

        # LOGIC: Is there a car in front of me?
        for other in vehicles_ahead:
            if other == self: continue  # Don't check against myself

            dist = 9999
            # Complex check: Only care about cars in MY lane and MY direction
            if self.direction == Direction.NORTH and other.direction == self.direction:
                dist = self.rect.y - (other.rect.y + other.rect.height)
            elif self.direction == Direction.SOUTH and other.direction == self.direction:
                dist = other.rect.y - (self.rect.y + self.rect.height)
            elif self.direction == Direction.EAST and other.direction == self.direction:
                dist = other.rect.x - (self.rect.x + self.rect.width)
            elif self.direction == Direction.WEST and other.direction == self.direction:
                dist = self.rect.x - (other.rect.x + other.rect.width)

            # If the car is within 800px ahead, it's an obstacle
            if 0 < dist < 800:
                target_dist = min(target_dist, dist)

        return target_dist

    def get_stop_line_coord(self):
        """Returns the Y or X coordinate of the white stop line."""
        center_x, center_y = WIDTH // 2, HEIGHT // 2
        if self.direction == Direction.NORTH: return center_y + LANE_WIDTH
        if self.direction == Direction.SOUTH: return center_y - LANE_WIDTH
        if self.direction == Direction.EAST: return center_x - LANE_WIDTH
        if self.direction == Direction.WEST: return center_x + LANE_WIDTH
        return 0

    def update(self, lights, vehicles_ahead):
        """Called every frame. Calculates physics and moves the car."""
        dt = 1.0

        # 1. Vision: How far can I go?
        actual_gap = self.get_target_distance(lights, vehicles_ahead)
        desired_gap = STOP_DIST  # I want to stop 40px away

        # 2. Control: Calculate error (Gap - Desired)
        clipped_error = min(actual_gap - desired_gap, 200)

        # 3. Brain: Ask PID controller for adjustment
        adjustment = self.pid.update(clipped_error, dt)

        # 4. Physics: Apply acceleration
        self.acceleration = adjustment
        self.acceleration -= self.velocity * 0.1  # Apply Drag (Air resistance)

        # Update velocity (Speed cannot go below 0 or above max_speed)
        self.velocity += self.acceleration * 0.1
        self.velocity = max(0.0, min(self.velocity, self.max_speed))

        # 5. Movement: Update position based on direction
        if self.direction == Direction.NORTH:
            self.pos_y -= self.velocity
        elif self.direction == Direction.SOUTH:
            self.pos_y += self.velocity
        elif self.direction == Direction.EAST:
            self.pos_x += self.velocity
        elif self.direction == Direction.WEST:
            self.pos_x -= self.velocity

        # Update the hitbox rect
        self.rect.x = int(self.pos_x)
        self.rect.y = int(self.pos_y)

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rect, border_radius=4)


class TrafficLight:
    """
    Represents the physical traffic light box.
    Handles the visuals of the 3 circles (Red/Yellow/Green).
    """

    def __init__(self, lane_id):
        self.lane_id = lane_id
        self.state = LightState.RED
        self.rect = pygame.Rect(0, 0, 0, 0)
        self._set_position()
        self.blink_timer = 0

    def _set_position(self):
        """Places the light box visually next to the correct stop line."""
        cx, cy = WIDTH // 2, HEIGHT // 2
        # Hardcoded offsets to make it look nice on the road shoulder
        if self.lane_id == 0:
            self.rect = pygame.Rect(cx + LANE_WIDTH + 5, cy + LANE_WIDTH, 20, 60)
        elif self.lane_id == 1:
            self.rect = pygame.Rect(cx - LANE_WIDTH - 25, cy - LANE_WIDTH - 60, 20, 60)
        elif self.lane_id == 2:
            self.rect = pygame.Rect(cx - LANE_WIDTH - 60, cy + LANE_WIDTH + 5, 60, 20)
        elif self.lane_id == 3:
            self.rect = pygame.Rect(cx + LANE_WIDTH, cy - LANE_WIDTH - 25, 60, 20)

    def set_state(self, new_state):
        self.state = new_state

    def draw(self, screen):
        # Draw the dark gray box
        pygame.draw.rect(screen, DARK_GRAY, self.rect, border_radius=4)

        # Default colors are "Dimmed" (dark red/yellow/green)
        c_red, c_yellow, c_green = (100, 0, 0), (100, 100, 0), (0, 100, 0)

        # Logic to turn ON the correct light based on State
        if self.state == LightState.RED:
            c_red = RED
        elif self.state == LightState.RED_YELLOW:
            c_red = RED
            c_yellow = YELLOW
        elif self.state == LightState.YELLOW:
            c_yellow = YELLOW
        elif self.state == LightState.GREEN:
            c_green = GREEN
        elif self.state == LightState.GREEN_BLINK:
            # Flashing effect
            self.blink_timer += 1
            if (self.blink_timer // 10) % 2 == 0:
                c_green = GREEN  # On
            else:
                c_green = (0, 100, 0)  # Off

        # Draw the 3 circles
        # Logic differs for Vertical vs Horizontal orientation
        if self.lane_id in [0, 1]:
            cx = self.rect.centerx
            y = self.rect.y
            pygame.draw.circle(screen, c_red, (cx, y + 10), 6)
            pygame.draw.circle(screen, c_yellow, (cx, y + 30), 6)
            pygame.draw.circle(screen, c_green, (cx, y + 50), 6)
        else:
            cy = self.rect.centery
            x = self.rect.x
            pygame.draw.circle(screen, c_red, (x + 10, cy), 6)
            pygame.draw.circle(screen, c_yellow, (x + 30, cy), 6)
            pygame.draw.circle(screen, c_green, (x + 50, cy), 6)


# ==========================================
#        SECTION 3: MAIN SIMULATION
# ==========================================

class Simulation:
    """
    The 'God' class. It controls the Game Loop, global events,
    time management, and holds lists of all objects.
    """

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Advanced Traffic Sim")
        self.clock = pygame.time.Clock()

        # Fonts for text
        self.font = pygame.font.SysFont('Arial', 24, bold=True)
        self.stats_font = pygame.font.SysFont('Consolas', 18)
        self.lane_font = pygame.font.SysFont('Arial', 14, bold=True)

        # Initialize the 4 traffic lights
        self.lights = {0: TrafficLight(0), 1: TrafficLight(1), 2: TrafficLight(2), 3: TrafficLight(3)}

        self.vehicles = []  # List to hold all active cars

        # Traffic Light Timer State
        self.light_timer = 0
        self.light_state_idx = 0

        # Time and Stat tracking
        self.start_ticks = pygame.time.get_ticks()
        self.cars_passed = 0
        self.sim_start_time = START_HOUR * 3600  # Convert start hour to seconds

        # Waiting queue counters
        self.waiting_counts = {0: 0, 1: 0, 2: 0, 3: 0}

        self.cycle = [
            # --- PHASE 1: VERTICAL TRAFFIC MOVES ---
            (60, LightState.RED_YELLOW, LightState.RED),  # Get Ready!
            (300, LightState.GREEN, LightState.RED),  # GO!
            (45, LightState.GREEN_BLINK, LightState.RED),  # Warning blink
            (60, LightState.YELLOW, LightState.RED),  # Stop if safe
            (60, LightState.RED, LightState.RED),  # All Red (Safety Buffer)

            # --- PHASE 2: HORIZONTAL TRAFFIC MOVES ---
            (60, LightState.RED, LightState.RED_YELLOW),
            (300, LightState.RED, LightState.GREEN),
            (45, LightState.RED, LightState.GREEN_BLINK),
            (60, LightState.RED, LightState.YELLOW),
            (60, LightState.RED, LightState.RED)  # All Red
        ]

        # Init Agent
        self.agent = DQNAgent(state_size=7, action_size=2)

    def reset(self):
        self.vehicles = []
        self.light_timer = 0
        self.light_state_idx = 0
        self.cars_passed = 0
        self.waiting_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        self.start_ticks = pygame.time.get_ticks()


    def get_simulated_time(self):
        """Converts real-world run time into 'Game Time' (24h clock)."""
        real_seconds = (pygame.time.get_ticks() - self.start_ticks) / 1000
        sim_seconds_elapsed = real_seconds * (3600 / SECONDS_PER_SIM_HOUR)
        total_sim_seconds = self.sim_start_time + sim_seconds_elapsed

        # Use Modulo (%) to wrap around midnight (86400 seconds in a day)
        day_seconds = total_sim_seconds % 86400
        return int(day_seconds // 3600), int((day_seconds % 3600) // 60)

    def get_current_spawn_rate(self, lane_id, current_hour):
        """Look up the HOURLY_TRAFFIC dictionary to see how heavy traffic is right now."""
        if current_hour in HOURLY_TRAFFIC:
            if lane_id in HOURLY_TRAFFIC[current_hour]:
                return HOURLY_TRAFFIC[current_hour][lane_id]
        return DEFAULT_RATE

    def step_physics_lights(self):
        """
        Handles the natural flow of time (Yellow -> Red -> Green).
        If the light is GREEN, it waits for the Agent.
        If the light is TRANSITIONING, it ignores the Agent and follows the timer.
        """
        self.light_timer += 1

        # Get duration of current state
        duration, v_state, h_state = self.cycle[self.light_state_idx]

        # Apply the states to the lights
        self.lights[0].set_state(v_state)
        self.lights[1].set_state(v_state)
        self.lights[2].set_state(h_state)
        self.lights[3].set_state(h_state)

        # CHECK: Is this a "Holding State" (Green Light)?
        # Index 1 is Vertical Green, Index 6 is Horizontal Green.
        is_green = (self.light_state_idx == 1 or self.light_state_idx == 6)

        # LOGIC:
        # If it's NOT Green, we auto-advance when timer runs out.
        # This ensures Yellow -> Red -> Red-Yellow -> Green happens automatically.
        if not is_green and self.light_timer > duration:
            self.light_timer = 0
            self.light_state_idx = (self.light_state_idx + 1) % len(self.cycle)

    def spawn_vehicles(self):
        """Calculates probabilities and spawns new cars."""
        hour, _ = self.get_simulated_time()
        for lane_id in range(4):
            rate = self.get_current_spawn_rate(lane_id, hour)

            # Random chance check
            if random.random() < rate:
                new_car = Vehicle(lane_id)

                # FAIL FAST: Do not spawn if there is a car blocking the entrance
                safe = True
                for car in self.vehicles:
                    if car.rect.colliderect(new_car.rect):
                        safe = False
                        break

                if safe: self.vehicles.append(new_car)

    def draw_background(self):
        """Draws the static elements (Roads, Lines, Numbers)."""
        self.screen.fill(WHITE)
        cx, cy = WIDTH // 2, HEIGHT // 2

        # Draw Roads
        pygame.draw.rect(self.screen, GRAY, (cx - LANE_WIDTH, 0, LANE_WIDTH * 2, HEIGHT))
        pygame.draw.rect(self.screen, GRAY, (0, cy - LANE_WIDTH, WIDTH, LANE_WIDTH * 2))

        # Draw Stop Lines (White Bars)
        line_w = 4
        pygame.draw.line(self.screen, WHITE, (cx, cy + LANE_WIDTH), (cx + LANE_WIDTH, cy + LANE_WIDTH), line_w)
        pygame.draw.line(self.screen, WHITE, (cx - LANE_WIDTH, cy - LANE_WIDTH), (cx, cy - LANE_WIDTH), line_w)
        pygame.draw.line(self.screen, WHITE, (cx - LANE_WIDTH, cy), (cx - LANE_WIDTH, cy + LANE_WIDTH), line_w)
        pygame.draw.line(self.screen, WHITE, (cx + LANE_WIDTH, cy - LANE_WIDTH), (cx + LANE_WIDTH, cy), line_w)

        # Helper function to draw "Waiting: 5" labels
        def draw_wait_label(lane, x, y, color=BLACK):
            count = self.waiting_counts[lane]
            txt = f"L{lane}: {count} Waiting"

            # Create a small text box with a border
            surf = self.lane_font.render(txt, True, color)
            bg = pygame.Rect(x - 2, y - 2, surf.get_width() + 4, surf.get_height() + 4)
            pygame.draw.rect(self.screen, WHITE, bg)
            pygame.draw.rect(self.screen, BLACK, bg, 1)
            self.screen.blit(surf, (x, y))

        # Position labels near the entrance of each lane
        draw_wait_label(0, cx + 10, HEIGHT - 100)
        draw_wait_label(1, cx - 90, 80)
        draw_wait_label(2, 80, cy + 10)
        draw_wait_label(3, WIDTH - 120, cy - 40)

    def draw_stats(self):
        """Draws the Heads-Up Display (HUD) in the corner."""
        seconds_total = (pygame.time.get_ticks() - self.start_ticks) // 1000
        sim_h, sim_m = self.get_simulated_time()

        # Determine "Rush Hour" text
        vol_str = "Traffic: Normal"
        if sim_h in HOURLY_TRAFFIC:
            rate = HOURLY_TRAFFIC[sim_h].get(0, DEFAULT_RATE)
            if rate > 0.04:
                vol_str = "Traffic: HEAVY"
            elif rate < 0.008:
                vol_str = "Traffic: Quiet"

        # Draw Blue Box
        hud_rect = pygame.Rect(10, 10, 260, 110)
        pygame.draw.rect(self.screen, BLUE_UI, hud_rect, border_radius=8)
        pygame.draw.rect(self.screen, WHITE, hud_rect, 2, border_radius=8)

        # Draw Text info
        self.screen.blit(self.stats_font.render(f"Day Time: {sim_h:02}:{sim_m:02}", True, ORANGE_UI), (25, 20))
        self.screen.blit(self.stats_font.render(vol_str, True, WHITE), (25, 45))

        pygame.draw.line(self.screen, WHITE, (25, 68), (250, 68), 1)  # Separator line

        self.screen.blit(self.stats_font.render(f"Real Time: {seconds_total}s", True, WHITE), (25, 75))
        self.screen.blit(self.stats_font.render(f"Cars Passed: {self.cars_passed}", True, WHITE), (25, 95))

    def get_state(self):
        """
        Returns the 7-dimensional vector:
        [Q_N, Q_S, Q_E, Q_W, Phase_NS, Phase_EW, Timer]
        """
        q = [self.waiting_counts[0] / 10, self.waiting_counts[1] / 10, self.waiting_counts[2] / 10,
             self.waiting_counts[3] / 10]

        if 0 <= self.light_state_idx <= 4:
            phase = [1, 0]
        else:
            phase = [0, 1]

        timer = [self.light_timer / 100]

        return np.array(q + phase + timer)

    def calculate_reward(self):
        """
        Reward is based on minimizing waiting cars.
        We use the negative sum of waiting cars as the reward.
        """
        total_waiting = sum(self.waiting_counts.values())
        reward = -total_waiting
        return reward

    def training_loop(self):
        """Handles the training of the DQN agent."""
        state = self.get_state()
        action = self.agent.act(state)

    def update_lights_agent(self, action):
        """
        The Agent only controls the SWITCH.
        It can only switch the light if it is currently GREEN (Index 1 or 6).
        """
        # If currently Vertical Green (Index 1) AND Agent wants Horizontal (Action 1)
        if self.light_state_idx == 1 and action == 1:
            self.light_timer = 0
            self.light_state_idx = 2  # Jump to Green Blink (Start of transition)

        # If currently Horizontal Green (Index 6) AND Agent wants Vertical (Action 0)
        elif self.light_state_idx == 6 and action == 0:
            self.light_timer = 0
            self.light_state_idx = 7  # Jump to Green Blink (Start of transition)

        # If the light is Yellow/Red, we IGNORE the agent and let physics finish the safety cycle.

    def run_use_brain(self, episodes: int = 1000):
        """The Main Game Loop using the DQN Agent."""
        running = True

        # We start with Turbo Mode OFF so you can see it working
        train_fast = False

        # Max steps per episode (instead of seconds)
        # 5000 frames @ 60 FPS is roughly 80 seconds of simulation
        max_steps = 5000

        best_score = 0

        for episode in range(episodes):
            if not running: break

            self.reset()
            current_step = 0

            # --- INNER LOOP (Step by Step) ---
            # We run until we hit the step limit, regardless of real time
            while current_step < max_steps:

                # 1. Event Handling (Include the Toggle)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        pygame.quit()
                        sys.exit()
                    # TOGGLE TURBO MODE WITH SPACEBAR
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            train_fast = not train_fast
                            print(f"Turbo Mode: {'ON' if train_fast else 'OFF'}")

                # 2. AI Logic (Observe -> Act)
                state = self.get_state()
                action = self.agent.act(state)

                self.update_lights_agent(action)
                self.step_physics_lights()  # Don't forget the physics helper!

                self.spawn_vehicles()

                # 3. Physics (ALWAYS RUNS)
                self.waiting_counts = {0: 0, 1: 0, 2: 0, 3: 0}
                active_vehicles = []
                for v in self.vehicles:
                    v.update(self.lights, self.vehicles)
                    if v.is_waiting():
                        self.waiting_counts[v.lane_id] += 1
                    if -100 < v.rect.x < WIDTH + 100 and -100 < v.rect.y < HEIGHT + 100:
                        active_vehicles.append(v)
                    else:
                        self.cars_passed += 1
                self.vehicles = active_vehicles

                # 4. Learning
                reward = self.calculate_reward()
                # Determine if this is the last step
                done = (current_step == max_steps - 1)

                self.agent.remember(state, action, reward, self.get_state(), done)
                self.agent.replay()

                current_step += 1

                # 5. Graphics & Clock (ONLY IN NORMAL MODE)
                # If train_fast is True, we skip all drawing to save time
                if not train_fast:
                    self.draw_background()
                    for lane_id in self.lights: self.lights[lane_id].draw(self.screen)
                    for car in self.vehicles: car.draw(self.screen)
                    self.draw_stats()
                    pygame.display.flip()
                    self.clock.tick(FPS)  # Limits speed to 60 FPS
                else:
                    # In Turbo mode, we just pump the event loop to keep the window from freezing
                    # But we DON'T wait for the clock.
                    pass

                    # End of Episode Stats
            epsilon = round(self.agent.epsilon, 2)
            print(f"Episode: {episode + 1}/{episodes} | Score: {self.cars_passed} | Epsilon: {epsilon}")

            if self.cars_passed > best_score:
                best_score = self.cars_passed
                self.agent.save("traffic_dqn_best.pth")
                print(f"New Best Score! Model saved. {best_score} cars passed.")

            if self.agent.epsilon > self.agent.epsilon_min:
                self.agent.epsilon *= self.agent.epsilon_decay

            if (episode + 1) % 50 == 0:
                self.agent.save(f"traffic_dqn_episode_{episode + 1}.pth")

        self.agent.save("traffic_dqn_final.pth")
        print("Training completed.")

    def run(self):
        """The Main Game Loop."""
        running = True
        while running:
            # 1. Event Handling (Check if user closed window)
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False

            # 2. Update Game State
            self.update_lights()
            self.spawn_vehicles()

            # Reset wait counts to 0 for this frame
            self.waiting_counts = {0: 0, 1: 0, 2: 0, 3: 0}

            # Update all cars and remove those that left the screen
            active_vehicles = []
            for v in self.vehicles:
                v.update(self.lights, self.vehicles)

                # If car is waiting, increment the counter for that lane
                if v.is_waiting():
                    self.waiting_counts[v.lane_id] += 1

                # Check if car is still inside the map
                if -100 < v.rect.x < WIDTH + 100 and -100 < v.rect.y < HEIGHT + 100:
                    active_vehicles.append(v)
                else:
                    self.cars_passed += 1  # Car successfully finished route
            self.vehicles = active_vehicles

            # 3. Draw Everything
            self.draw_background()

            for lane_id in self.lights:
                self.lights[lane_id].draw(self.screen)

            for car in self.vehicles:
                car.draw(self.screen)

            self.draw_stats()

            # 4. Refresh Display
            pygame.display.flip()
            self.clock.tick(FPS)  # Keep game at 60 FPS

        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    sim = Simulation()

    if __name__ == "__main__":
        sim = Simulation()

        # We prefer the "Best" file, but fall back to "Final" or specific episodes if needed
        brain_file = "traffic_best.pth"

        # If best doesn't exist, try the last run
        if not os.path.exists(brain_file):
            brain_file = "traffic_dqn_final.pth"

        if os.path.exists(brain_file):
            sim.agent.load(brain_file)
            print(f"Loaded brain: {brain_file}")

            # If you want to WATCH the best performance:
            sim.agent.epsilon = sim.agent.epsilon_min

            # If you want to KEEP TRAINING the best model to make it even better:
            # sim.agent.epsilon = 0.5
        else:
            print("No saved brain found. Starting fresh.")

        sim.run_use_brain()
    # sim.run()
    sim.run_use_brain()
