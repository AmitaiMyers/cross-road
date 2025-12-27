import os
import random
import sys
import numpy as np
import pygame
from src.agent import DQNAgent
from src.configuration import WIDTH, HEIGHT, HOURLY_TRAFFIC, SPAWN_LANE_INDICES, DEFAULT_RATE, START_HOUR, \
    SECONDS_PER_SIM_HOUR, WHITE, LANE_WIDTH, GRAY, YELLOW, FPS, BLACK
from src.structs_enum import LightState
from src.traffic_light import TrafficLight
from src.vehicle import Vehicle


class Simulation:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Phase 4: 8-Phase Super Intersection")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 24, bold=True)

        # 4 Lights: 0=N, 1=S, 2=E, 3=W
        self.lights = {0: TrafficLight(0), 1: TrafficLight(1), 2: TrafficLight(2), 3: TrafficLight(3)}
        self.vehicles = []
        self.light_timer = 0
        self.light_state_idx = 0
        self.start_ticks = pygame.time.get_ticks()
        self.cars_passed = 0
        self.min_switch_time = 120  # ~2 seconds min green time

        # --- THE 8 PHASES ---
        # Format: (Duration, N_Str, N_L, S_Str, S_L, E_Str, E_L, W_Str, W_L)
        # R=Red, G=Green, Y=Yellow, RY=RedYellow
        R = LightState.RED
        G = LightState.GREEN
        Y = LightState.YELLOW
        RY = LightState.RED_YELLOW

        self.cycle = [
            # --- PHASE 0: NS STRAIGHT (Combined) --- [Action 0]
            (60, RY, R, RY, R, R, R, R, R),  # 0: Prep
            (300, G, R, G, R, R, R, R, R),  # 1: GREEN
            (60, Y, R, Y, R, R, R, R, R),  # 2: Yellow
            (60, R, R, R, R, R, R, R, R),  # 3: All Red

            # --- PHASE 1: EW STRAIGHT (Combined) --- [Action 1]
            (60, R, R, R, R, RY, R, RY, R),  # 4: Prep
            (300, R, R, R, R, G, R, G, R),  # 5: GREEN
            (60, R, R, R, R, Y, R, Y, R),  # 6: Yellow
            (60, R, R, R, R, R, R, R, R),  # 7: All Red

            # --- PHASE 2: NS LEFT (Combined) --- [Action 2]
            (60, R, RY, R, RY, R, R, R, R),  # 8: Prep
            (300, R, G, R, G, R, R, R, R),  # 9: GREEN
            (60, R, Y, R, Y, R, R, R, R),  # 10: Yellow
            (60, R, R, R, R, R, R, R, R),  # 11: All Red

            # --- PHASE 3: EW LEFT (Combined) --- [Action 3]
            (60, R, R, R, R, R, RY, R, RY),  # 12: Prep
            (300, R, R, R, R, R, G, R, G),  # 13: GREEN
            (60, R, R, R, R, R, Y, R, Y),  # 14: Yellow
            (60, R, R, R, R, R, R, R, R),  # 15: All Red

            # --- PHASE 4: NORTH FULL (Straight + Left) --- [Action 4]
            (60, RY, RY, R, R, R, R, R, R),  # 16: Prep
            (300, G, G, R, R, R, R, R, R),  # 17: GREEN
            (60, Y, Y, R, R, R, R, R, R),  # 18: Yellow
            (60, R, R, R, R, R, R, R, R),  # 19: All Red

            # --- PHASE 5: SOUTH FULL (Straight + Left) --- [Action 5]
            (60, R, R, RY, RY, R, R, R, R),  # 20: Prep
            (300, R, R, G, G, R, R, R, R),  # 21: GREEN
            (60, R, R, Y, Y, R, R, R, R),  # 22: Yellow
            (60, R, R, R, R, R, R, R, R),  # 23: All Red

            # --- PHASE 6: EAST FULL (Straight + Left) --- [Action 6]
            (60, R, R, R, R, RY, RY, R, R),  # 24: Prep
            (300, R, R, R, R, G, G, R, R),  # 25: GREEN
            (60, R, R, R, R, Y, Y, R, R),  # 26: Yellow
            (60, R, R, R, R, R, R, R, R),  # 27: All Red

            # --- PHASE 7: WEST FULL (Straight + Left) --- [Action 7]
            (60, R, R, R, R, R, R, RY, RY),  # 28: Prep
            (300, R, R, R, R, R, R, G, G),  # 29: GREEN
            (60, R, R, R, R, R, R, Y, Y),  # 30: Yellow
            (60, R, R, R, R, R, R, R, R),  # 31: All Red
        ]

        # Initialize Brain with 12 Inputs (8 queues + 3 phase + 1 timer) and 8 Actions
        self.agent = DQNAgent(state_size=12, action_size=8)

    def reset(self):
        self.vehicles = []
        self.light_timer = 0
        self.light_state_idx = 0
        self.cars_passed = 0
        self.start_ticks = pygame.time.get_ticks()

    def step_physics_lights(self):
        self.light_timer += 1
        # Unpack the 9 values for the current frame
        data = self.cycle[self.light_state_idx]
        duration = data[0]

        # Apply states to lights (N, S, E, W)
        self.lights[0].set_state(data[1], data[2])  # North
        self.lights[1].set_state(data[3], data[4])  # South
        self.lights[2].set_state(data[5], data[6])  # East
        self.lights[3].set_state(data[7], data[8])  # West

        # The Green indices are the second in each block of 4
        green_indices = [1, 5, 9, 13, 17, 21, 25, 29]
        is_green = self.light_state_idx in green_indices

        if not is_green and self.light_timer > duration:
            self.light_timer = 0
            self.light_state_idx = (self.light_state_idx + 1) % len(self.cycle)

    def update_lights_agent(self, action):
        if self.light_timer < self.min_switch_time:
            return

        # Map Action 0-7 to their "Prep" index in the cycle
        target_idx = -1

        # Phase 0-3 (The Standard Splits)
        if action == 0 and self.light_state_idx not in [0, 1]:
            target_idx = 0
        elif action == 1 and self.light_state_idx not in [4, 5]:
            target_idx = 4
        elif action == 2 and self.light_state_idx not in [8, 9]:
            target_idx = 8
        elif action == 3 and self.light_state_idx not in [12, 13]:
            target_idx = 12

        # Phase 4-7 (The New Full Directions)
        elif action == 4 and self.light_state_idx not in [16, 17]:
            target_idx = 16  # North Full
        elif action == 5 and self.light_state_idx not in [20, 21]:
            target_idx = 20  # South Full
        elif action == 6 and self.light_state_idx not in [24, 25]:
            target_idx = 24  # East Full
        elif action == 7 and self.light_state_idx not in [28, 29]:
            target_idx = 28  # West Full

        if target_idx != -1:
            self.light_timer = 0
            self.light_state_idx = target_idx

    def spawn_vehicles(self):
        hour = (pygame.time.get_ticks() - self.start_ticks) // 1000 // (3600 // SECONDS_PER_SIM_HOUR) + START_HOUR
        current_schedule = HOURLY_TRAFFIC.get(hour % 24, {})

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
        # 1. Queues (8 Inputs)
        queues = [0] * 8
        for v in self.vehicles:
            if v.is_waiting():
                queues[v.lane_index] += 1
        state = [q / 10.0 for q in queues]

        # 2. Phase Encoding (3 Inputs)
        # We map the 8 Phases to binary [0,0,0] -> [1,1,1]
        idx = self.light_state_idx
        phase = [0, 0, 0]  # Default

        # Determine which block of 4 we are in
        block = idx // 4

        # Convert block number (0-7) to binary list
        # 0 -> 000, 1 -> 001, ... 7 -> 111
        phase[0] = 1 if (block & 4) else 0
        phase[1] = 1 if (block & 2) else 0
        phase[2] = 1 if (block & 1) else 0

        # 3. Timer (1 Input)
        timer = [self.light_timer / 100.0]

        # Total = 8 + 3 + 1 = 12 Inputs
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

    def run_use_brain(self, episodes=1000):
        running, train_fast, best_score = True, False, 0
        for episode in range(episodes):
            if not running: break
            self.reset()
            step, max_steps = 0, 5000

            while step < max_steps:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT: running = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE: train_fast = not train_fast

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

                done = (step == max_steps - 1)
                self.agent.remember(state, action, self.calculate_reward(), self.get_state(), done)
                self.agent.replay()
                step += 1

                if not train_fast:
                    self.draw_background()
                    for l in self.lights.values(): l.draw(self.screen)
                    for c in self.vehicles: c.draw(self.screen)

                    # HUD Update
                    block = self.light_state_idx // 4
                    p_names = ["NS Straight", "EW Straight", "NS Left", "EW Left", "NORTH Full", "SOUTH Full",
                               "EAST Full", "WEST Full"]
                    p_name = p_names[block] if block < 8 else "Unknown"

                    info = [f"Phase: {p_name}", f"Epsilon: {self.agent.epsilon:.2f}", f"Cars: {self.cars_passed}"]
                    for i, t in enumerate(info):
                        self.screen.blit(self.font.render(t, True, BLACK), (10, 10 + i * 30))

                    pygame.display.flip()
                    self.clock.tick(FPS)

            print(f"Ep {episode}: Score {self.cars_passed} Epsilon {self.agent.epsilon:.3f}")
            if self.cars_passed > best_score:
                best_score = self.cars_passed
                self.agent.save("traffic_best_8phase.pth")
            if self.agent.epsilon > self.agent.epsilon_min: self.agent.epsilon *= 0.998


if __name__ == "__main__":
    sim = Simulation()
    # Check for new 8-phase brain file
    if os.path.exists("traffic_best_8phase.pth"):
        try:
            sim.agent.load("traffic_best_8phase.pth")
            print("Loaded Brain.")
            sim.agent.epsilon = 0.8
        except:
            print("Brain mismatch, starting fresh.")
    sim.run_use_brain()