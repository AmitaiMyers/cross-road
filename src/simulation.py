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
                self.agent.epsilon *= 0.998
