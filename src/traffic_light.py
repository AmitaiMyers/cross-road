import pygame


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
