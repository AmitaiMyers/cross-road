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

# Keys: 0=N_Str, 1=N_Left, 2=S_Str, 3=S_Left, 4=E_Str, 5=E_Left, 6=W_Str, 7=W_Left
HOURLY_TRAFFIC = {
    # 06:00 - early but already building (heavier than before)
    6: {0: 0.06, 1: 0.04, 2: 0.06, 3: 0.04, 4: 0.06, 5: 0.04, 6: 0.06, 7: 0.04},

    # 07:00 - MORNING RUSH: extreme NORTH only, zero east/west (your requirement)
    7: {0: 0.35, 1: 0.28, 2: 0.00, 3: 0.00, 4: 0.00, 5: 0.00, 6: 0.00, 7: 0.00},

    # 08:00 - MORNING RUSH peak: even more north, tiny south appears, still no east/west
    8: {0: 0.42, 1: 0.32, 2: 0.03, 3: 0.03, 4: 0.00, 5: 0.00, 6: 0.00, 7: 0.00},

    # 09:00 - still heavy, east/west start to wake up
    9: {0: 0.26, 1: 0.18, 2: 0.12, 3: 0.08, 4: 0.08, 5: 0.05, 6: 0.08, 7: 0.05},

    # 10:00 - LEFT-TURN CHALLENGE: heavy left turn demand everywhere
    10: {0: 0.08, 1: 0.22, 2: 0.08, 3: 0.22, 4: 0.08, 5: 0.22, 6: 0.08, 7: 0.22},

    # 11:00 - heavy balanced
    11: {0: 0.16, 1: 0.12, 2: 0.16, 3: 0.12, 4: 0.16, 5: 0.12, 6: 0.16, 7: 0.12},

    # 12:00 - LUNCH GRIDLOCK: very heavy all sides
    12: {0: 0.26, 1: 0.20, 2: 0.26, 3: 0.20, 4: 0.26, 5: 0.20, 6: 0.26, 7: 0.20},

    # 13:00 - still very busy
    13: {0: 0.20, 1: 0.16, 2: 0.20, 3: 0.16, 4: 0.20, 5: 0.16, 6: 0.20, 7: 0.16},

    # 14:00 - afternoon "lull" but still medium-heavy
    14: {0: 0.12, 1: 0.08, 2: 0.12, 3: 0.08, 4: 0.12, 5: 0.08, 6: 0.12, 7: 0.08},

    # 15:00 - school pickup: heavy + left turns noticeably higher
    15: {0: 0.14, 1: 0.20, 2: 0.14, 3: 0.20, 4: 0.16, 5: 0.20, 6: 0.16, 7: 0.20},

    # 16:00 - evening build: east/west surge starts
    16: {0: 0.10, 1: 0.08, 2: 0.10, 3: 0.08, 4: 0.28, 5: 0.18, 6: 0.28, 7: 0.18},

    # 17:00 - EVENING RUSH PEAK: extremely heavy east/west + meaningful north/south
    17: {0: 0.16, 1: 0.12, 2: 0.16, 3: 0.12, 4: 0.40, 5: 0.22, 6: 0.40, 7: 0.22},

    # 18:00 - evening rush wave 2: still heavy, more left turns
    18: {0: 0.14, 1: 0.16, 2: 0.14, 3: 0.16, 4: 0.32, 5: 0.24, 6: 0.32, 7: 0.24},

    # 19:00 - post-work shopping: heavy balanced
    19: {0: 0.22, 1: 0.18, 2: 0.22, 3: 0.18, 4: 0.22, 5: 0.18, 6: 0.22, 7: 0.18},

    # 20:00 - night still busy (medium-heavy)
    20: {0: 0.14, 1: 0.10, 2: 0.14, 3: 0.10, 4: 0.14, 5: 0.10, 6: 0.14, 7: 0.10},

    # 21:00 - late night light-medium
    21: {0: 0.08, 1: 0.05, 2: 0.08, 3: 0.05, 4: 0.08, 5: 0.05, 6: 0.08, 7: 0.05},

    # 22:00 - late night light
    22: {0: 0.44, 1: 0.02, 2: 0.04, 3: 0.02, 4: 0.04, 5: 0.02, 6: 0.04, 7: 0.02},

    # 23:00 - near empty (not fully empty so you still see occasional cars)
    23: {0: 0.41, 1: 0.00, 2: 0.01, 3: 0.00, 4: 0.01, 5: 0.00, 6: 0.01, 7: 0.00},
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
