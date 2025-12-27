import os.path

from src.simulation import Simulation

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
