import os
import time
from src.circuit_utils_12q import get_cut_subcircuits
from src.executor import VirtualExecutor
from src.ml_model_s import LearnedCorrector
from src.calibration import generate_calibration_circuits

# কনফিগারেশন
NUM_SAMPLES = 50
SHOTS = 4000
MODEL_DIR = "saved_models"

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

def main():
    print("=== STEP 1: Offline Training & Model Saving ===\n")
    
    # আমরা ফেক ব্যাকএন্ড ব্যবহার করছি দ্রুত ট্রেনিংয়ের জন্য
    executor = VirtualExecutor(backend_name='fez')
    subcircuits = get_cut_subcircuits()
    
    print(f"Preparing to train models for {len(subcircuits)} fragments...")
    
    start_time = time.time()
    
    for i, sub_qc in enumerate(subcircuits):
        print(f"\n>>> Training Model for Fragment {i+1} <<<")
        
        # ১. ক্যালিব্রেশন ডেটা জেনারেট
        train_circuits, train_ideals = generate_calibration_circuits([sub_qc], num_samples=NUM_SAMPLES)
        
        # ২. নয়েজ রান (Simulation)
        train_noisy_vals = executor.run_mitigation_experiments(train_circuits, shots=SHOTS)
        
        # ৩. মডেল ট্রেনিং
        corrector = LearnedCorrector()
        corrector.train(train_noisy_vals, train_ideals)
        
        # ৪. মডেল সেভ করা
        filename = f"{MODEL_DIR}/model_fragment_{i}.pkl"
        corrector.save_model(filename)

    end_time = time.time()
    print("\n" + "="*40)
    print(f"TRAINING COMPLETE.")
    print(f"Total Time Taken: {end_time - start_time:.2f} seconds")
    print(f"All models saved in '{MODEL_DIR}/'")
    print("="*40)

if __name__ == "__main__":
    main()