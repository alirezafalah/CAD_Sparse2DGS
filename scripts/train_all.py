import os

# The list of scenes to be processed
scenes = [24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122]

# --- Configuration ---
dtu_path = 'data/dtu_sparse/dtu_sparse'
output_dir = "exp_dtu/release"
dry_run = False

# --- Main Training Loop ---
print("Starting training process for all scenes...")

# Define the path where your custom modules were installed
PYTHON_PATH_FIX = "/usr/lib/python3.8/site-packages"

for scene in scenes:
    print(f"\n----- Processing Scene: {scene} -----")
    
    # =================================================================================
    # THE FIX IS HERE: Prepending PYTHONPATH to the command
    # =================================================================================
    cmd = (
        f"PYTHONPATH={PYTHON_PATH_FIX}:$PYTHONPATH "  # <-- This tells Python where to find the modules
        f"OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python3.8 train.py " # Using python3.8 explicitly
        f"-s {dtu_path}/scan{scene} "
        f"-m {output_dir}/scan{scene} "
        f"-r 2 --depth_ratio 1 --lambda_dist 1000 --port 13{scene}"
    )
    
    print(f"Executing command:\n{cmd}")
    
    if not dry_run:
        os.makedirs(f"{output_dir}/scan{scene}", exist_ok=True)
        os.system(cmd)

print("\n\n✅ All scenes have been processed.")
