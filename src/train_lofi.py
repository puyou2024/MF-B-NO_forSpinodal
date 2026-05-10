# train_lofi_bayesian_activelearning
# (Refactored for 4 AL modes and score logging)

import os
import re
import glob
import time
import copy
import joblib
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from collections import defaultdict 
import shutil 

from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_kl_loss

def set_seed(seed):
    """Sets the seed for random, numpy, and torch for reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # No print statement here to avoid clutter during repeated runs

# --- 1. DeepONet Model Definition (Unchanged) ---
class DeepONet(nn.Module):
    def __init__(self, branch_input_dim, trunk_input_dim, latent_dim=100):
        super(DeepONet, self).__init__()
        self.branch_net = nn.Sequential(
            nn.Linear(branch_input_dim, 32), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(32, 32), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(32, latent_dim)
        )
        self.trunk_net = nn.Sequential(
            nn.Linear(trunk_input_dim, 32), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(32, 32), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(32, latent_dim)
        )

    def forward(self, x_branch, x_trunk):
        branch_out = self.branch_net(x_branch)
        trunk_out = self.trunk_net(x_trunk)
        outputs = torch.sum(branch_out * trunk_out, dim=1, keepdim=True)
        return outputs

# --- 2. Data Loading and Processing (Unchanged) ---
def process_files(file_list):
    branch_data, trunk_data, targets = [], [], []
    pattern = re.compile(r"[sS]pin_([\d\.\-]+)_([\d\.\-]+)_([\d\.\-]+).+?_([XYZ])_")
    for file_path in file_list:
        filename = os.path.basename(file_path)
        match = pattern.search(filename)
        if not match: continue
        params = [float(p) for p in match.groups()[:3]]
        direction = match.groups()[3].upper()
        dir_one_hot = {'X': [1.,0.,0.], 'Y': [0.,1.,0.], 'Z': [0.,0.,1.]}[direction]
        try:
            df = pd.read_csv(file_path, header=None)
            if df.empty:
                continue
            for _, row in df.iterrows():
                phase, strain, stress = row[0], row[1], row[2]
                branch_data.append(params)
                trunk_data.append([strain, phase] + dir_one_hot)
                targets.append(stress)
        except (pd.errors.EmptyDataError, FileNotFoundError):
            continue
    if not branch_data:
         return (np.array([], dtype=np.float32).reshape(0,3),
                 np.array([], dtype=np.float32).reshape(0,5),
                 np.array([], dtype=np.float32).reshape(0, 1))
    return (np.array(branch_data, dtype=np.float32),
            np.array(trunk_data, dtype=np.float32),
            np.array(targets, dtype=np.float32).reshape(-1, 1))

# --- 3. Training Function (Unchanged) ---
def run_training(model, train_loader, test_loader, epochs, lr, kl_weight, model_save_path, device, USE_BAYESIAN, **prediction_kwargs):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    history = {
        'train_loss': [], 'test_loss': [], 'lr': [],
        'train_mse': [], 'test_mse': [],
        'train_kl': [], 'test_kl': []
    }
    
    for epoch in range(epochs):
        model.train() 
        running_train_loss, running_train_mse, running_train_kl = 0.0, 0.0, 0.0
        for branch, trunk, target in train_loader:
            branch, trunk, target = branch.to(device), trunk.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(branch, trunk)
            
            mse_loss = criterion(output, target)
            
            if USE_BAYESIAN:
                kl_loss = get_kl_loss(model)
                loss = mse_loss + kl_weight * kl_loss
                running_train_kl += kl_loss.item() * branch.size(0)
            else:
                loss = mse_loss
                running_train_kl += 0.0

            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item() * branch.size(0)
            running_train_mse += mse_loss.item() * branch.size(0)

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        history['train_loss'].append(epoch_train_loss)
        history['train_mse'].append(running_train_mse / len(train_loader.dataset))
        history['train_kl'].append(running_train_kl / len(train_loader.dataset))

        epoch_test_loss = np.nan
        epoch_test_mse = np.nan
        epoch_test_kl = np.nan
        
        torch.save(model.state_dict(), model_save_path)
        
        history['test_loss'].append(epoch_test_loss)
        history['test_mse'].append(epoch_test_mse)
        history['test_kl'].append(epoch_test_kl)
        
        #print(f"Epoch {epoch+1}/{epochs} | Train Loss: {history['train_loss'][-1]:.6f}")

        history['lr'].append(optimizer.param_groups[0]['lr'])
            
    return history
# --- End of Modified Function ---


# --- 4. Bayesian Prediction Saving Function (Unchanged) ---
def save_bayesian_predictions_by_file(model, prediction_dir, files_list, strain_scaler, device, num_samples):
    os.makedirs(prediction_dir, exist_ok=True)
    model.train() # Use train mode for BNN sampling
    for file_path in files_list:
        base_filename = os.path.basename(file_path) # Moved up
        try:
            branch_data, trunk_data, target_data = process_files([file_path])
            if branch_data.size == 0: continue
            
            branch_data_s = branch_data / 90.0
            
            trunk_data_s = trunk_data.copy()
            trunk_data_s[:, 0:1] = strain_scaler.transform(trunk_data[:, 0:1])
            branch_tensor = torch.from_numpy(branch_data_s).to(device)
            trunk_tensor = torch.from_numpy(trunk_data_s).to(device)
            with torch.no_grad():
                all_predictions = [model(branch_tensor, trunk_tensor).cpu().numpy() for _ in range(num_samples)]
            all_predictions = np.hstack(all_predictions)
            pred_mean = np.mean(all_predictions, axis=1)
            pred_std = np.std(all_predictions, axis=1)
            result_df = pd.DataFrame({
                'phase': trunk_data[:, 1], 'strain': trunk_data[:, 0],
                'original_stress': target_data.flatten(),
                'predicted_stress_mean': pred_mean,
                'predicted_stress_std': pred_std,
            })
            output_filename = f"prediction_uq_{base_filename}"
            output_path = os.path.join(prediction_dir, output_filename)
            result_df.to_csv(output_path, index=False)
        except Exception as e:
            print(f"Could not process file {base_filename} for prediction. Error: {e}")
    model.eval()

# --- 5. Overall MSE Evaluation Function (Unchanged) ---
def evaluate_model_overall_mse(model, loader, device, USE_BAYESIAN):
    criterion = nn.MSELoss() 
    
    if USE_BAYESIAN:
        model.train()
    else:
        model.eval()
    
    running_mse = 0.0 
    
    with torch.no_grad():
        for branch, trunk, target in loader:
            branch, trunk, target = branch.to(device), trunk.to(device), target.to(device)
            output = model(branch, trunk)
            mse_loss = criterion(output, target) 
            running_mse += mse_loss.item() * branch.size(0) 
    
    model.eval() # Set back to eval mode
    if len(loader.dataset) == 0: return np.nan
    return running_mse / len(loader.dataset) 

# --- 6. Per-File MSE Evaluation Function (Unchanged) ---
def evaluate_mse_per_file(model, file_list, strain_scaler, device, USE_BAYESIAN):
    
    if USE_BAYESIAN:
        model.train()
    else:
        model.eval()
    
    criterion = nn.MSELoss(reduction='mean') 
    file_mses = {} 
    for file_path in file_list:
        base_filename = os.path.basename(file_path) # Moved up
        try:
            branch_data, trunk_data, target_data = process_files([file_path])
            if branch_data.size == 0:
                file_mses[base_filename] = np.nan; continue
                
            branch_data_s = branch_data / 90.0
            
            trunk_data_s = trunk_data.copy(); trunk_data_s[:, 0:1] = strain_scaler.transform(trunk_data[:, 0:1])
            branch_tensor = torch.from_numpy(branch_data_s).to(device)
            trunk_tensor = torch.from_numpy(trunk_data_s).to(device)
            target_tensor = torch.from_numpy(target_data).to(device)
            with torch.no_grad():
                output = model(branch_tensor, trunk_tensor)
                mse = criterion(output, target_tensor).item() 
            file_mses[base_filename] = mse 
        except Exception as e:
            file_mses[base_filename] = np.nan 
            
    model.eval() # Set back to eval
    return file_mses 

# --- 7. Absolute Error Calculation (Unchanged) ---
def calculate_absolute_errors(model, loader, device, USE_BAYESIAN):
    
    if USE_BAYESIAN:
        model.train()
    else:
        model.eval()
    
    all_errors = []
    with torch.no_grad():
        for branch, trunk, target in loader:
            branch, trunk, target = branch.to(device), trunk.to(device), target.to(device)
            output = model(branch, trunk)
            errors = torch.abs(output - target).cpu().numpy()
            all_errors.extend(errors.flatten())
            
    model.eval() # Set back to eval
    return np.array(all_errors)

# --- 8. File Pool Parsing Function (Unchanged) ---
def get_file_pool_details(file_list):
    file_details = []; unique_angle_sets = set()
    pattern = re.compile(r"[sS]pin_([\d\.\-]+)_([\d\.\-]+)_([\d\.\-]+).+?_([XYZ])_")
    for file_path in file_list:
        filename = os.path.basename(file_path)
        match = pattern.search(filename)
        if not match: continue
        angles = tuple(float(p) for p in match.groups()[:3])
        direction = match.groups()[3].upper()
        file_details.append((file_path, angles, direction))
        unique_angle_sets.add(angles)
    return file_details, unique_angle_sets

# --- 9. Acquisition Helper Functions (Unchanged) ---
def get_angle_type(angles):
    """Classifies angles into one of three types."""
    if angles[0] != 0:
        return 3 # (theta_3, theta_2, theta_1)
    elif angles[1] != 0:
        return 2 # (0, theta_2, theta_1)
    else:
        return 1 # (0, 0, theta_1)

def calculate_scaled_distance(pool_angles, train_angles_np, type1_w, type2_w, type3_w):
    """Calculates the scaled distance from one pool sample to one training sample."""
    dist = np.linalg.norm(pool_angles - train_angles_np)
    angle_type = get_angle_type(pool_angles)
    
    if angle_type == 1:
        return type1_w * dist
    elif angle_type == 2:
        return type2_w * dist / np.sqrt(2)
    else: # Type 3
        return type3_w * dist / np.sqrt(3)

# --- 10. REFACTORED ACQUISITION ---

def calculate_variance_scores(model, pool_angle_map, strain_scaler, device, num_samples):
    """
    Calculates the 'Max Variance / Max Mean Stress' score for each angle set in the pool.
    """
    print("  Calculating Term 1 (Uncertainty) for all pool samples...")
    model.train()
    variance_scores = {}
    
    for angles, files in pool_angle_map.items():
        angle_max_rel_variance = -np.inf
        for file_path in files:
       
            try:
                branch_data, trunk_data, _ = process_files([file_path])
                if branch_data.size == 0: continue
                
                branch_data_s = branch_data / 90.0
                trunk_data_s = trunk_data.copy(); trunk_data_s[:, 0:1] = strain_scaler.transform(trunk_data[:, 0:1])
                branch_tensor = torch.from_numpy(branch_data_s).to(device)
                trunk_tensor = torch.from_numpy(trunk_data_s).to(device)
                
                with torch.no_grad():
                    preds = [model(branch_tensor, trunk_tensor).cpu().numpy() for _ in range(num_samples)]
                preds = np.hstack(preds).T
                
                mean_curve = np.mean(preds, axis=0)
                variance_curve = np.var(preds, axis=0)
                epsilon = 1e-8 
                
                max_variance = np.max(variance_curve)
                max_abs_mean_stress = np.max(np.abs(mean_curve))
                
                file_score = max_variance / (max_abs_mean_stress + epsilon)
                #file_score = max_variance
                angle_max_rel_variance = max(angle_max_rel_variance, file_score)
                
            except Exception as e: pass
            
        if angle_max_rel_variance > -np.inf:
            variance_scores[angles] = angle_max_rel_variance
        else:
            variance_scores[angles] = 0.0
            
    return variance_scores

def calculate_distance_scores(pool_angle_map, current_training_details, dist_weights):
    """
    Calculates the minimum scaled distance to the training set for each angle set in the pool.
    """
    print("  Calculating Term 2 (Diversity) for all pool samples...")
    distance_scores = {}
    temp_training_angles = set(angles for _, angles, _ in current_training_details)
    
    for angles in pool_angle_map.keys():
        pool_angles_np = np.array(angles)
        min_dist = np.inf
        
        if not temp_training_angles:
             min_dist = 0.0 # Should not happen after step 0, but safe
        else:
            for train_angles in temp_training_angles:
                train_angles_np = np.array(train_angles)
                scaled_dist = calculate_scaled_distance(pool_angles_np, train_angles_np, **dist_weights)
                min_dist = min(min_dist, scaled_dist)
        
        distance_scores[angles] = min_dist
        
    return distance_scores

def find_next_batch(
    model, remaining_file_details, current_training_details,
    strain_scaler, device, num_samples, 
    batch_size, dist_weights, 
    acquisition_mode, log_scores,
    pool_score_dir, al_step,
    group_probs=[0.33, 0.33, 0.34] # Default argument
):
    """
    Selects the next batch of samples based on the chosen acquisition mode.
    Logs pool scores if requested.
    """
    print(f"--- Running Acquisition for {batch_size} samples (Mode: {acquisition_mode}) ---")
    
    # 1. Group pool files by angle
    pool_angle_map = defaultdict(list)
    for fp, ang, dr in remaining_file_details:
        pool_angle_map[ang].append(fp)
    
    if not pool_angle_map:
        print("Acquisition: No more files in training pool.")
        return set()
        
    print(f"Analyzing {len(pool_angle_map)} unique angle sets in the pool...")
    
    # 2. Handle Random acquisition separately (Weighted by Group)
    if acquisition_mode == 'random':
        unique_pool_angles_list = list(pool_angle_map.keys())
        
        if len(unique_pool_angles_list) <= batch_size:
            print(f"  Pool size ({len(unique_pool_angles_list)}) is <= batch size. Acquiring all.")
            newly_selected_angles = set(unique_pool_angles_list)
        else:
            print(f"  Weighted Random sampling {batch_size} angles (Probs: {group_probs})...")
            
            # A. Calculate weight for every candidate in the pool
            weights = []
            for ang in unique_pool_angles_list:
                # Determine group type (1, 2, or 3)
                # Note: get_angle_type returns 1, 2, or 3. We map to index 0, 1, 2.
                g_type = get_angle_type(ang) 
                weight = group_probs[g_type - 1]
                weights.append(weight)
            
            # B. Normalize weights to sum to 1 (numpy.random.choice requires this)
            weights = np.array(weights)
            weights = weights / np.sum(weights)
            
            # C. Select samples (replace=False ensures we don't pick the same one twice)
            # We use indices to select from the list
            selected_indices = np.random.choice(
                len(unique_pool_angles_list), 
                size=batch_size, 
                replace=False, 
                p=weights
            )
            
            newly_selected_angles = set([unique_pool_angles_list[i] for i in selected_indices])

        
        # Log dummy scores if requested
        if log_scores:
            log_data = []
            for ang in pool_angle_map.keys():
                log_data.append([ang[0], ang[1], ang[2], 0.0, 0.0, 0.0, 0.0, 0.0])
            log_df = pd.DataFrame(log_data, columns=[
                'angle_1', 'angle_2', 'angle_3', 'raw_variance_score', 'raw_distance_score',
                'norm_variance_score', 'norm_distance_score', 'final_score'
            ])
            log_path = os.path.join(pool_score_dir, f'pool_scores_step_{al_step}.csv')
            log_df.to_csv(log_path, index=False)
            print(f"  Dummy scores for random mode logged to {log_path}")

        print("\n--- Top Acquisition Targets (Random) ---")
        for ang in newly_selected_angles: print(f"  {ang}")
        return newly_selected_angles

    # 3. Calculate Scores for non-random modes
    # Calculate Uncertainty (Variance)
    if acquisition_mode in ['uncertainty', 'hybrid']:
        raw_variance_scores = calculate_variance_scores(model, pool_angle_map, strain_scaler, device, num_samples)
    else:
        raw_variance_scores = {ang: 0.0 for ang in pool_angle_map.keys()}

    # Calculate Diversity (Distance)
    if acquisition_mode in ['diversity', 'hybrid']:
        raw_distance_scores = calculate_distance_scores(pool_angle_map, current_training_details, dist_weights)
    else:
        raw_distance_scores = {ang: 0.0 for ang in pool_angle_map.keys()}
        
    # 4. Normalize Scores and set Final Score
    pool_samples_data = []
    
    # --- Normalization ---
    all_var_scores = np.array(list(raw_variance_scores.values()))
    all_dist_scores = np.array(list(raw_distance_scores.values()))
    
    min_var = np.min(all_var_scores); max_var = np.max(all_var_scores)
    min_dist = np.min(all_dist_scores); max_dist = np.max(all_dist_scores)
    
    epsilon = 1e-8
    
    for ang in pool_angle_map.keys():
        raw_var = raw_variance_scores[ang]
        raw_dist = raw_distance_scores[ang]
        
        norm_var = (raw_var - min_var) / (max_var - min_var + epsilon)
        norm_dist = (raw_dist - min_dist) / (max_dist - min_dist + epsilon)
        
        # Set scores to 0 if not used by this mode
        if acquisition_mode not in ['uncertainty', 'hybrid']:
            norm_var = 0.0
        if acquisition_mode not in ['diversity', 'hybrid']:
            norm_dist = 0.0
            
        # Calculate final score based on mode
        final_score = 0.0
        if acquisition_mode == 'uncertainty':
            final_score = norm_var
        elif acquisition_mode == 'diversity':
            final_score = norm_dist
        elif acquisition_mode == 'hybrid':
            final_score = norm_var + norm_dist
            
        pool_samples_data.append({
            'angles': ang,
            'raw_var': raw_var,
            'raw_dist': raw_dist,
            'norm_var': norm_var,
            'norm_dist': norm_dist,
            'final_score': final_score
        })

    # 5. Log Scores to CSV if requested
    if log_scores:
        log_data = []
        for s in pool_samples_data:
            log_data.append([
                s['angles'][0], s['angles'][1], s['angles'][2],
                s['raw_var'], s['raw_dist'],
                s['norm_var'], s['norm_dist'],
                s['final_score']
            ])
        log_df = pd.DataFrame(log_data, columns=[
            'angle_1', 'angle_2', 'angle_3', 'raw_variance_score', 'raw_distance_score',
            'norm_variance_score', 'norm_distance_score', 'final_score'
        ])
        log_path = os.path.join(pool_score_dir, f'pool_scores_step_{al_step}.csv')
        log_df.to_csv(log_path, index=False)
        print(f"  Pool scores logged to {log_path}")

    # 6. Greedy Selection
    newly_selected_angles = set()
    print(f"\n  Starting greedy selection loop for {batch_size} samples...")
    
    temp_training_angles = set(angles for _, angles, _ in current_training_details)
    
    for i in range(batch_size):
        if not pool_samples_data:
            print(f"  Pool exhausted. Selected {i} samples.")
            break
            
        # Sort by the final_score calculated earlier
        pool_samples_data.sort(key=lambda x: x['final_score'], reverse=True)
        best_sample = pool_samples_data.pop(0) 
        
        best_angles = best_sample['angles']
        newly_selected_angles.add(best_angles)
        print(f"    Selected {i+1}/{batch_size}: {best_angles} (Score: {best_sample['final_score']:.4f})")
        
        # Update distances for next greedy step (only if diversity is a factor)
        if acquisition_mode in ['diversity', 'hybrid']:
            newly_selected_angles_np = np.array(best_angles)
            for s in pool_samples_data:
                # Update raw distance score
                new_scaled_dist = calculate_scaled_distance(np.array(s['angles']), newly_selected_angles_np, **dist_weights)
                s['raw_dist'] = min(s['raw_dist'], new_scaled_dist)
            
            # Re-normalize distances and update final scores for the next pick
            all_dist_scores = np.array([s['raw_dist'] for s in pool_samples_data])
            min_dist = np.min(all_dist_scores); max_dist = np.max(all_dist_scores)

            for s in pool_samples_data:
                s['norm_dist'] = (s['raw_dist'] - min_dist) / (max_dist - min_dist + epsilon)
                if acquisition_mode == 'diversity':
                    s['final_score'] = s['norm_dist']
                elif acquisition_mode == 'hybrid':
                    s['final_score'] = s['norm_var'] + s['norm_dist']
                # 'uncertainty' mode doesn't need updating as final_score doesn't depend on distance
            
    print(f"\n--- Top Acquisition Targets ({acquisition_mode}) ---")
    for ang in newly_selected_angles: print(f"  {ang}")
    
    return newly_selected_angles

# --- 11. Main Execution Block (MODIFIED) ---
if __name__ == "__main__":
    main_start_time = time.time()
    device = torch.device("cpu")

    # --- NEW: Refactor Settings ---
    # Set ACQUISITION_MODE to one of: 'random', 'uncertainty', 'diversity', 'hybrid'
    ACQUISITION_MODE = 'hybrid'
    LOG_POOL_SCORES = False
    # --- End New Settings ---

    USE_BAYESIAN = True
    GROUP_PROBS = [0.1, 0.4, 0.5]

    BASE_DIR = ".."
    LOFI_DATA_DIR = os.path.join(BASE_DIR, "data/lofi/lofi_raw")
    TEST_DATA_DIR = os.path.join(BASE_DIR, "data/lofi/test_cases")
    INITIAL_TRAIN_DIR = os.path.join(BASE_DIR, "data/lofi/initial_train")
    
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs/lofi_results")
    MSE_DIST_DIR = os.path.join(OUTPUT_DIR, 'mse_distributions') 
    ACQUIRED_ANGLES_DIR = os.path.join(OUTPUT_DIR, 'acquired_angles_per_step')
    ACQUIRED_FILES_DIR = os.path.join(OUTPUT_DIR, 'acquired_files_per_step') 
    POOL_SCORE_DIR = os.path.join(OUTPUT_DIR, 'pool_scores_per_step') # NEW: Log dir
    HISTORY_DIR = os.path.join(OUTPUT_DIR, 'training_histories')
    os.makedirs(HISTORY_DIR, exist_ok=True)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MSE_DIST_DIR, exist_ok=True) 
    os.makedirs(ACQUIRED_ANGLES_DIR, exist_ok=True)
    os.makedirs(ACQUIRED_FILES_DIR, exist_ok=True) 
    os.makedirs(POOL_SCORE_DIR, exist_ok=True) # NEW: Create log dir

    STRAIN_SCALER_PATH = os.path.join(OUTPUT_DIR, 'strain_scaler.gz')

    RANDOM_STATE = 24
    BATCH_SIZE = 1024
    LATENT_DIM = 128
    EPOCHS = 1000
    LR = 1e-3
    KL_WEIGHT = 1e-5
    PREDICTION_SAMPLES = 50

    MAX_AL_STEPS = 20
    ACQUISITION_BATCH_SIZE = 10

    TRAINING_SEEDS = [1]; NUM_RUNS = len(TRAINING_SEEDS)
    
    DISTANCE_WEIGHTS = {
        'type1_w': 1.0,
        'type2_w': 1.0,
        'type3_w': 1.0
    }

    const_bnn_prior_parameters = {
        "prior_mu": 0.0, "prior_sigma": 1.0, "posterior_mu_init": 0.0,
        "posterior_rho_init": -4.0, "type": "Reparameterization", "moped_enable": False,
    }

    set_seed(RANDOM_STATE) # This seed now controls everything
    print(f"--- Using device: {device} ---"); print(f"--- Initial Pool Seed: {RANDOM_STATE} ---"); print(f"--- Training Seeds: {TRAINING_SEEDS} ---")
    print(f"--- ACQUISITION_MODE: {ACQUISITION_MODE} ---")
    print(f"--- LOG_POOL_SCORES: {LOG_POOL_SCORES} ---")
    
    print("\n--- STAGE 1: Loading Data ---")

    train_pool_files = glob.glob(os.path.join(LOFI_DATA_DIR, '*.csv'))
    if not train_pool_files: raise FileNotFoundError(f"No files in pool: {LOFI_DATA_DIR}")
    print(f"Found {len(train_pool_files)} total pool files.")
    
    test_files = glob.glob(os.path.join(TEST_DATA_DIR, '*.csv'))
    if not test_files: raise FileNotFoundError(f"No files in test set: {TEST_DATA_DIR}")
    branch_test, trunk_test, targets_test = process_files(test_files)
    if branch_test.size == 0: raise ValueError("Processing test set yielded no data.")
    
    print(f"Loaded {len(test_files)} test files.")
    test_file_details, _ = get_file_pool_details(test_files)

    print("\n--- STAGE 1b: Loading Initial Training Set ---")
    initial_train_files = glob.glob(os.path.join(INITIAL_TRAIN_DIR, '*.csv'))
    if not initial_train_files: 
        raise FileNotFoundError(f"No initial training files found in: {INITIAL_TRAIN_DIR}")
    
    current_training_details, initial_angle_sets = get_file_pool_details(initial_train_files)
    current_training_files = [fp for fp, _, _ in current_training_details]
    initial_basenames = set(os.path.basename(f) for f in current_training_files)
    print(f"Loaded {len(current_training_files)} initial training files from {INITIAL_TRAIN_DIR} ({len(initial_angle_sets)} angle sets)")

    print("Fitting scaler on initial training files for consistency...")
    branch_initial_train, trunk_initial_train, _ = process_files(initial_train_files)
    if branch_initial_train.size == 0:
        raise ValueError("Initial training files yielded no data. Cannot fit scaler.")
    
    strain_scaler = MinMaxScaler().fit(trunk_initial_train[:, 0:1])
    joblib.dump(strain_scaler, STRAIN_SCALER_PATH)
    print("Strain scaler fit on initial train data and saved.")

    branch_test_s = branch_test / 90.0
    trunk_test_s = trunk_test.copy(); trunk_test_s[:, 0:1] = strain_scaler.transform(trunk_test[:, 0:1])
    test_dataset = TensorDataset(torch.from_numpy(branch_test_s), torch.from_numpy(trunk_test_s), torch.from_numpy(targets_test))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    file_details_pool, _ = get_file_pool_details(train_pool_files)
    remaining_file_details = []
    for fp, ang, dr in file_details_pool:
        if os.path.basename(fp) not in initial_basenames:
            remaining_file_details.append((fp, ang, dr))
            
    print(f"Remaining pool: {len(remaining_file_details)} files")

    # --- STAGE 2: Active Learning Loop (Unchanged, uses modified training function) ---
    all_step_all_run_histories = []
    all_step_all_run_train_mses = []; all_step_all_run_test_mses = [] 
    best_model_test_abs_errors_per_step = []
    best_model_paths_per_step = []

    for al_step in range(MAX_AL_STEPS):
        step_start_time = time.time()
        print(f"\n--- AL STEP {al_step}/{MAX_AL_STEPS-1} ---") 
        print(f"Train set size: {len(current_training_files)} files")

        current_angle_sets = sorted(list(set([angles for _, angles, _ in current_training_details])))
        angles_save_path = os.path.join(ACQUIRED_ANGLES_DIR, f'al_step_{al_step}_train_angles.txt')
        with open(angles_save_path, 'w') as f:
            for ang in current_angle_sets:
                f.write(f"{ang[0]},{ang[1]},{ang[2]}\n")
        print(f"Saved {len(current_angle_sets)} unique training angles to {angles_save_path}")

        step_histories = []; step_train_mses = []; step_test_mses = []; step_model_paths = []
        step_train_mses_per_file = {}; step_test_mses_per_file = {}

        print(f"Starting {NUM_RUNS} runs...")
        for i, run_seed in enumerate(TRAINING_SEEDS):
            run_start_time = time.time()
            print(f"  Run {i+1}/{NUM_RUNS} (Seed: {run_seed})")
            
            branch_train, trunk_train, targets_train = process_files(current_training_files)
            if branch_train.size == 0:
                 print(f"    Skipping Run {i+1}: empty train data."); step_train_mses.append(np.nan); step_test_mses.append(np.nan) 
                 step_model_paths.append(None); step_histories.append({}); continue
            
            branch_train_s = branch_train / 90.0
            
            trunk_train_s = trunk_train.copy(); trunk_train_s[:, 0:1] = strain_scaler.transform(trunk_train[:, 0:1])
            train_dataset = TensorDataset(torch.from_numpy(branch_train_s), torch.from_numpy(trunk_train_s), torch.from_numpy(targets_train))
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            
            run_model = DeepONet(3, 5, LATENT_DIM)
            dnn_to_bnn(run_model, const_bnn_prior_parameters)
            run_model.to(device)
            
            RUN_MODEL_PATH = os.path.join(OUTPUT_DIR, f'model_step_{al_step}_run_{i}.pth'); step_model_paths.append(RUN_MODEL_PATH)
            
            history = run_training(
                run_model, train_loader, test_loader, EPOCHS, LR, 
                KL_WEIGHT, RUN_MODEL_PATH, device, USE_BAYESIAN, 
                run_seed=run_seed 
            )
            step_histories.append(history)
            
            # Create a DataFrame for the history of this specific run
            history_df = pd.DataFrame({
                'epoch': range(1, len(history['train_mse']) + 1),
                'mse': history['train_mse'],
                'kl': history['train_kl'],
                'loss': history['train_loss']
            })
            
            # Define filename: e.g., history_step_0_run_0.csv
            history_filename = f"history_step_{al_step}_run_{i}.csv"
            history_save_path = os.path.join(HISTORY_DIR, history_filename)
            
            # Save to CSV
            history_df.to_csv(history_save_path, index=False)
            print(f"    Saved training history to {history_filename}")
            
            run_model.load_state_dict(torch.load(RUN_MODEL_PATH, weights_only=True))
            
            final_train_mse = evaluate_model_overall_mse(run_model, train_loader, device, USE_BAYESIAN)
            final_test_mse = evaluate_model_overall_mse(run_model, test_loader, device, USE_BAYESIAN)
            step_train_mses.append(final_train_mse); step_test_mses.append(final_test_mse)
            
            train_mses_dict = evaluate_mse_per_file(run_model, current_training_files, strain_scaler, device, USE_BAYESIAN)
            for fname, mse in train_mses_dict.items(): step_train_mses_per_file.setdefault(fname, [np.nan]*NUM_RUNS)[i] = mse
            test_mses_dict = evaluate_mse_per_file(run_model, test_files, strain_scaler, device, USE_BAYESIAN)
            for fname, mse in test_mses_dict.items(): step_test_mses_per_file.setdefault(fname, [np.nan]*NUM_RUNS)[i] = mse
            
            run_end_time = time.time()
            print(f"  Finished Run {i+1}. Test MSE (post-eval): {final_test_mse:.6f}. Time: {(run_end_time - run_start_time)/60:.2f} min") 
            del run_model; torch.cuda.empty_cache()

        all_step_all_run_histories.append(step_histories); all_step_all_run_train_mses.append(step_train_mses); all_step_all_run_test_mses.append(step_test_mses) 
        
        valid_test_mses = np.array(step_test_mses) 
        if np.all(np.isnan(valid_test_mses)): 
            print(f"Step {al_step}: All runs failed."); best_run_index = -1; best_model_path = None
            best_model_paths_per_step.append(best_model_path); best_model_test_abs_errors_per_step.append(np.array([]))
            current_best_model_for_acq = None
        else:
            best_run_index = np.nanargmin(valid_test_mses); best_run_test_mse = valid_test_mses[best_run_index] 
            best_model_path = step_model_paths[best_run_index]; best_model_paths_per_step.append(best_model_path)
            print(f"Step {al_step} Best Run: {best_run_index+1} (Seed: {TRAINING_SEEDS[best_run_index]}) Test MSE: {best_run_test_mse:.6f}") 
            best_model = DeepONet(3, 5, LATENT_DIM)
            dnn_to_bnn(best_model, const_bnn_prior_parameters)
            best_model.load_state_dict(torch.load(best_model_path, weights_only=True)); best_model.to(device)
            
            abs_errors = calculate_absolute_errors(best_model, test_loader, device, USE_BAYESIAN)
            best_model_test_abs_errors_per_step.append(abs_errors)
            BEST_PRED_DIR = os.path.join(OUTPUT_DIR, f'al_step_{al_step}_test_predictions_best')
            
            save_bayesian_predictions_by_file(best_model, BEST_PRED_DIR, test_files, strain_scaler, device, PREDICTION_SAMPLES)
            print(f"Best model predictions saved.")
            
            current_best_model_for_acq = best_model

        train_mse_df = pd.DataFrame.from_dict(step_train_mses_per_file, orient='index', columns=[f'run_{s}' for s in TRAINING_SEEDS])
        test_mse_df = pd.DataFrame.from_dict(step_test_mses_per_file, orient='index', columns=[f'run_{s}' for s in TRAINING_SEEDS])
        train_mse_df.to_csv(os.path.join(MSE_DIST_DIR, f'al_step_{al_step}_train_mses.csv'))
        test_mse_df.to_csv(os.path.join(MSE_DIST_DIR, f'al_step_{al_step}_test_mses.csv'))

        if al_step < MAX_AL_STEPS - 1:
            if current_best_model_for_acq is None: print("--- Skipping acquisition: no valid model. ---"); continue
            if not remaining_file_details: print("--- No more files. Stopping AL. ---"); break
            
            # --- MODIFIED: Call the new refactored acquisition function ---
            new_angle_sets = find_next_batch(
                current_best_model_for_acq, remaining_file_details, current_training_details,
                strain_scaler, device, 
                PREDICTION_SAMPLES, ACQUISITION_BATCH_SIZE, DISTANCE_WEIGHTS,
                ACQUISITION_MODE, LOG_POOL_SCORES,
                POOL_SCORE_DIR, al_step,
                group_probs=GROUP_PROBS
            )
            
            if not new_angle_sets: print("--- Acquisition returned no files. Stopping. ---"); break
            new_files = []; next_remaining = []
            for fp, ang, dr in remaining_file_details:
                if ang in new_angle_sets: 
                    new_files.append(fp); 
                    current_training_details.append((fp, ang, dr))
                else: 
                    next_remaining.append((fp, ang, dr))

            step_acquired_dir = os.path.join(ACQUIRED_FILES_DIR, f'al_step_{al_step}_added_files')
            os.makedirs(step_acquired_dir, exist_ok=True)
            print(f"--- Copying {len(new_files)} new files to {step_acquired_dir} ---")
            for src_file_path in new_files:
                try:
                    dst_file_path = os.path.join(step_acquired_dir, os.path.basename(src_file_path))
                    shutil.copy2(src_file_path, dst_file_path)
                except Exception as e:
                    print(f"    Warning: Could not copy file {src_file_path}. Error: {e}")

            remaining_file_details = next_remaining; 
            current_training_files.extend(new_files)
            print(f"--- Added {len(new_files)} files ({len(new_angle_sets)} angle sets). ---")
            print(f"--- Remaining pool: {len(remaining_file_details)} files ---")
        else: print("--- Max AL steps reached. ---")

        if current_best_model_for_acq is not None: del current_best_model_for_acq; torch.cuda.empty_cache()
        step_end_time = time.time(); print(f"--- Step {al_step} finished. Time: {(step_end_time - step_start_time)/60:.2f} min ---")


    # === STAGE 3: FINAL PLOTTING (Unchanged) ===
    print("\n--- Generating Final Plots ---")

    num_al_steps_done = len(all_step_all_run_histories)
    steps_for_plot = range(num_al_steps_done) 
    steps_for_label = range(num_al_steps_done) 

    # Plot 1: Total Training Loss Subplots
    if num_al_steps_done > 0:
        rows = int(np.ceil(num_al_steps_done / 5)); cols = min(num_al_steps_done, 5)
        fig1, axes1 = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), squeeze=False, sharey=True) 
        fig1.suptitle('Total Training Loss Histories per AL Step (5 Runs Each)', fontsize=16)
        axes1 = axes1.flatten()
        for i in steps_for_plot:
            ax = axes1[i]
            if all_step_all_run_histories[i] and any(all_step_all_run_histories[i]):
                for run_idx, history in enumerate(all_step_all_run_histories[i]):
                     if history and 'train_loss' in history: 
                         epochs_ran = len(history['train_loss'])
                         if epochs_ran > 0: ax.plot(range(1, epochs_ran + 1), history['train_loss'], label=f'Run {run_idx+1}')
            ax.set_title(f'AL Step {i}'); ax.set_xlabel('Epoch'); ax.set_ylabel('Log(Total Loss)') 
            ax.set_yscale('log'); ax.grid(True)
            if i==0: ax.legend()
        for j in range(num_al_steps_done, len(axes1)): fig1.delaxes(axes1[j])
        plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.savefig(os.path.join(OUTPUT_DIR, 'al_total_loss_history_subplots.png'))

    # Plot 2: KL Divergence History Subplots
    if num_al_steps_done > 0:
        rows = int(np.ceil(num_al_steps_done / 5)); cols = min(num_al_steps_done, 5)
        fig2, axes2 = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), squeeze=False, sharey=True)
        fig2.suptitle('Training KL Divergence Histories per AL Step (5 Runs Each)', fontsize=16)
        axes2 = axes2.flatten()
        for i in steps_for_plot:
            ax = axes2[i]
            if all_step_all_run_histories[i] and any(all_step_all_run_histories[i]):
                for run_idx, history in enumerate(all_step_all_run_histories[i]):
                     if history and 'train_kl' in history:
                         epochs_ran = len(history['train_kl'])
                         if epochs_ran > 0: ax.plot(range(1, epochs_ran + 1), history['train_kl'], label=f'Run {run_idx+1}')
            ax.set_title(f'AL Step {i}'); ax.set_xlabel('Epoch'); ax.set_ylabel('Log(KL Divergence)') 
            ax.set_yscale('log'); ax.grid(True)
            if i==0: ax.legend()
        for j in range(num_al_steps_done, len(axes2)): fig2.delaxes(axes2[j])
        plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.savefig(os.path.join(OUTPUT_DIR, 'al_kl_history_subplots.png'))

    # Plot 3: MSE Loss History Subplots
    if num_al_steps_done > 0:
        rows = int(np.ceil(num_al_steps_done / 5)); cols = min(num_al_steps_done, 5)
        fig3, axes3 = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), squeeze=False, sharey=True) 
        fig3.suptitle('Training MSE Loss Histories per AL Step (5 Runs Each)', fontsize=16) 
        axes3 = axes3.flatten()
        for i in steps_for_plot:
            ax = axes3[i]
            if all_step_all_run_histories[i] and any(all_step_all_run_histories[i]):
                for run_idx, history in enumerate(all_step_all_run_histories[i]):
                     if history and 'train_mse' in history: 
                         epochs_ran = len(history['train_mse']) 
                         if epochs_ran > 0: ax.plot(range(1, epochs_ran + 1), history['train_mse'], label=f'Run {run_idx+1}') 
            ax.set_title(f'AL Step {i}'); ax.set_xlabel('Epoch'); ax.set_ylabel('Log(MSE Loss)') 
            ax.set_yscale('log'); ax.grid(True)
            if i==0: ax.legend()
        for j in range(num_al_steps_done, len(axes3)): fig3.delaxes(axes3[j])
        plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.savefig(os.path.join(OUTPUT_DIR, 'al_mse_loss_history_subplots.png'))


    # Plot 4: Mean Train/Test MSE vs. AL Step
    if num_al_steps_done > 0:
        mean_train_mses = np.nanmean(all_step_all_run_train_mses, axis=1)
        std_train_mses = np.nanstd(all_step_all_run_train_mses, axis=1)
        mean_test_mses = np.nanmean(all_step_all_run_test_mses, axis=1)
        std_test_mses = np.nanstd(all_step_all_run_test_mses, axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.errorbar(steps_for_label, mean_train_mses, yerr=std_train_mses, marker='s', linestyle='--', label='Train MSE (Mean +/- Std)', capsize=5) 
        plt.errorbar(steps_for_label, mean_test_mses, yerr=std_test_mses, marker='o', linestyle='-', label='Test MSE (Mean +/- Std)', capsize=5) 
        plt.title('Mean Train & Test MSE vs. Active Learning Step (5 Runs)', fontsize=16) 
        plt.xlabel('Active Learning Step'); plt.ylabel('MSE'); plt.yscale('log') 
        plt.legend(); plt.xticks(steps_for_label); plt.grid(True)
        plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, 'al_mean_mse_vs_step.png'))

    # Plot 5: Max Absolute Error vs. AL Step
    if num_al_steps_done > 0:
        max_abs_errors = [np.max(err_arr) if err_arr.size > 0 else np.nan for err_arr in best_model_test_abs_errors_per_step]
        plt.figure(figsize=(10, 6))
        valid_steps = [s for s, err in zip(steps_for_label, max_abs_errors) if not np.isnan(err)]
        valid_errors = [err for err in max_abs_errors if not np.isnan(err)]
        if valid_steps:
            plt.plot(valid_steps, valid_errors, marker='^', linestyle='-.', color='green')
            plt.title('Max Abs Error on Test Set vs. AL Step (Best Model)', fontsize=16)
            plt.xlabel('Active Learning Step'); plt.ylabel('Max Absolute Error')
            plt.xticks(steps_for_label); plt.grid(True)
            plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, 'al_max_abs_error_vs_step.png'))
        else: print("Plot 5 skipped: No valid max absolute error data.")


    # Plot 6: Per-File MSE Distribution Histograms
    if num_al_steps_done > 0:
        rows = int(np.ceil(num_al_steps_done / 5)); cols = min(num_al_steps_done, 5)
        fig6, axes6 = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4), squeeze=False, sharex=True, sharey=True)
        fig6.suptitle('Per-File MSE Distribution per AL Step (Best Model)', fontsize=16) 
        axes6 = axes6.flatten()
        plot6_generated = False

        for i in steps_for_plot:
            ax = axes6[i]
            ax.set_title(f'AL Step {i}') 
            step_train_mse_file = os.path.join(MSE_DIST_DIR, f'al_step_{i}_train_mses.csv')
            step_test_mse_file = os.path.join(MSE_DIST_DIR, f'al_step_{i}_test_mses.csv')

            step_test_mses = np.array(all_step_all_run_test_mses[i]) 
            best_run_idx = -1
            if not np.all(np.isnan(step_test_mses)):
                best_run_idx = np.nanargmin(step_test_mses) 

            if best_run_idx != -1 and os.path.exists(step_train_mse_file) and os.path.exists(step_test_mse_file): 
                try:
                    train_df = pd.read_csv(step_train_mse_file, index_col=0) 
                    test_df = pd.read_csv(step_test_mse_file, index_col=0) 
                    best_run_col = f'run_{TRAINING_SEEDS[best_run_idx]}'

                    if best_run_col in train_df.columns and best_run_col in test_df.columns:
                        train_mses = train_df[best_run_col].dropna() 
                        test_mses = test_df[best_run_col].dropna() 
 
                        plotted_train = False
                        if not train_mses.empty:
                            ax.hist(train_mses, bins='auto', density=True, color='blue', alpha=0.6, label='Train MSEs') 
                            plotted_train = True; plot6_generated = True

                        plotted_test = False
                        if not test_mses.empty:
                            ax.hist(test_mses, bins='auto', density=True, color='red', alpha=0.6, label='Test MSEs') 
                            plotted_test = True; plot6_generated = True

                        if plotted_train or plotted_test:
                             ax.set_xscale('log')
                             ax.set_xlabel('MSE (per file)') 
                             ax.set_ylabel('Probability Density')
                             ax.legend()
                             ax.grid(True, which='both', linestyle='--', linewidth=0.5)

                except Exception as e:
                    print(f"Plot 6 (Step {i}): Error processing MSE files: {e}") 
            else:
                 ax.text(0.5, 0.5, 'No valid data for best run', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

        for j in range(num_al_steps_done, len(axes6)): fig6.delaxes(axes6[j])
        if plot6_generated:
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig(os.path.join(OUTPUT_DIR, 'al_mse_distribution_hist_subplots.png')) 
        else:
            print("Plot 6 skipped: No valid data found for any step.")
            plt.close(fig6)

    plt.show()

    main_end_time = time.time()
    total_elapsed = main_end_time - main_start_time
    print(f"\nTotal script execution time: {total_elapsed / 60:.2f} minutes.")
