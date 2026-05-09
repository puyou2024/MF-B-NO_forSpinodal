"""
Multi-Fidelity Bayesian Neural Operator (MF-B-NO) - Stage 2: High-Fidelity Training.
Leverages frozen Stage-1 Lo-Fi predictions and 3D SDF geometry to train a 
residual Bayesian DeepONet for experimental data alignment.

Structure: src/train_hifi.py
"""

import os
import re
import glob
import time
import copy
import joblib
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# from sklearn.model_selection import train_test_split # Removed as we use all data
from collections import defaultdict
import matplotlib.pyplot as plt
import random 

# --- Import Bayesian Torch ---
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_kl_loss

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"--- Global Seed set to {seed} ---")

# --- 1. LIGHTWEIGHT DATASET (Stores only Indices) ---
class SDFLookupDataset(Dataset):
    def __init__(self, sdf_indices, trunks, lofi_preds, targets):
        self.sdf_indices = torch.from_numpy(sdf_indices).long()
        self.trunks = torch.from_numpy(trunks).float()
        self.lofi_preds = torch.from_numpy(lofi_preds).float()
        self.targets = torch.from_numpy(targets).float()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.sdf_indices[idx], self.trunks[idx], self.lofi_preds[idx], self.targets[idx]

# --- 2. Model Definitions ---
class LoFiDeepONet(nn.Module):
    def __init__(self, branch_input_dim, trunk_input_dim, latent_dim=128):
        super(LoFiDeepONet, self).__init__()
        self.branch_net = nn.Sequential(
            nn.Linear(branch_input_dim, 32), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(32, 32), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(32, 128)
        )
        self.trunk_net = nn.Sequential(
            nn.Linear(trunk_input_dim, 32), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(32, 32), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(32, 128)
        )

    def forward(self, x_branch, x_trunk):
        branch_out = self.branch_net(x_branch)
        trunk_out = self.trunk_net(x_trunk)
        outputs = torch.sum(branch_out * trunk_out, dim=1, keepdim=True)
        return outputs

class CnnBranchNet(nn.Module):
    def __init__(self, latent_dim=128, sdf_grid_size=11):
        super(CnnBranchNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv3d(1, 4, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(4, 16, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.MaxPool3d(2)
        )
        conv_output_size = self._get_conv_output_size(sdf_grid_size)
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_output_size, 32), nn.ReLU(), nn.Dropout(0.0),
            nn.Linear(32, latent_dim)
        )
    def _get_conv_output_size(self, grid_size):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, grid_size, grid_size, grid_size)
            output = self.conv_layers(dummy_input)
            return int(np.prod(output.shape))
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

class SdfDeepONet(nn.Module):
    def __init__(self, trunk_input_dim=6, latent_dim=128, sdf_grid_size=11): 
        super(SdfDeepONet, self).__init__()
        self.branch_net = CnnBranchNet(latent_dim, sdf_grid_size)
        self.trunk_net = nn.Sequential(
            nn.Linear(in_features=trunk_input_dim, out_features=32), nn.ReLU(), 
            nn.Linear(in_features=32, out_features=32), nn.ReLU(),
            nn.Linear(in_features=32, out_features=latent_dim)
        )
    def forward(self, x_branch, x_trunk):
        branch_out = self.branch_net(x_branch)
        trunk_out = self.trunk_net(x_trunk)
        outputs = torch.sum(branch_out * trunk_out, dim=1, keepdim=True)
        return outputs

# --- 3. Utilities ---

# MODIFIED: Renamed parameter to 'folder_name' to support both training and test output folders
def predict_lofi_on_hifi(net_lofi, hifi_files, output_dir, strain_scaler, sdf_grid_size, sdf_dir, device, folder_name, num_samples=100):
    PREDICTION_OUTPUT_DIR = os.path.join(output_dir, folder_name)
    os.makedirs(PREDICTION_OUTPUT_DIR, exist_ok=True)
    
    net_lofi.train() 
    pattern = re.compile(r"([\d\.\-]+)_([\d\.\-]+)_([\d\.\-]+)_([xyz])")
    sdf_files = glob.glob(os.path.join(sdf_dir, '*.csv'))
    
    # Map Params -> SDF Filename
    sdf_map = {
        tuple(float(p) for p in re.search(r'([\d\.\-]+)_([\d\.\-]+)_([\d\.\-]+)', os.path.basename(f)).groups()): f
        for f in sdf_files if re.search(r'([\d\.\-]+)_([\d\.\-]+)_([\d\.\-]+)', os.path.basename(f))
    }

    processed_files = []
    
    print(f"   -> Processing {len(hifi_files)} files for LoFi prediction in {folder_name}...")

    for curve_path in hifi_files:
        filename = os.path.basename(curve_path)
        match = pattern.search(filename)
        if not match: continue

        lofi_params = tuple(float(p) for p in match.groups()[:3])
        if lofi_params not in sdf_map: continue
        
        direction = match.groups()[3].upper()
        dir_one_hot = {'X': [1.,0.,0.], 'Y': [0.,1.,0.], 'Z': [0.,0.,1.]}[direction]

        try:
            curve_df = pd.read_csv(curve_path, header=None, names=['phase', 'strain', 'original_stress'])
            num_points = len(curve_df)
            if num_points == 0: continue
            
            lofi_branch_np = np.array([lofi_params] * num_points, dtype=np.float32)
            lofi_branch_s = torch.from_numpy(lofi_branch_np / 90.0).to(device)
            
            strains = curve_df['strain'].values.reshape(-1, 1).astype(np.float32)
            strains_s = strain_scaler.transform(strains)
            phases = curve_df['phase'].values.reshape(-1, 1).astype(np.float32)
            dirs = np.array([dir_one_hot] * num_points, dtype=np.float32)
            
            trunk_np = np.hstack([strains_s, phases, dirs])
            trunk_tensor = torch.from_numpy(trunk_np.astype(np.float32)).to(device)
            
            with torch.no_grad():
                all_predictions = [net_lofi(lofi_branch_s, trunk_tensor).cpu().numpy().astype(np.float32) for _ in range(num_samples)]
            
            all_predictions = np.hstack(all_predictions)
            pred_mean = np.mean(all_predictions, axis=1).astype(np.float32)
            pred_std = np.std(all_predictions, axis=1).astype(np.float32)
            
            result_df = curve_df.copy()
            result_df['predicted_stress_mean'] = pred_mean
            result_df['predicted_stress_std'] = pred_std
            
            output_filename = f"prediction_uq_{filename}"
            output_path = os.path.join(PREDICTION_OUTPUT_DIR, output_filename)
            result_df.to_csv(output_path, index=False)
            processed_files.append(output_path)
            
        except Exception:
            continue
            
    return processed_files

# --- 4. Loading Data (MODIFIED: Returns Map) ---
def load_residual_data_optimized(file_list, strain_scaler, sdf_grid_size, sdf_dir):
    pattern = re.compile(r"prediction_uq_([\d\.\-]+)_([\d\.\-]+)_([\d\.\-]+)_([xyz])")
    
    # 1. Identify and Load Unique SDFs
    sdf_files = glob.glob(os.path.join(sdf_dir, '*.csv'))
    sdf_path_map = {
        tuple(float(p) for p in re.search(r'([\d\.\-]+)_([\d\.\-]+)_([\d\.\-]+)', os.path.basename(f)).groups()): f
        for f in sdf_files if re.search(r'([\d\.\-]+)_([\d\.\-]+)_([\d\.\-]+)', os.path.basename(f))
    }
    
    unique_params_in_split = set()
    valid_file_list = []
    
    for pred_path in file_list:
        match = pattern.search(os.path.basename(pred_path))
        if match:
            lofi_params = tuple(float(p) for p in match.groups()[:3])
            if lofi_params in sdf_path_map:
                unique_params_in_split.add(lofi_params)
                valid_file_list.append((pred_path, lofi_params, match.groups()[3].upper()))

    sorted_unique_params = sorted(list(unique_params_in_split))
    # --- FIX 1: Map created here ---
    param_to_idx_map = {param: i for i, param in enumerate(sorted_unique_params)}
    unique_sdfs_list = []
    
    print(f"   -> Loading {len(sorted_unique_params)} unique SDF volumes into memory...")
    for param in sorted_unique_params:
        sdf_path = sdf_path_map[param]
        sdf_df = pd.read_csv(sdf_path, header=0, sep=',')
        sdf_df = sdf_df.sort_values(by=['z', 'y', 'x'])
        sdf_vol = sdf_df['sdf'].values.astype(np.float32).reshape(1, sdf_grid_size, sdf_grid_size, sdf_grid_size)
        unique_sdfs_list.append(sdf_vol)
    
    if not unique_sdfs_list:
        # Return empty map too
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), {}

    unique_sdfs_tensor = np.stack(unique_sdfs_list, axis=0) 

    # 2. Process Data Points (Storing only INDICES)
    all_sdf_indices = []
    all_trunks = []
    all_lofi_preds = []
    all_targets = []
    
    for pred_path, lofi_params, direction_char in valid_file_list:
        try:
            pred_df = pd.read_csv(pred_path)
            if pred_df.empty: continue
            
            sdf_idx = param_to_idx_map[lofi_params]
            num_points = len(pred_df)
            
            all_sdf_indices.extend([sdf_idx] * num_points)
            
            strains = strain_scaler.transform(pred_df['strain'].values.reshape(-1, 1).astype(np.float32))
            phases = pred_df['phase'].values.reshape(-1, 1).astype(np.float32)
            lofi_preds = pred_df['predicted_stress_mean'].values.reshape(-1, 1).astype(np.float32)
            targets = pred_df['original_stress'].values.reshape(-1, 1).astype(np.float32)
            
            dir_one_hot = {'X': [1.,0.,0.], 'Y': [0.,1.,0.], 'Z': [0.,0.,1.]}[direction_char]
            dir_mat = np.tile(dir_one_hot, (num_points, 1)).astype(np.float32)
            
            trunk_block = np.concatenate([strains, phases, dir_mat, lofi_preds], axis=1)
            
            all_trunks.append(trunk_block)
            all_lofi_preds.append(lofi_preds)
            all_targets.append(targets)
            
        except Exception:
            continue

    # --- FIX 1: Return the map (last item) ---
    return (
        np.array(all_sdf_indices, dtype=np.int64),
        np.concatenate(all_trunks, axis=0) if all_trunks else np.array([], dtype=np.float32), 
        np.concatenate(all_lofi_preds, axis=0) if all_lofi_preds else np.array([], dtype=np.float32),
        np.concatenate(all_targets, axis=0) if all_targets else np.array([], dtype=np.float32),
        unique_sdfs_tensor,
        param_to_idx_map 
    )

# --- 5. Training Function (Unchanged logic, just args) ---
def run_bayesian_residual_training(model_hifi, train_loader, val_loader, 
                                   unique_sdfs_train, unique_sdfs_val, 
                                   epochs, lr, patience, model_save_path, device, kl_weight, scheduler_factor, scheduler_patience):
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model_hifi.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=scheduler_factor, patience=scheduler_patience)
    
    history = {
        'epoch': [], 'lr': [],
        'train_loss': [], 'train_kl': [], 'train_mse': [],
        'val_loss': [], 'val_kl': [], 'val_mse': []
    }
    best_val_loss = float('inf')
    
    PHYSICAL_BATCH_SIZE = train_loader.batch_size
    VIRTUAL_BATCH_SIZE = 256
    ACCUMULATION_STEPS = max(1, VIRTUAL_BATCH_SIZE // PHYSICAL_BATCH_SIZE)
    
    print(f"   -> Training Started (Acc Steps: {ACCUMULATION_STEPS}, GPU Lookup Active)...")
    
    for epoch in range(epochs):
        model_hifi.train()
        running_train_mse = 0.0
        running_train_loss = 0.0
        running_train_kl = 0.0
        num_samples = 0
        optimizer.zero_grad()
        
        for i, (sdf_idx, trunk, lofi_pred, target) in enumerate(train_loader):
            sdf_idx = sdf_idx.to(device)
            sdf = unique_sdfs_train[sdf_idx] # GPU Lookup
            trunk, lofi_pred, target = trunk.to(device), lofi_pred.to(device), target.to(device)
            
            pred_residual = model_hifi(sdf, trunk) 
            target_residual = target - lofi_pred 
            
            mse_loss = criterion(pred_residual, target_residual)
            kl = get_kl_loss(model_hifi)
            loss = mse_loss + kl_weight * kl
            
            loss_scaled = loss / ACCUMULATION_STEPS
            loss_scaled.backward()
            
            if (i + 1) % ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            batch_size = sdf.size(0)
            running_train_mse += mse_loss.item() * batch_size
            running_train_loss += loss.item() * batch_size
            running_train_kl += kl.item() * batch_size
            num_samples += batch_size
            
        epoch_train_mse = running_train_mse / max(1, num_samples)
        epoch_train_loss = running_train_loss / max(1, num_samples)
        epoch_train_kl = running_train_kl / max(1, num_samples)

        model_hifi.eval()
        kl_val = get_kl_loss(model_hifi).item()
        
        running_val_mse = 0.0
        num_val_samples = 0
        with torch.no_grad():
            for sdf_idx_v, trunk_v, lofi_pred_v, target_v in val_loader:
                sdf_idx_v = sdf_idx_v.to(device)
                sdf_v = unique_sdfs_val[sdf_idx_v] # GPU Lookup
                trunk_v, lofi_pred_v, target_v = trunk_v.to(device), lofi_pred_v.to(device), target_v.to(device)
                
                pred_residual_val = model_hifi(sdf_v, trunk_v)
                final_pred = lofi_pred_v + pred_residual_val
                
                mse_loss = criterion(final_pred, target_v)
                running_val_mse += mse_loss.item() * sdf_v.size(0)
                num_val_samples += sdf_v.size(0)
        
        epoch_val_mse = running_val_mse / max(1, num_val_samples)
        epoch_val_loss = epoch_val_mse + kl_weight * kl_val
        
        history['epoch'].append(epoch + 1)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        history['train_loss'].append(epoch_train_loss)
        history['train_kl'].append(epoch_train_kl)
        history['train_mse'].append(epoch_train_mse)
        history['val_loss'].append(epoch_val_loss)
        history['val_kl'].append(kl_val)
        history['val_mse'].append(epoch_val_mse)
        
        if (epoch + 1) % 10 == 0:
            print(f"      Epoch {epoch+1}/{epochs} | Val MSE: {epoch_val_mse:.6f}")
        
        scheduler.step(epoch_val_loss)
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model_hifi.state_dict(), model_save_path)
            
    return model_hifi, history

# --- 6. Prediction Function (MODIFIED: Uses Memory Lookup) ---
def save_bayesian_validation_predictions(net_hifi, val_files_processed, unique_sdfs_gpu, param_to_idx_map, output_dir, output_subdir, strain_scaler, device, num_samples=1000):
    prediction_dir = os.path.join(output_dir, output_subdir) 
    os.makedirs(prediction_dir, exist_ok=True)
    
    net_hifi.train() 

    pattern = re.compile(r"prediction_uq_([\d\.\-]+)_([\d\.\-]+)_([\d\.\-]+)_([xyz])")
    
    for pred_path in val_files_processed:
        filename = os.path.basename(pred_path)
        match = pattern.search(filename)
        if not match: continue
        
        lofi_params = tuple(float(p) for p in match.groups()[:3])
        
        if lofi_params not in param_to_idx_map: 
            print(f"Skipping {filename}: Parameters not found in memory map.")
            continue
        
        sdf_idx = param_to_idx_map[lofi_params]

        try:
            pred_df = pd.read_csv(pred_path)
            num_points = len(pred_df)
            if num_points == 0: continue
            
            sdf_vol = unique_sdfs_gpu[sdf_idx] # Shape: (GRID, GRID, GRID)
            sdf_tensor = sdf_vol.unsqueeze(0).repeat(num_points, 1, 1, 1, 1)

            strains = strain_scaler.transform(pred_df['strain'].values.reshape(-1, 1).astype(np.float32))
            phases = pred_df['phase'].values.reshape(-1, 1).astype(np.float32)
            
            direction_char = match.groups()[3].upper()
            dir_one_hot = {'X': [1.,0.,0.], 'Y': [0.,1.,0.], 'Z': [0.,0.,1.]}[direction_char]
            dirs = np.array([dir_one_hot] * num_points, dtype=np.float32)
            lofi_pred_mean = pred_df['predicted_stress_mean'].values.reshape(-1, 1).astype(np.float32)

            trunk_np = np.hstack([strains, phases, dirs, lofi_pred_mean]) 
            trunk_tensor = torch.from_numpy(trunk_np.astype(np.float32)).to(device)
            lofi_pred_tensor = torch.from_numpy(lofi_pred_mean.astype(np.float32)).to(device)
            
            all_predictions = []
            with torch.no_grad():
                for _ in range(num_samples):
                    pred_residual_sample = net_hifi(sdf_tensor, trunk_tensor)
                    final_prediction_sample = lofi_pred_tensor + pred_residual_sample
                    all_predictions.append(final_prediction_sample)
            
            predictions_stacked = torch.stack(all_predictions, dim=0)
            pred_means = torch.mean(predictions_stacked, dim=0).cpu().numpy().flatten().astype(np.float32)
            pred_stds = torch.std(predictions_stacked, dim=0).cpu().numpy().flatten().astype(np.float32)
            
            final_df = pred_df.copy()
            final_df['hifi_predicted_mean'] = pred_means
            final_df['hifi_predicted_std'] = pred_stds
            
            output_filename = f"{filename.replace('lofi_pred_', '')}"
            final_df.to_csv(os.path.join(prediction_dir, output_filename), index=False)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
    print(f"   -> Predictions saved to: {prediction_dir}")

# --- 7. Main Execution ---
if __name__ == "__main__":
    start_time_global = time.time()
    SEEDS = [1, 2, 3, 4, 5] 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Using device: {device} ---")

    # --- PATH SETTINGS ---
    BASE_DIR = ".."
    HIFI_CURVE_DIR = os.path.join(BASE_DIR, "data/hifi/curves/train")
    HIFI_SDF_DIR = os.path.join(BASE_DIR, "data/hifi/sdf/train")
    
    # Test Data (NEW)
    # WARNING: Ensure this directory exists and contains the curve CSVs for the test cases
    HIFI_CURVE_TEST_DIR = os.path.join(BASE_DIR, "data/hifi/curves/test")
    HIFI_SDF_TEST_DIR = os.path.join(BASE_DIR, "data/hifi/sdf/test")
    
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    LOFI_MODEL_PATH = os.path.join(MODEL_DIR, "lofi_model.pth")
    STRAIN_SCALER_PATH = os.path.join(MODEL_DIR, "strain_scaler.gz")
    BASE_OUTPUT_DIR = os.path.join(BASE_DIR, "outputs/hifi_results")

    SDF_GRID_SIZE = 41
    # HF_VAL_SPLIT = 0.2 # Removed split
    BATCH_SIZE = 16
    LATENT_DIM = 128
    KL_WEIGHT = 1e-5
    EPOCHS = 500
    LR = 1e-3
    PATIENCE = 5000
    SCHEDULER_FACTOR = 0.3
    SCHEDULER_PATIENCE = 50

    const_bnn_prior_parameters = {
        "prior_mu": 0.0, "prior_sigma": 1.0, "posterior_mu_init": 0.0,
        "posterior_rho_init": -5.0, "type": "Flipout", "moped_enable": False,
    }

    strain_scaler = joblib.load(STRAIN_SCALER_PATH)

    for seed in SEEDS:
        print(f"\n\n{'='*50}\n   RUNNING EXPERIMENT WITH SEED: {seed}\n{'='*50}")
        set_seed(seed)
        seed_output_dir = os.path.join(BASE_OUTPUT_DIR, f"seed_{seed}")
        os.makedirs(seed_output_dir, exist_ok=True)
        HIFI_MODEL_PATH = os.path.join(seed_output_dir, 'bayesian_hifi_augmented_residual_model.pth')
        
        # --- Stage 1: Lo-Fi Preds ---
        print("\n--- STAGE 1: Generating Lo-Fi Augmented Inputs ---")
        net_lofi_frozen = LoFiDeepONet(branch_input_dim=3, trunk_input_dim=5, latent_dim=LATENT_DIM)
        dnn_to_bnn(net_lofi_frozen, const_bnn_prior_parameters) 
        net_lofi_frozen.load_state_dict(torch.load(LOFI_MODEL_PATH, map_location=device, weights_only=True))
        net_lofi_frozen.to(device)
        net_lofi_frozen.eval()
        
        # 1.1 Process TRAIN
        hf_train_files_all = glob.glob(os.path.join(HIFI_CURVE_DIR, '*.csv'))
        processed_train_paths = predict_lofi_on_hifi(
            net_lofi_frozen, hf_train_files_all, seed_output_dir, strain_scaler, SDF_GRID_SIZE, HIFI_SDF_DIR, device,
            folder_name='lofi_predictions_on_hifi'
        )

        # 1.2 Process TEST
        hf_test_files_all = glob.glob(os.path.join(HIFI_CURVE_TEST_DIR, '*.csv'))
        processed_test_paths = []
        if hf_test_files_all:
            processed_test_paths = predict_lofi_on_hifi(
                net_lofi_frozen, hf_test_files_all, seed_output_dir, strain_scaler, SDF_GRID_SIZE, HIFI_SDF_TEST_DIR, device,
                folder_name='lofi_predictions_on_test'
            )
        else:
            raise ValueError(f"No test files found in {HIFI_CURVE_TEST_DIR}. Cannot track Test Loss.")
        
        del net_lofi_frozen
        torch.cuda.empty_cache()
        
        # --- Stage 2: Load Data ---
        print("\n--- STAGE 2: Loading Data & Training ---")
        
        # Load Train
        idx_train, trk_train, lofi_train, tgt_train, unique_sdf_train_np, map_train = load_residual_data_optimized(processed_train_paths, strain_scaler, SDF_GRID_SIZE, HIFI_SDF_DIR)
        unique_sdf_train_gpu = torch.from_numpy(unique_sdf_train_np).float().to(device)
        
        # Load Test (MOVED UP)
        idx_test, trk_test, lofi_test, tgt_test, unique_sdf_test_np, map_test = load_residual_data_optimized(processed_test_paths, strain_scaler, SDF_GRID_SIZE, HIFI_SDF_TEST_DIR)
        unique_sdf_test_gpu = torch.from_numpy(unique_sdf_test_np).float().to(device)
        
        train_dataset = SDFLookupDataset(idx_train, trk_train, lofi_train, tgt_train)
        test_dataset = SDFLookupDataset(idx_test, trk_test, lofi_test, tgt_test)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        # CHANGE: val_loader uses test_dataset
        val_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0) 

        # --- Stage 3: Train ---
        # Note: trunk_input_dim=6 here as per your original file
        net_hifi = SdfDeepONet(trunk_input_dim=6, latent_dim=LATENT_DIM, sdf_grid_size=SDF_GRID_SIZE)
        dnn_to_bnn(net_hifi, const_bnn_prior_parameters)
        net_hifi.to(device)
        
        # CHANGE: Pass unique_sdf_test_gpu
        net_hifi, hifi_history = run_bayesian_residual_training(
            net_hifi, train_loader, val_loader, 
            unique_sdf_train_gpu, unique_sdf_test_gpu, 
            epochs=EPOCHS, lr=LR, patience=PATIENCE, model_save_path=HIFI_MODEL_PATH, 
            device=device, kl_weight=KL_WEIGHT, 
            scheduler_factor=SCHEDULER_FACTOR, scheduler_patience=SCHEDULER_PATIENCE
        )

        # --- Save History ---
        print("   -> Saving Training History CSV...")
        history_df = pd.DataFrame(hifi_history)
        history_df.to_csv(os.path.join(seed_output_dir, 'training_history.csv'), index=False)

        # --- Plotting ---
        print("   -> Generating Plots...")
        plt.figure(figsize=(16, 10))
        plt.suptitle(f'Bayesian Training Metrics (Seed {seed}) - TEST LOSS TRACKING', fontsize=16)
        plt.subplot(2, 2, 1)
        plt.plot(hifi_history['train_loss'], label='Train', color='blue')
        plt.plot(hifi_history['val_loss'], label='Test', color='orange', linestyle='--')
        plt.title('Total Loss'); plt.yscale('log'); plt.legend(); plt.grid(True, alpha=0.3)
        # ... (Other subplots) ...
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(seed_output_dir, 'training_metrics.png'))
        plt.close()

        # --- Stage 4: Prediction ---
        print("\n--- STAGE 3: Saving Test Predictions ---")
        save_bayesian_validation_predictions(
            net_hifi, processed_test_paths, unique_sdf_test_gpu, map_test, 
            seed_output_dir, 'final_test_preds', strain_scaler, device
        )

        # --- Cleanup ---
        del net_hifi, train_loader, val_loader, train_dataset, test_dataset
        del idx_train, trk_train, lofi_train, tgt_train
        del unique_sdf_train_gpu, unique_sdf_test_gpu
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\nTotal Execution Time: {(time.time() - start_time_global)/3600:.2f} hours")