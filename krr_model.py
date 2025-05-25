# -*- coding: utf-8 -*-
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import os
model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
print("Running on", os.uname())
print("SLURM_CPUS_PER_TASK:", os.environ.get("SLURM_CPUS_PER_TASK"))


ATOMIC_MASSES = {
    'H': 1.00784, 'C': 12.00, 'N': 14.0067, 'O': 15.999, 'F': 18.998,
    'P': 30.973, 'S': 32.06, 'Cl': 35.45, 'Br': 79.904, 'I': 126.90,
}

POWER_SERIES_TERMS = [-1, -3, -5, -6, -7, -9, -12] 

TIMESTEP = 0.5 
TARGET_TRAINING_TIME_FS = 1000.0 
REQUIRED_TRAINING_FRAMES = int(TARGET_TRAINING_TIME_FS / TIMESTEP) + 1
TARGET_PREDICTION_TIME_FS = 1000.0
TOTAL_OUTPUT_FRAMES = int(TARGET_PREDICTION_TIME_FS / TIMESTEP) + 1 
def parse_xyz_trajectory(filepath):
    """Parses an XYZ trajectory file and returns a list of (atom_types, coordinates) for each frame."""
    all_frames_data = []
    with open(filepath, 'r') as f:
        while True:
            line1 = f.readline()
            if not line1: 
                break
            try:
                num_atoms = int(line1.strip())
            except ValueError:
                print(f"Error: Could not parse number of atoms from line: '{line1.strip()}' in {filepath}.")
                break 
            
            f.readline() 
            
            atom_types = []
            coords = []
            for _ in range(num_atoms):
                parts = f.readline().strip().split()
                if len(parts) < 4:
                    print(f"Warning: Incomplete atom line: '{' '.join(parts)}' in frame. Skipping frame.")
                    atom_types = [] 
                    coords = []
                    
                    for _ in range(num_atoms - len(atom_types)): 
                        f.readline()
                    break 
                atom_types.append(parts[0])
                coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
            
            if atom_types and coords: 
                all_frames_data.append((atom_types, np.array(coords)))
    return all_frames_data


def calculate_displacement(r_i, r_j):
    """Calculates the displacement vector from atom j to atom i."""
    return r_i - r_j

def calculate_encoded_force(r_prev, r_curr, r_next, mass):
    """
    Calculates the 'encoded force' (mass * acceleration * dt^2) from positions.
    This is equivalent to Force * dt^2 based on the Verlet algorithm.
    F * dt^2 = mass * (r_next - 2*r_curr + r_prev)
    """
    return mass * (r_next - 2 * r_curr + r_prev)

def calculate_force_basis(r_ij_vec, power_terms):
    """
    Calculates the basis terms for the force interaction (F_ij = C * r_ij^(s-1) * unit_r_ij_vec).
    """
    r_ij_norm = np.linalg.norm(r_ij_vec)
    
    if r_ij_norm < 1e-8: 
        return {s: np.zeros(3) for s in power_terms}
    
    unit_r_ij_vec = r_ij_vec / r_ij_norm
    basis_terms = {}
    for s in power_terms:
 
        if s < 0 and r_ij_norm < 0.05:
            capped_r_ij_norm = 0.05
            basis_terms[s] = (capped_r_ij_norm**(s-1)) * unit_r_ij_vec
        else:
            basis_terms[s] = (r_ij_norm**(s-1)) * unit_r_ij_vec
    return basis_terms

# --- Data Preprocessing ---
def preprocess_trajectory(trajectory_data, power_terms, atom_masses_map):
    """
    Preprocesses trajectory data to generate features (X) and labels (Y) for ML model.
    X: Flattened array of force basis contributions for all atoms across all frames.
    Y: Flattened array of encoded forces (Force * dt^2) for all atoms across all frames.
    """
    num_frames = len(trajectory_data)
    num_atoms = len(trajectory_data[0][0])
    
    all_X_features_flat = []
    all_Y_labels_flat = []

    print(f"Starting pre-processing for {num_atoms} atoms across {num_frames} frames...")


    for t_idx in range(1, num_frames - 1):
        r_prev_frame = trajectory_data[t_idx - 1][1]
        r_curr_frame = trajectory_data[t_idx][1]
        r_next_frame = trajectory_data[t_idx + 1][1]

        for i in range(num_atoms):
            encoded_force_i = calculate_encoded_force(
                r_prev_frame[i], r_curr_frame[i], r_next_frame[i], atom_masses_map[i]
            )
            all_Y_labels_flat.append(encoded_force_i.tolist())

            current_atom_i_features = []
            for j in range(num_atoms):
                if i == j: 
                    current_atom_i_features.extend([0.0] * len(power_terms) * 3)
                    continue
                
                r_ij_vec = calculate_displacement(r_curr_frame[i], r_curr_frame[j])
                basis_terms_dict = calculate_force_basis(r_ij_vec, power_terms)
                
                for s in power_terms:
                    basis_vec = basis_terms_dict[s]
                    current_atom_i_features.extend(basis_vec.tolist())
            all_X_features_flat.append(current_atom_i_features)

    X_flat = np.array(all_X_features_flat)
    Y_flat = np.array(all_Y_labels_flat)

    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_flat)
    
    print(f"Pre-processing complete. Generated data for {X_scaled.shape[0]} effective atom-frames.")
    return X_scaled, Y_flat, scaler_X


def train_krr_model(X_train, Y_train):
    """Trains global KRR models for each coordinate (x,y,z) using GridSearchCV."""
    print("Starting Kernel Ridge Regression training...")

  
    param_grid = {
        'alpha': np.logspace(-4, -1, 4), 
        'gamma': np.logspace(-2, 0, 3)  
    }
    
    base_krr_model = KernelRidge(kernel='rbf')


    cv_splitter = KFold(n_splits=3, shuffle=True, random_state=42) 
    
  
    N_JOBS_FOR_GRIDSEARCH = 1
    
    krr_x = GridSearchCV(base_krr_model, param_grid, cv=cv_splitter, scoring='neg_mean_squared_error', n_jobs=N_JOBS_FOR_GRIDSEARCH, verbose=0)
    krr_y = GridSearchCV(base_krr_model, param_grid, cv=cv_splitter, scoring='neg_mean_squared_error', n_jobs=N_JOBS_FOR_GRIDSEARCH, verbose=0)
    krr_z = GridSearchCV(base_krr_model, param_grid, cv=cv_splitter, scoring='neg_mean_squared_error', n_jobs=N_JOBS_FOR_GRIDSEARCH, verbose=0)

    print("Tuning KRR for X component (this might take a while due to reduced n_jobs)...")
    krr_x.fit(X_train, Y_train[:, 0])
    print(f"Best KRR parameters for X: {krr_x.best_params_}, Best MSE: {-krr_x.best_score_:.4e}")

    print("Tuning KRR for Y component...")
    krr_y.fit(X_train, Y_train[:, 1])
    print(f"Best KRR parameters for Y: {krr_y.best_params_}, Best MSE: {-krr_y.best_score_:.4e}")

    print("Tuning KRR for Z component...")
    krr_z.fit(X_train, Y_train[:, 2])
    print(f"Best KRR parameters for Z: {krr_z.best_params_}, Best MSE: {-krr_z.best_score_:.4e}")

    learned_models_global = {
        'x': krr_x.best_estimator_,
        'y': krr_y.best_estimator_,
        'z': krr_z.best_estimator_,
        '__metadata__': {
            'power_terms': POWER_SERIES_TERMS,
        }
    }
    
    print("Kernel Ridge Regression training complete.")
    return learned_models_global


def predict_forces_from_model(coords, learned_models_global, power_terms_order, scaler_X, num_atoms):
    """
    Predicts the total force * dt^2 on each atom at a given set of coordinates using the trained KRR models.
    """
    all_atom_features = []
    for i in range(num_atoms):
        current_atom_i_features = []
        for j in range(num_atoms):
            if i == j:
                current_atom_i_features.extend([0.0] * len(power_terms_order) * 3)
                continue
            
            r_ij_vec = calculate_displacement(coords[i], coords[j])
            basis_terms_dict = calculate_force_basis(r_ij_vec, power_terms_order)
            
            for s_term in power_terms_order:
                basis_vec = basis_terms_dict[s_term]
                current_atom_i_features.extend(basis_vec.tolist())
        all_atom_features.append(current_atom_i_features)

    X_predict_flat = np.array(all_atom_features)
    X_predict_scaled = scaler_X.transform(X_predict_flat)

    predicted_force_dt2 = np.zeros_like(coords)
    predicted_force_dt2[:, 0] = learned_models_global['x'].predict(X_predict_scaled)
    predicted_force_dt2[:, 1] = learned_models_global['y'].predict(X_predict_scaled)
    predicted_force_dt2[:, 2] = learned_models_global['z'].predict(X_predict_scaled)

    return predicted_force_dt2

def predict_trajectory(initial_coords_frame1, initial_coords_frame2, learned_models_global,
                         atom_masses_map, power_terms, timestep, num_prediction_steps,
                         atom_types_list_global, scaler_X):
    """
    Predicts molecular trajectory using the Verlet algorithm and learned force model.
    """
    num_atoms = len(atom_masses_map)
    predicted_trajectory = []
    
    power_terms_order = learned_models_global['__metadata__']['power_terms']

    r_prev = np.copy(initial_coords_frame1)
    r_curr = np.copy(initial_coords_frame2)

    predicted_trajectory.append((atom_types_list_global, r_prev))
    predicted_trajectory.append((atom_types_list_global, r_curr))

    print(f"\nStarting trajectory prediction for {num_prediction_steps} steps (0 fs to {num_prediction_steps * timestep} fs)...")

    for step in range(num_prediction_steps):
        if (step + 1) % 200 == 0 or (step + 1) == num_prediction_steps: 
            print(f"Predicting step {step + 1}/{num_prediction_steps} (Time: {(step + 1) * timestep:.2f} fs)")


        predicted_force_dt2 = predict_forces_from_model(r_curr, learned_models_global, 
                                                        power_terms_order, scaler_X, num_atoms)
        
        mass_vector = np.array([atom_masses_map[i] for i in range(num_atoms)]).reshape(-1, 1)
        mass_vector[mass_vector == 0] = 1.0 

 
        acceleration_dt2 = predicted_force_dt2 / mass_vector
        

        r_next = 2 * r_curr - r_prev + acceleration_dt2

        predicted_trajectory.append((atom_types_list_global, r_next))
        
 
        r_prev = r_curr
        r_curr = r_next
        
    print("Trajectory prediction complete.")
    return predicted_trajectory


def calculate_rmsd(coords1, coords2):
    """Calculates the Root Mean Square Deviation (RMSD) between two sets of coordinates."""
    diff = coords1 - coords2
    return np.sqrt(np.sum(diff**2) / coords1.shape[0])


def write_trajectory_xyz(filename, trajectory_data, initial_time, timestep):
    """Writes trajectory data to an XYZ file."""
    with open(filename, 'w') as f:
        for i, (atom_types, coords) in enumerate(trajectory_data):
            f.write(f"{len(atom_types)}\n")
            current_time = initial_time + i * timestep
            f.write(f"t={current_time:.5f} fs\n")
            for j, atom_type in enumerate(atom_types):
                f.write(f"{atom_type} {coords[j, 0]:.9f} {coords[j, 1]:.9f} {coords[j, 2]:.9f}\n")


if __name__ == "__main__":
    trajectory_xyz_file = 'trajectory.xyz' 
    ml_predicted_output_file = 'trajml_predicted_0_1000fs.xyz'

    if not os.path.exists(trajectory_xyz_file):
        print(f"Error: The reference trajectory file '{trajectory_xyz_file}' was not found.")
        print("Please ensure 'trajectory.xyz' is in the same directory as the script and contains enough frames.")
        exit()

    print(f"Loading full trajectory from {trajectory_xyz_file} for training and reference...")
    full_trajectory_data = parse_xyz_trajectory(trajectory_xyz_file)
    print(f"Loaded {len(full_trajectory_data)} frames from {trajectory_xyz_file}.")


    if len(full_trajectory_data) < REQUIRED_TRAINING_FRAMES:
        print(f"Error: The provided trajectory has only {len(full_trajectory_data)} frames.")
        print(f"It must contain at least {REQUIRED_TRAINING_FRAMES} frames ({TARGET_TRAINING_TIME_FS} fs) for training.")
        print("Please provide a longer trajectory file.")
        exit()


    training_data_subset = full_trajectory_data[:REQUIRED_TRAINING_FRAMES]
    print(f"Using {len(training_data_subset)} frames (0 to {TARGET_TRAINING_TIME_FS} fs) for training the force model.")

    if len(training_data_subset) < 3:
        print("Error: Training data subset must contain at least 3 frames for force calculation (t-dt, t, t+dt).")
        exit()

    atom_types_list_global = training_data_subset[0][0]
    atom_masses_map = {i: ATOMIC_MASSES[atom_type] for i, atom_type in enumerate(atom_types_list_global)}

    X_train_scaled, Y_train_labels, scaler_X = preprocess_trajectory(
        training_data_subset, POWER_SERIES_TERMS, atom_masses_map
    )

    learned_models_global = train_krr_model(X_train_scaled, Y_train_labels)

    initial_coords_for_prediction_frame1 = training_data_subset[0][1]
    initial_coords_for_prediction_frame2 = training_data_subset[1][1]
    num_prediction_steps_needed = TOTAL_OUTPUT_FRAMES - 2 

    print(f"\nModel trained. Now predicting trajectory from 0 fs to {TARGET_PREDICTION_TIME_FS} fs (total {TOTAL_OUTPUT_FRAMES} frames).")
    predicted_trajectory = predict_trajectory(
        initial_coords_for_prediction_frame1,
        initial_coords_for_prediction_frame2,
        learned_models_global,
        atom_masses_map,
        POWER_SERIES_TERMS,
        TIMESTEP,
        num_prediction_steps_needed,
        atom_types_list_global,
        scaler_X
    )

    write_trajectory_xyz(ml_predicted_output_file, predicted_trajectory, 0.0, TIMESTEP)
    print(f"\nPredicted trajectory saved to {ml_predicted_output_file}")

    print(f"\n--- Comparing Predicted Trajectory with AIMD Reference ({trajectory_xyz_file}) ---")

    if len(full_trajectory_data) < TOTAL_OUTPUT_FRAMES:
        print(f"Warning: Reference AIMD trajectory '{trajectory_xyz_file}' is shorter than the predicted trajectory.")
        print(f"Comparison will be limited to {len(full_trajectory_data)} frames.")
        comparison_frames = min(len(predicted_trajectory), len(full_trajectory_data))
    else:
        comparison_frames = TOTAL_OUTPUT_FRAMES
    
    rmsd_values = []
    for i in range(comparison_frames):
        try:
            predicted_coords = predicted_trajectory[i][1]
            aimd_coords = full_trajectory_data[i][1]
            
            if predicted_coords.shape[0] != aimd_coords.shape[0]:
                print(f"Warning: Mismatch in number of atoms at frame {i}. Skipping RMSD for this frame.")
                continue

            rmsd = calculate_rmsd(predicted_coords, aimd_coords)
            rmsd_values.append(rmsd)
        except IndexError:
            print(f"Error: Could not retrieve frame {i} for comparison. Trajectory lengths might differ.")
            break

    if rmsd_values:
        avg_rmsd = np.mean(rmsd_values)
        max_rmsd = np.max(rmsd_values)
        print(f"\nAverage RMSD (0 - {(comparison_frames-1)*TIMESTEP:.2f} fs): {avg_rmsd:.4f} Angstrom")
        print(f"Maximum RMSD (0 - {(comparison_frames-1)*TIMESTEP:.2f} fs): {max_rmsd:.4f} Angstrom")
    else:
        print("No RMSD values could be calculated for comparison.")

    print("\n--- Simulation Complete ---")
    