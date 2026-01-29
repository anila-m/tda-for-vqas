import os
import json
import numpy as np
from orquestra.quantum.operators import convert_op_to_dict

def save_qnn_landscape(landscape, meta_data, directory, file_name):
    # check if directory exists and create it if not
    if not os.path.exists(directory):
        os.mkdir(directory)

    min = float(np.min(landscape[:, -1]))
    max = float(np.max(landscape[:, -1]))
    dict = meta_data.copy()

    dict.update({
        "min cost": min,
        "max cost": max,
        "landscape": landscape.tolist()
    })
    with open(os.path.join(directory, file_name), "w") as f:
        json.dump(dict, f, indent=4)

def save_landscape(landscape, meta_data, directory, file_name, id):
    # check if directory exists and create it if not
    if not os.path.exists(directory):
        os.mkdir(directory)

    min = np.min(landscape[:, -1])
    max = np.max(landscape[:, -1])
    dict = meta_data.copy()
    if "hamiltonian" in dict.keys():
        dict["hamiltonian"] = convert_op_to_dict(dict["hamiltonian"])
    dict.update({
        "config id": id,
        "min cost": min,
        "max cost": max,
        "landscape": landscape.tolist()
    })
    with open(os.path.join(directory, file_name), "w") as f:
        json.dump(dict, f, indent=4)



def save_persistence_diagrams(ripser_result, N, ham_file_label, sample_points_file_label, timestamp, p, id=""):
    dir = os.path.join(os.path.join(os.path.join(os.path.join(os.getcwd(), f"experiment_results"),f"Initial_QAOA_tests"),"H2"),f"persistence_{timestamp}")
    if not os.path.exists(dir):
        os.mkdir(dir)
    file_name = f"persistence_N_{N}_p_{p}.json"
    
    dict = {
        "config id": id,
        "hamiltonian": ham_file_label,
        "sample_points": sample_points_file_label,
        "timestamp": timestamp,
        "number_sample_points": N,
        "P": p,
        "persistence diagram": {}
    }
    # prep and save ripser result 
    d = [v.tolist() for v in ripser_result["dgms"]]
    c = [[v.tolist() for v in l] for l in ripser_result["cocycles"]]
    ripser_dict = {
        "dgms" : d,
        "cocycles": c,
        "num_edges": ripser_result["num_edges"],
        #"distance_matrix": ripser_result["dperm2all"].tolist(), #too large to save
        "r_cover": ripser_result["r_cover"]
    }
    if ripser_result["idx_perm"] is not None:
        ripser_dict["idx_perm"] = ripser_result["idx_perm"].tolist()

    dict["persistence diagram"] = ripser_dict
    with open(os.path.join(dir, file_name), "w") as f:
        json.dump(dict, f, indent=4)


def ripser_list_to_giotto(all_ripser_dgms):
    """
    Converts a list of Ripser persistence diagrams into one 3D array for Giotto-TDA.
    
    :param all_ripser_dgms: List of 'dgms' lists from Ripser.
    """
    # Determine the max points found in each homology dimension
    num_homology_levels = len(all_ripser_dgms[0])
    max_pts_per_dim = [0] * num_homology_levels
    
    for dgms in all_ripser_dgms:
        for i, dgm in enumerate(dgms):
            # Filter infinite points before counting
            clean_len = len(dgm[np.isfinite(dgm).all(axis=1)])
            if clean_len > max_pts_per_dim[i]:
                max_pts_per_dim[i] = clean_len

    # process and pad each diagram
    processed_samples = []
    for dgms in all_ripser_dgms:
        sample_parts = []
        for i, dgm in enumerate(dgms):
            clean_dgm = dgm[np.isfinite(dgm).all(axis=1)]
            n_pts = len(clean_dgm)
            
            # Add homology dimension column
            dim_col = np.full((n_pts, 1), i)
            tagged_dgm = np.hstack([clean_dgm, dim_col])
            
            # Pad this dimension to its relative maximum
            if n_pts < max_pts_per_dim[i]:
                padding_size = max_pts_per_dim[i] - n_pts
                # Pad with (0, 0, i) 
                padding = np.zeros((padding_size, 3))
                padding[:, 2] = i 
                tagged_dgm = np.vstack([tagged_dgm, padding])
            
            sample_parts.append(tagged_dgm)
        
        processed_samples.append(np.vstack(sample_parts))

    return np.array(processed_samples)