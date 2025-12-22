import os
import json
import numpy as np

def save_landscape(landscape, ham_file_label, sample_points_file_label, timestamp, p, id):
    directory_name = os.path.join(
        os.path.join(os.getcwd(), f"results\landscapes_{timestamp}")
    )
    # check if directory exists and create it if not

    if not os.path.exists(os.path.join(os.getcwd(), "results")):
        os.mkdir(os.path.join(os.getcwd(), "results"))
    if not os.path.exists(directory_name):
        os.mkdir(directory_name)
    file_name = f"landscape_{ham_file_label}_{sample_points_file_label}_p_{p}.json"
    min = np.min(landscape[:, -1])
    max = np.max(landscape[:, -1])
    dict = {
        "config ID": id,
        "hamiltonian": ham_file_label,
        "sample_points": sample_points_file_label,
        "timestamp": timestamp,
        "P": p,
        "min cost": min,
        "max cost": max,
        "landscape": landscape.tolist()
    }
    with open(os.path.join(directory_name, file_name), "w") as f:
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