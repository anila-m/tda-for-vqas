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