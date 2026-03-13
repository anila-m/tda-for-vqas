from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from datetime import datetime
import os
from pathlib import Path

import networkx as nx
import numpy as np
from orquestra.integrations.qulacs.simulator import QulacsSimulator

from ripser import ripser
from persim import plot_diagrams
from matplotlib import pyplot as plt

from src.qaoa.data_generation import prepare_cost_function
from src.qaoa.utils import generate_timestamp_str
from scipy.optimize import rosen
from src.qnn.qnn_cost_func import CostFunction
from src.utils.metrics import calc_TV_ASC_for_function
from src.utils.sampling_utils import get_latin_hypercube_samples
from orquestra.quantum.operators import get_pauli_strings, convert_dict_to_op
import ast

# Directories
BASE_DIR = Path(__file__).resolve().parent.parent
QAOA_LANDSCAPES = BASE_DIR / "experiment_results" / "QAOA" / "ripser_results"
QAOA_HAM_LANDSCAPES = BASE_DIR / "experiment_results" / "hamiltonian_experiment" / "ripser_results"
QNN_not_trans_LANDSCAPES = BASE_DIR / "experiment_results" / "QNN" / "not_transformed" / "ripser_results"
QNN_trans_LANDSCAPES = BASE_DIR / "experiment_results" / "QNN" / "transformed_50_100" / "ripser_results"
QAOA_RESULTS = BASE_DIR / "experiment_results" / "QAOA"
QAOA_HAM_RESULTS = BASE_DIR / "experiment_results" / "hamiltonian_experiment" 
QNN_not_trans_RESULTS = BASE_DIR / "experiment_results" / "QNN" / "not_transformed"
QNN_trans_RESULTS = BASE_DIR / "experiment_results" / "QNN" / "transformed_50_100"

max_number_of_ids = {"QAOA": 450, "QNN not transformed": 500, "QNN transformed": 500, "QAOA hamiltonian": 6}
landscape_dirs = {"QAOA": QAOA_LANDSCAPES, "QNN not transformed": QNN_not_trans_LANDSCAPES, "QNN transformed": QNN_trans_LANDSCAPES, "QAOA hamiltonian": QAOA_HAM_LANDSCAPES}
results_dirs = {"QAOA": QAOA_RESULTS, "QNN not transformed": QNN_not_trans_RESULTS, "QNN transformed": QNN_trans_RESULTS, "QAOA hamiltonian": QAOA_HAM_RESULTS}


backend = QulacsSimulator()
min_gamma=-np.pi
max_gamma=np.pi
min_beta=-np.pi/2
max_beta=np.pi/2


def compute_roughness_metrics_per_landscape(file, vqa_type):
    dict = json.load(open(file))
    landscape = np.asarray(dict["landscape"])
    
    if "QAOA" in vqa_type:
        if "hamiltonian" in dict:
            hamiltonian = convert_dict_to_op(dict["hamiltonian"])
        elif "new hamiltonian" in dict:
            hamiltonian = convert_dict_to_op(dict["new hamiltonian"])
        loss_func = prepare_cost_function(hamiltonian, backend)
        num_qubits = dict["num_qubits"]
        p = dict["p"]
        lower_left = np.concatenate((np.ones(p)*min_gamma, np.ones(p)*min_beta))
        upper_right = np.concatenate((np.ones(p)*max_gamma, np.ones(p)*max_beta))
        id = dict["config id"]
    elif "QNN" in vqa_type:
        num_qubits = 2
        unitary = np.array(ast.literal_eval(dict["unitary"]))
        x = np.array(ast.literal_eval(dict["databatch"]))
        loss_func = CostFunction(num_qubits=num_qubits, unitary=unitary, inputs=x)
        lower_left = np.zeros(6)
        upper_right = np.ones(6)*2*np.pi
        id = dict["qnn_id"]
    sample_points = landscape[:,:-1]
    
    asc, total_variation, gradient_summary, hessian_summary = calc_TV_ASC_for_function(loss_func, sample_points, lower_left, upper_right)
    metrics = {"average absolute scalar curvature": asc, 
               "total variation": total_variation, 
               "gradient": {"median": gradient_summary[0], 
                            "mean": gradient_summary[1], 
                            "min": gradient_summary[2], 
                            "max": gradient_summary[3]
                            },
                "hessian": {"median": hessian_summary[0], 
                            "mean": hessian_summary[1], 
                            "min": hessian_summary[2], 
                            "max": hessian_summary[3]
                            }
                }
    return id, metrics

 
    

def main_roughness_metrics_experiment(vqa_type):
    timestamp = generate_timestamp_str()
    assert vqa_type in max_number_of_ids.keys()
    max_ids = max_number_of_ids[vqa_type]
    landscape_directory = landscape_dirs[vqa_type]
    results_directory = results_dirs[vqa_type]
    landscape_directory.iterdir()

    config_dict = {"info": "Roughness metrics for loss landscapes different QAOA instances (keys correspond to QAOA IDs)", "source files": "resources\QAOA\landscapes"}
    cpu_count = os.cpu_count() 
    print(cpu_count)
    assert cpu_count is not None
    # for file in landscape_directory.iterdir():
    #     compute_roughness_metrics_per_landscape(file, vqa_type)
    #compute same-dimensional landscapes concurrently
    with ProcessPoolExecutor(max_workers=cpu_count) as exe:
        futures = [exe.submit(compute_roughness_metrics_per_landscape,file, vqa_type) for file in landscape_directory.iterdir()]            
        # await results & save them:
        for future in as_completed(futures):
            id, metrics = future.result()
            config_dict[id] = metrics
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[DONE] {now}: {id}")
    
    # save results
    METRICS_DIR = results_directory / "roughness_metrics_results"
    metrics_file = METRICS_DIR / f"QAOA_roughness_metrics_{timestamp}.json"
    metrics_file.write_text(config_dict)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[DONE] {now}: {vqa_type}")


if __name__ == "__main__":
    #main_roughness_metrics_experiment("QAOA")
    #test_gradient()
    #main_roughness_metrics_experiment(vqa_type="QAOA")
    main_roughness_metrics_experiment(vqa_type="QAOA hamiltonian")