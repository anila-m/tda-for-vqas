from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from datetime import datetime
import math
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
from src.utils.metrics import calc_TV_ASC_for_function, calc_TV_FD_ASC_for_grid_landscape
from src.utils.sampling_utils import get_latin_hypercube_samples, get_grid_landscapes_from_stepsize
from orquestra.quantum.operators import get_pauli_strings, convert_dict_to_op
import ast

# Directories
BASE_DIR = Path(__file__).resolve().parent.parent
QAOA_LANDSCAPES = BASE_DIR / "experiment_results" / "QAOA" / "ripser_results"
QAOA_HAM_LANDSCAPES = BASE_DIR / "experiment_results" / "hamiltonian_experiment" / "ripser_results"
QAOA_HAM_LANDSCAPES_scaled_50 = BASE_DIR / "experiment_results" / "hamiltonian_experiment" / "scaled50" / "ripser_results"
QNN_not_trans_LANDSCAPES = BASE_DIR / "experiment_results" / "QNN" / "not_transformed" / "ripser_results"
QNN_trans_LANDSCAPES = BASE_DIR / "experiment_results" / "QNN" / "transformed_50_100" / "ripser_results"
QAOA_RESULTS = BASE_DIR / "experiment_results" / "QAOA"
QAOA_HAM_RESULTS = BASE_DIR / "experiment_results" / "hamiltonian_experiment" 
QNN_not_trans_RESULTS = BASE_DIR / "experiment_results" / "QNN" / "not_transformed"
QNN_trans_RESULTS = BASE_DIR / "experiment_results" / "QNN" / "transformed_50_100"

max_number_of_ids = {"QAOA": 450, "QNN not transformed": 500, "QNN transformed": 500, "QAOA hamiltonian": 6}
landscape_dirs = {"QAOA": QAOA_LANDSCAPES, "QNN not transformed": QNN_not_trans_LANDSCAPES, "QNN transformed": QNN_trans_LANDSCAPES, "QAOA hamiltonian": QAOA_HAM_LANDSCAPES}
results_dirs = {"QAOA": QAOA_RESULTS, "QNN not transformed": QNN_not_trans_RESULTS, "QNN transformed": QNN_trans_RESULTS, "QAOA hamiltonian": QAOA_HAM_RESULTS}
titles = {"QAOA": "QAOA", "QNN not transformed": "QNN_not_transformed", "QNN transformed": "QNN_transformed_50_100", "QAOA hamiltonian": "QAOA_hamiltonian", "BP/NG": "QAOA_BP_NG"}

gamma_grid_size = 65
beta_grid_size = 25
min_gamma_bp_ng = -0.15
min_beta_bp_ng = np.pi/4 + np.pi/16

backend = QulacsSimulator()
# min_gamma=-np.pi
max_gamma=np.pi
# min_beta=-np.pi/2
max_beta=np.pi/2
min_gamma= - np.pi #-3.05
min_beta= - np.pi/2 #-1.5
qaoa_grid_size = [125,12,5] #roughly 15625 points
#qaoa_grid_size = [1000, 33, 10] # grid sizes for QAOA, keys are p values, all grids have roughly 1M grid points
#qaoa_grid_size = [707, 26, 9] # lower grid sized for QAPA, all grids have roughly 500K grid points
cpu_count = os.cpu_count() 

def get_qaoa_ids(p=None, num_qubits = None):
    '''
    Get correct qaoa ids (removes duplicates due to different sample sets)
    '''
    if p is None and num_qubits is None:
        id_list = [(p-1)*(5*6*5)+i for p in [1,2,3] for i in range(30)]
    else:
        id_list = [(p-1)*(5*6*5)+(num_qubits//3-1)*5+ i for i in range(5)]
    return id_list

def get_qnn_ids(s_rank=None):
    '''
    Get correct qnn ids (removes duplacted due to different samples sets)
    '''
    assert s_rank in [1,2,3,4,None]
    if s_rank==None:
        id_list = [5*i for i in range(100)]
    else:
        id_list = [(s_rank-1)*125+ 5*i for i in range(25)]
    return id_list

######## QNN ###########
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

def compute_roughness_metrics_for_transformed_qnn():
    roughness_dict = {"info": "Roughness metric values (Total Variation, Scalar Curvature, Fourier Density) for qnn instances. Computed on grid landscapes, which are definied by entry in fields with keys landscape... "}
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[START] {now}")
    timestamp = generate_timestamp_str()
    METRICS_DIR = QNN_trans_RESULTS / "roughness_metrics_results"
    METRICS_DIR.mkdir(exist_ok=True)
    id_list = get_qnn_ids()
    print(id_list)
    # for file in landscape_directory.iterdir():
    #     compute_roughness_metrics_per_landscape(file, vqa_type)
    #compute same-dimensional landscapes concurrently
    cpu_count = os.cpu_count() 
    print(cpu_count)
    assert cpu_count is not None
    with ProcessPoolExecutor(max_workers=cpu_count-3) as exe:
        futures = [exe.submit(compute_roughness_metrics_for_QNN_file,id) for id in id_list]            
        # await results & save them:
        for future in as_completed(futures):
            id, metrics = future.result()
            roughness_dict[id] = metrics
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[DONE] {now}: {id}")
    
    metrics_file = METRICS_DIR / f"QNN_roughness_metrics_{timestamp}.json"
    metrics_file.write_text(json.dumps(roughness_dict, indent=4))
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[DONE] {now}: transformed QNN")

def compute_roughness_metrics_for_QNN_file(id):
    file = QNN_trans_LANDSCAPES / f"persistence_qnn_{id}_transformed_50_100_H1.json"
    qnn_dict = json.load(open(file))
    sample_set = qnn_dict["sample_set"]
    assert sample_set == 0
    num_qubits = 2
    unitary = np.array(ast.literal_eval(qnn_dict["unitary"]))
    x = np.array(ast.literal_eval(qnn_dict["databatch"]))
    loss_func = CostFunction(num_qubits=num_qubits, unitary=unitary, inputs=x)
    lowerleft = np.zeros(6)
    #upperright = np.concatenate((np.ones(p)*max_gamma, np.ones(p)*max_beta))
    step = 2*np.pi/6 # since loss func is periodic, divide landscape limit by 6 instead of 5 to remove last sample points
    stepsize = np.ones(6)*step
    grid_size = np.ones(6)*5
    coordinates, grid_landscape = get_grid_landscapes_from_stepsize(lowerleft,grid_size, loss_func, stepsize)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[INFO] {now}: landscape of {id} done")
    asc, scalar_curvature, total_variation, fourier_density = calc_TV_FD_ASC_for_grid_landscape(grid_landscape)
    metrics_dict = {"qnn id": id,
                        "unitary": qnn_dict["unitary"],
                        "data points": qnn_dict["databatch"],
                        "landscape min": lowerleft.tolist(),
                        "landscape grid_size": grid_size.tolist(),
                        "landscape stepsize": stepsize.tolist(),
                        "landscape coordinates": [coords.tolist() for coords in coordinates],
                        "loss landscape": grid_landscape.tolist(),
                        "average absolute scalar curvature": asc, 
                        "scalar curvature values": scalar_curvature.tolist(),
                        "total variation": total_variation, 
                        "fourier density": fourier_density
                    }
    # for k in metrics_dict.keys():
    #     print(k, type(metrics_dict[k]))
    return id, metrics_dict
  

######### QAOA ###########
def compute_roughness_metrics_for_qaoa():
    roughness_dict = {"info": "Roughness metric values (Total Variation, Scalar Curvature, Fourier Density) for qaoa instances. Computed on grid landscapes, which are definied by entry in fields with keys landscape... "}
    METRICS_DIR = QAOA_RESULTS / "roughness_metrics_results"
    METRICS_DIR.mkdir(exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[START] {now}")
    timestamp = generate_timestamp_str()
    #id_list = get_qaoa_ids()
    id_list = range(30)
    i = 0
    #print(id_list)
    # for file in landscape_directory.iterdir():
    #     compute_roughness_metrics_per_landscape(file, vqa_type)
    #compute same-dimensional landscapes concurrently
    with ProcessPoolExecutor(max_workers=cpu_count-1) as exe:
        futures = [exe.submit(compute_roughness_metrics_for_QAOA_file,id) for id in id_list]            
        # await results & save them:
        for future in as_completed(futures):
            id, metrics = future.result()
            roughness_dict[id] = metrics
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            i += 1
            print(f"[DONE] {now}: {i} / {len(id_list)}")
    
    
    metrics_file = METRICS_DIR / f"QAOA_roughness_metrics_{timestamp}.json"
    metrics_file.write_text(json.dumps(roughness_dict, indent=4))
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[DONE] {now}: QAOA")

def compare_roughness_metrics_for_qaoa_per_p(roughness_metric):
    assert roughness_metric in ["average absolute scalar curvature", "total variation", "fourier density"]
    # load file
    file = BASE_DIR / "experiment_results" / "QAOA" / "roughness_metrics_results" / "QAOA_roughness_metrics_2026_04_28_06_31_25.json"
    metric_dict = json.load(open(file))
    # average/median/std per number of qubits and p
    result_dict = {"roughness metric": roughness_metric, "info": "entries in each list correspond to numbers of qubits [3,6,9,12,15,18]", "source file": "QAOA_roughness_metrics_2026_04_28_06_31_25.json"}
    for p in [1,2,3]:
        median = []
        mean = []
        std = []
        i = 0
        for num_qubits in [3,6,9,12,15,18]:
            id_list = get_qaoa_ids(p=p, num_qubits=num_qubits)
            values = []
            for id in id_list:
                values.append(metric_dict[str(id)][roughness_metric])
            median.append(np.median(values))
            mean.append(np.mean(values))
            std.append(np.std(values))
            i += 1
        result_dict[p] = {"p": p, "median": median, "mean": mean,"std": std,}
    save_file = file = BASE_DIR / "experiment_results" / "QAOA" / "roughness_metrics_results" / f"QAOA_{roughness_metric}_statistics.json"
    save_file.write_text(json.dumps(result_dict, indent=4))

def compute_roughness_metrics_for_QAOA_Hamiltonian_exp():
    ids = [1,5,10,15,20,25]
    roughness_dict = {"info": "Roughness metric values (Total Variation, Scalar Curvature, Fourier Density) for qaoa instances. Computed on grid landscapes, which are definied by entry in fields with keys landscape... ", "source path": str(QAOA_HAM_LANDSCAPES_scaled_50)}
    timestamp = generate_timestamp_str()
    i=0
    with ProcessPoolExecutor(max_workers=cpu_count-1) as exe:
        futures = [exe.submit(compute_roughness_metrics_for_QAOA_file_path,QAOA_HAM_LANDSCAPES_scaled_50,f"persistence_qaoa_{id}_flipped_True_scaled50_True_not_transformed_H1.json") for id in ids]            
        # await results & save them:
        for future in as_completed(futures):
            id, metrics = future.result()
            roughness_dict[id] = metrics
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            i += 1
            print(f"[DONE] {now}: {i} / {len(ids)}")
    
    
    RESULTS_DIR = BASE_DIR / "experiment_results" / "hamiltonian_experiment" / "scaled50" / "roughness_metrics"
    RESULTS_DIR.mkdir(exist_ok=True)
    metrics_file = RESULTS_DIR / f"QAOA_hamiltonian_scaled50_roughness_metrics.json"
    metrics_file.write_text(json.dumps(roughness_dict, indent=4))
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[DONE] {now}: QAOA")

def compute_roughness_metrics_for_QAOA_file_path(path, filename):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[START] {now}: {filename}")
    file = path / filename
    qaoa_dict = json.load(open(file))
    sample_set = qaoa_dict["sample_set"]
    assert sample_set == 0
    p = np.asarray(qaoa_dict["p"])
    #print(id, p)
    grid_size = qaoa_grid_size[p-1] # grid size depends on dimensionality, i.e. p value
    id = qaoa_dict["config id"]
    ham = convert_dict_to_op(qaoa_dict.get("hamiltonian", qaoa_dict.get("new hamiltonian")))
    loss_func = prepare_cost_function(ham, backend)
    lowerleft = np.concatenate((np.ones(p)*min_gamma, np.ones(p)*min_beta))
    #upperright = np.concatenate((np.ones(p)*max_gamma, np.ones(p)*max_beta))
    gamma_step = 2*np.pi/grid_size 
    beta_step = np.pi/grid_size 
    stepsize = np.concatenate((np.ones(p)*gamma_step, np.ones(p)*beta_step))
    grid_size = np.ones(2*p)*grid_size
    coordinates, grid_landscape = get_grid_landscapes_from_stepsize(lowerleft,grid_size, loss_func, stepsize)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[START] {now}: landscape done")
    asc, scalar_curvature, total_variation, fourier_density = calc_TV_FD_ASC_for_grid_landscape(grid_landscape)
    metrics_dict = {"qaoa id": id,
                    "filename": str(filename),
                    "hamiltonian": qaoa_dict.get("hamiltonian", qaoa_dict.get("new hamiltonian")),
                    "landscape min": lowerleft.tolist(),
                    "landscape grid_size": grid_size.tolist(),
                    "landscape stepsize": stepsize.tolist(),
                    #"landscape coordinates": [coords.tolist() for coords in coordinates],
                    #"loss landscape": grid_landscape.tolist(), # too large
                    "average absolute scalar curvature": asc, 
                    #"scalar curvature values": scalar_curvature.tolist(),
                    "total variation": total_variation, 
                    "fourier density": fourier_density
                    }
    return id, metrics_dict

def compute_roughness_metrics_for_QAOA_file(id):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[START] {now}: QAOA ID {id}")
    file = QAOA_LANDSCAPES / f"persistence_qaoa_{id}_not_transformed_H1.json"
    qaoa_dict = json.load(open(file))
    sample_set = qaoa_dict["sample_set"]
    assert sample_set == 0
    p = np.asarray(qaoa_dict["p"])
    #print(id, p)
    grid_size = qaoa_grid_size[p-1] # grid size depends on dimensionality, i.e. p value
    id = qaoa_dict["config id"]
    ham = convert_dict_to_op(qaoa_dict["hamiltonian"])
    loss_func = prepare_cost_function(ham, backend)
    lowerleft = np.concatenate((np.ones(p)*min_gamma, np.ones(p)*min_beta))
    #upperright = np.concatenate((np.ones(p)*max_gamma, np.ones(p)*max_beta))
    gamma_step = 2*np.pi/grid_size 
    beta_step = np.pi/grid_size 
    stepsize = np.concatenate((np.ones(p)*gamma_step, np.ones(p)*beta_step))
    grid_size = np.ones(2*p)*grid_size
    coordinates, grid_landscape = get_grid_landscapes_from_stepsize(lowerleft,grid_size, loss_func, stepsize)
    asc, scalar_curvature, total_variation, fourier_density = calc_TV_FD_ASC_for_grid_landscape(grid_landscape)
    metrics_dict = {"qaoa id": id,
                        "hamiltonian": qaoa_dict["hamiltonian"],
                        "landscape min": lowerleft.tolist(),
                        "landscape grid_size": grid_size.tolist(),
                        "landscape stepsize": stepsize.tolist(),
                        "landscape coordinates": [coords.tolist() for coords in coordinates],
                        #"loss landscape": grid_landscape.tolist(), # too large
                        "average absolute scalar curvature": asc, 
                        "scalar curvature values": scalar_curvature.tolist(),
                        "total variation": total_variation, 
                        "fourier density": fourier_density
                    }
    return id, metrics_dict

def post_process_roughness_metrics_file(directory, filename):
    """
    Removes "landscape coordinates" and "scalar curvature values" from file to reduce file size
    """
    # load file
    file = directory / f"{filename}.json"
    data = json.load(open(file))

    # Remove keys
    for key, value in data.items():
        if isinstance(value, dict):
            value.pop("loss landscape", None)
            value.pop("landscape coordinates", None)
            value.pop("scalar curvature values", None)

    # Save cleaned JSON back to file
    save_file = file = directory / f"{filename}_cleaned.json"
    save_file.write_text(json.dumps(data, indent=4))
    

def main_roughness_metrics_experiment(vqa_type):
    timestamp = generate_timestamp_str()
    assert vqa_type in max_number_of_ids.keys()
    max_ids = max_number_of_ids[vqa_type]
    landscape_directory = landscape_dirs[vqa_type]
    results_directory = results_dirs[vqa_type]
    title = titles[vqa_type]
    landscape_directory.iterdir()

    config_dict = {"info": "Roughness metrics for loss landscapes different QAOA instances (keys correspond to QAOA IDs)", "source files": "resources\QAOA\landscapes"}
    cpu_count = os.cpu_count() 
    #print(cpu_count)
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
    metrics_file = METRICS_DIR / f"{title}_roughness_metrics_{timestamp}.json"
    metrics_file.write_text(config_dict)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[DONE] {now}: {vqa_type}")

def compute_roughness_metrics_for_QAOA_BP_NG():
    RESULTS_BASE_DIR = BASE_DIR / "experiment_results" / "BP_NG" / "small_excerpt" / "LHS_samples"
    filename = "qaoa_id_20_landscape_BP_NG_LHS.json"

    roughness_dict = {"info": "Roughness metric values (Total Variation, Scalar Curvature, Fourier Density) for qaoa bp/ng experiment instances. Computed on grid landscapes, which are definied by entry in fields with keys landscape... ", "source path": str(RESULTS_BASE_DIR / filename )}
    timestamp = generate_timestamp_str()

    file = RESULTS_BASE_DIR / filename
    qaoa_dict = json.load(open(file))
    ham = convert_dict_to_op(qaoa_dict.get("hamiltonian", qaoa_dict.get("new hamiltonian")))
    id = qaoa_dict["qaoa id"] 
    p = qaoa_dict["p"]
    loss_func = prepare_cost_function(ham, backend)
    lowerleft = np.concatenate((np.ones(p)*min_gamma_bp_ng, np.ones(p)*min_beta_bp_ng))
    grid_res = 40 # due to the roughness metric computations its not possible to have a rectangular grid, where one dimension has a lower number of grid points than the other
    for k in range(5):
        gamma_upper = 2/(2**k)
        gamma_step = 2/((2**k)*grid_res)
        beta_step = np.pi/(8*grid_res)
        step_size = np.concatenate((np.ones(p)*gamma_step, np.ones(p)*beta_step))
        grid_size = np.ones(2*p)*grid_res
        print(grid_size)
        print(step_size)
        coordinates, grid_landscape = get_grid_landscapes_from_stepsize(lowerleft,grid_size, loss_func, step_size)

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[START] {now}: landscape done")
        asc, scalar_curvature, total_variation, fourier_density = calc_TV_FD_ASC_for_grid_landscape(grid_landscape)
        metrics_dict = {"qaoa id": id,
                        "p": p,
                        "filename": str(filename),
                        "hamiltonian": qaoa_dict.get("hamiltonian", qaoa_dict.get("new hamiltonian")),
                        "k": k,
                        "landscape min": lowerleft.tolist(),
                        "landscape grid_size": grid_size.tolist(),
                        "landscape stepsize": step_size.tolist(),
                        #"landscape coordinates": [coords.tolist() for coords in coordinates],
                        #"loss landscape": grid_landscape.tolist(), # too large
                        "average absolute scalar curvature": asc, 
                        #"scalar curvature values": scalar_curvature.tolist(),
                        "total variation": total_variation, 
                        "fourier density": fourier_density
                        }
        roughness_dict[k] = metrics_dict
    
    RESULTS_DIR = RESULTS_BASE_DIR / "roughness_metrics"
    RESULTS_DIR.mkdir(exist_ok=True)
    metrics_file = RESULTS_DIR / f"qaoa_id_20_roughness_metric_BP_NG_grid.json"
    metrics_file.write_text(json.dumps(roughness_dict, indent=4))
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[DONE] {now}: QAOA BP/NG")
        

def test():
    RESULTS_BASE_DIR = BASE_DIR / "experiment_results" / "BP_NG" / "small_excerpt" /"grid_samples"
    filename = "qaoa_id_20_landscape_BP_NG_grid.json"

    file = RESULTS_BASE_DIR / filename
    qaoa_dict = json.load(open(file))
    ham = convert_dict_to_op(qaoa_dict.get("hamiltonian", qaoa_dict.get("new hamiltonian")))
    id = qaoa_dict["qaoa id"] 
    p = qaoa_dict["p"]
    loss_func = prepare_cost_function(ham, backend)
    lowerleft = np.concatenate((np.ones(p)*min_gamma_bp_ng, np.ones(p)*min_beta_bp_ng))

    #upperright = np.concatenate((np.ones(p)*max_gamma, np.ones(p)*max_beta))
    gamma_grid_size_k = 3
    beta_grid_size_k = 5
    gamma_step = 2/gamma_grid_size_k 
    beta_step = np.pi/(8*beta_grid_size_k)
    stepsize = np.concatenate((np.ones(p)*gamma_step, np.ones(p)*beta_step))
    grid_size = np.concatenate((np.ones(p)*gamma_grid_size_k, np.ones(p)*beta_grid_size))
    print(grid_size)
    print(stepsize)
    coordinates, grid_landscape = get_grid_landscapes_from_stepsize(lowerleft,grid_size, loss_func, stepsize)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[START] {now}: landscape done")
    asc, scalar_curvature, total_variation, fourier_density = calc_TV_FD_ASC_for_grid_landscape(grid_landscape)


if __name__ == "__main__":
    #main_roughness_metrics_experiment("QAOA")
    #test_gradient()
    #main_roughness_metrics_experiment(vqa_type="QAOA")
    #compute_roughness_metrics_for_transformed_qnn()
    #compute_roughness_metrics_for_qaoa()
    directory = BASE_DIR / "experiment_results" / "QNN" / "transformed_50_100" / "roughness_metrics_results"
    filename = "QNN_roughness_metrics_2026_03_21_20_23_46"
    #post_process_roughness_metrics_file(directory=directory, filename=filename)
    #compute_roughness_metrics_for_QAOA_Hamiltonian_exp()
    # for metric in ["average absolute scalar curvature", "total variation", "fourier density"]:
    #     compare_roughness_metrics_for_qaoa_per_p(metric)
    compute_roughness_metrics_for_QAOA_BP_NG()
    #test()