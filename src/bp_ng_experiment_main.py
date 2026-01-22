"""
Experiment of 2D QAOA landscape containing both barren plateau (BP) and narrow gorge (NG)
"""

from datetime import datetime
import json
from pathlib import Path
from typing import Dict

import networkx as nx
import numpy as np
from orquestra.integrations.qulacs.simulator import QulacsSimulator
from orquestra.opt.problems.maxcut import MaxCut
from orquestra.quantum.operators import PauliSum, PauliTerm
from orquestra.integrations.qulacs.simulator import QulacsSimulator
from orquestra.quantum.operators import convert_dict_to_op
from ripser import ripser
from persim import plot_diagrams, bottleneck, bottleneck_matching
from matplotlib import pyplot as plt

from qaoa.hamiltonian_generation import assign_random_weights, assign_weight_for_term
from qaoa.utils import generate_timestamp_str
from qaoa.data_generation import prepare_cost_function, generate_landscape
from utils.sampling_utils import get_2D_grid_samples, get_latin_hypercube_samples
from scipy.interpolate import griddata

# delta to determine epsilon for gamma value
delta = 0.01
threshold = 1e-2
epsilon = 0.15 # determined epsilon rounded to two decimal places
upper_limit_gamma = 2
dim = 2 # max homology dimension

# backend simulator
backend = QulacsSimulator()
# Directories
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_BASE_DIR = BASE_DIR / "experiment_results" / "BP_NG"
HAMILTONIAN_DIR = BASE_DIR / "resources" / "QAOA" / "landscapes"
LANDSCAPE_DIR = BASE_DIR / "resources" / "BP_NG" 
SAMPLE_POINT_DIR = BASE_DIR / "resources" / "sample_points" / "BP_NG"

def perform_BP_NG_experiment(grid=True):
    timestamp = generate_timestamp_str()
    LANDSCAPE_DIR.mkdir(exist_ok=True)
    RESULTS_BASE_DIR.mkdir(exist_ok=True)
    CURR_RESULTS_DIR = RESULTS_BASE_DIR / timestamp
    CURR_RESULTS_DIR.mkdir(exist_ok=True)
    num_qubits = 15
    run = 0
    all_hamiltonians = HAMILTONIAN_DIR.iterdir()
    for file in all_hamiltonians:
        for num_qubits in [15]:
            for run in [0]:
                filename_ending = f"qubits_{num_qubits}_run_{run}_p_1_set_0_2026_01_03_17_02_45"
                if file.name.endswith(filename_ending):
                    with open(file) as f:
                        qaoa_dict =  json.load(f)
                        id = qaoa_dict["config id"]
                        hamiltonian_dict = qaoa_dict["hamiltonian"]
                        hamiltonian = convert_dict_to_op(hamiltonian_dict)
                        loss_function = prepare_cost_function(hamiltonian, backend)
                        
                        # load samples points
                        file_name = f"samples_gridsize1_65_gridsize225_gamma_-0.15-1.85_beta_.25pi-.5pi.json" 
                        sample_type_string = "grid"
                        if(not grid):
                            file_name = "samples_LHS_1600points_gamma_-0.15-1.85_beta_.25pi-.5pi.json"
                            sample_type_string = "LHS"
                        samples_file_dir= SAMPLE_POINT_DIR / file_name 
                        with open(samples_file_dir) as f:
                            sample_points = np.asarray(json.load(f))
                        # generate loss landscape
                        landscape = generate_landscape(loss_function, sample_points)
                        min = np.min(landscape[:, -1])
                        max = np.max(landscape[:, -1])
                        # (plot and) save loss landscape
                        landscape_dict = {}
                        landscape_dict["qaoa id"] = id
                        landscape_dict["num_qubits"] = 15
                        landscape_dict["p"] = 1
                        landscape_dict["sample_file_label"] = file_name
                        landscape_dict["hamiltonian"] = hamiltonian_dict
                        landscape_dict["min cost"] = min
                        landscape_dict["max cost"] = max
                        landscape_dict["landscape"] = landscape.tolist()

                        landscape_file_dir = CURR_RESULTS_DIR / f"qaoa_id_{id}_landscape_BP_NG_{sample_type_string}.json"
                        landscape_file_dir.write_text(json.dumps(landscape_dict, indent=4))

                        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        print(f"[DONE] generating landscape: {now}")
                        # compute persistence diagrams 
                        for k in range(0,5):
                            gamma_limit = upper_limit_gamma / 2**k - epsilon
                            partial_landscape = np.asarray([x for x in landscape if x[0] <= gamma_limit])
                            print("shape", partial_landscape.shape)

                            persistence_dict = {}
                            persistence_dict["qaoa id"] = id
                            persistence_dict["gamma limit"] = gamma_limit
                            persistence_dict["k"] = k
                            persistence_dict["partial_landscape"] = partial_landscape.tolist()
                            persistence_dict["number_sample_points"] = partial_landscape.shape[0]

                            start = datetime.now()
                            ripser_result = ripser(partial_landscape, maxdim=dim)
                            elapsed_time = datetime.now()-start
                            persistence_dict["runtime"] = str(elapsed_time).split(".",1)[0]

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

                            persistence_dict["persistence diagram"] = ripser_dict

                            # save ripser result, including landscape, etc.
                            file_name = f"persistence_qaoa_{id}_BP_NG_k={k}_H{dim}_{sample_type_string}.json"
                            file_name_plot = f"persistence_diagram_qaoa_{id}_BP_NG_k={k}_H{dim}_{sample_type_string}.png"
                            ripser_path = CURR_RESULTS_DIR / "ripser_results" 
                            ripser_path.mkdir(exist_ok=True)
                            plot_path = CURR_RESULTS_DIR / "persistence_diagrams"
                            plot_path.mkdir(exist_ok=True)
                            ripser_file = ripser_path / file_name
                            ripser_file.write_text(json.dumps(persistence_dict, indent=4))

                            # generate and save persistence diagram
                            plot_diagrams(ripser_result["dgms"], show=False)
                            plt.savefig(plot_path / file_name_plot)
                            plt.close()
                            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            print(f"[DONE] {k}/4 at {now}: {file_name}")


def generate_grid_sample_points(): #DONE
    SAMPLE_POINT_DIR.mkdir(exist_ok=True)

    # generate grid samples
    min_gamma = -epsilon
    max_gamma = upper_limit_gamma-epsilon
    min_beta = np.pi/4
    max_beta = np.pi/2
    beta_limits = ".25pi-.5pi"
    sample_points = get_2D_grid_samples(min_gamma, min_beta, max_gamma, max_beta, grid_size1=65, grid_size2=25)

    # save samples
    file_name = f"samples_gridsize1_{65}_gridsize2{25}_gamma_{min_gamma}-{max_gamma}_beta_{beta_limits}.json" 
    file_dir = SAMPLE_POINT_DIR / file_name
    file_dir.write_text(json.dumps(sample_points.tolist(), indent=4))

def generate_LHS_sample_points():
    SAMPLE_POINT_DIR.mkdir(exist_ok=True)

    # generate grid samples
    min_gamma = -epsilon
    max_gamma = upper_limit_gamma-epsilon
    min_beta = np.pi/4
    max_beta = np.pi/2
    beta_limits = ".25pi-.5pi"
    n=1600
    lowerleft = np.asarray([min_gamma, min_beta])
    upperright = np.asarray([max_gamma, max_beta])
    sample_points = get_latin_hypercube_samples(lowerleft, upperright, dim=2, number_of_samples=n)

    # save samples
    file_name = f"samples_LHS_{n}points_gamma_{min_gamma}-{max_gamma}_beta_{beta_limits}.json" 
    file_dir = SAMPLE_POINT_DIR / file_name
    file_dir.write_text(json.dumps(sample_points.tolist(), indent=4))
    

def determine_epsilon_for_gamma():
    num_qubits = 15
    run = 0
    all_hamiltonians = HAMILTONIAN_DIR.iterdir()
    for file in all_hamiltonians:
        for num_qubits in [15]:
            for run in [0]:
                filename_ending = f"qubits_{num_qubits}_run_{run}_p_1_set_0_2026_01_03_17_02_45"
                if file.name.endswith(filename_ending):
                    with open(file) as f:
                        qaoa_dict =  json.load(f)
                        id = qaoa_dict["config id"]
                        hamiltonian_dict = qaoa_dict["hamiltonian"]
                        hamiltonian = convert_dict_to_op(hamiltonian_dict)
                        loss_function = prepare_cost_function(hamiltonian, backend)

                        beta = np.pi/4 + np.pi/8
                        old_loss = np.inf
                        for k in range(10,50):
                            gamma = 0 - k * delta
                            parameters = np.asarray([gamma, beta])
                            loss = loss_function(parameters)
                            print(f"k={k}, epsilon={k*delta}: loss = {loss}")
                            if(loss < 1 and np.abs(old_loss-loss) < threshold):
                                epsilon = k* delta
                                break
                            old_loss = loss
    print(np.round(epsilon, 2))
    return np.round(epsilon, 2)

def compute_bottleneck_distances(h_dim, grid=True):
    assert h_dim in [0,1,2]
    if grid:
        timestamp_str = "2026_01_20_09_42_16"
        file_ending = ""
    else:
        timestamp_str = "2026_01_21_16_02_21"
        file_ending = "_LHS"
        
    RIPSER_RESULTS_DIR = RESULTS_BASE_DIR / timestamp_str / "ripser_results"
    distance_matrix = np.zeros((5,5))
    #k1 and k2 determine part of loss landscape that was used to compute persistence diagram
    for k1 in range(5):
        file1 = RIPSER_RESULTS_DIR / f"persistence_qaoa_20_BP_NG_k={k1}_H2{file_ending}.json"
        results_dict1 = json.load(open(file1))
        dgm1 = np.asarray(results_dict1["persistence diagram"]["dgms"][h_dim])
        del results_dict1
        for k2 in range(k1+1, 5):
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[INFO] {now}: Start k1={k1}, k2={k2}")
            file2 = RIPSER_RESULTS_DIR / f"persistence_qaoa_20_BP_NG_k={k2}_H2{file_ending}.json"
            results_dict2 = json.load(open(file2))
            dgm2 = np.asarray(results_dict2["persistence diagram"]["dgms"][h_dim])
            del results_dict2
            distance, matching = bottleneck(dgm1, dgm2, matching=True)
            bottleneck_matching(dgm1, dgm2, matching=matching, labels=[f"1/{(2**k1)}", f"1/{(2**k2)}"])
            directory = RESULTS_BASE_DIR / timestamp_str / "bottleneck"
            directory.mkdir(exist_ok=True)
            fig_dir = RESULTS_BASE_DIR /  timestamp_str / "bottleneck" / f"bottleneck_matching_k1_{k1}_k2_{k2}_H{h_dim}.png"
            plt.savefig(fig_dir)
            plt.close()
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[INFO] k1={k1}, k2={k2}: bottleneck dist = {distance}")
            distance_matrix[k1,k2] = distance
    return distance_matrix

def plot_loss_landscape():
    file = RESULTS_BASE_DIR / "2026_01_20_09_42_16" / "qaoa_id_20_landscape_BP_NG_grid.json"
    results_dict = json.load(open(file))
    landscape = np.asarray(results_dict["landscape"])
    gamma = landscape[:,0]
    beta = landscape[:,1]
    loss = landscape[:,2]
    plt.scatter(gamma, beta, c=loss, cmap="viridis")
    plt.xlabel("gamma")
    plt.ylabel("beta")
    plt.xlim([0-epsilon, upper_limit_gamma-epsilon])
    plt.ylim([np.pi/4, np.pi/2])
    plt.show()

def plot_loss_landscape2():
    """
        random plot. just trying stuff out
    """
    file = RESULTS_BASE_DIR / "2026_01_20_09_42_16" / "qaoa_id_20_landscape_BP_NG_grid.json"
    results_dict = json.load(open(file))
    landscape = np.asarray(results_dict["landscape"])
    gamma = landscape[:,0]
    beta = landscape[:,1]
    loss = landscape[:,2]
    # target grid to interpolate to

    xi = np.arange(0-epsilon, 2-epsilon+0.01,0.01)
    yi = np.arange(np.pi/4, np.pi/2+0.01, 0.01)
    xi,yi = np.meshgrid(xi,yi)

    # set mask
    mask = (xi > 0.5) & (xi < 0.6) & (yi > 0.5) & (yi < 0.6)

    # interpolate
    zi = griddata((gamma,beta),loss,(xi,yi),method='linear')

    # mask out the field
    zi[mask] = np.nan

    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.contourf(xi,yi,zi,np.arange(0,1.01,0.01))
    plt.plot(gamma,beta,'k.')
    plt.xlabel('xi',fontsize=16)
    plt.ylabel('yi',fontsize=16)
    plt.savefig('interpolated.png',dpi=100)
    #plt.close(fig)

if __name__ == "__main__":
    #random()
    #determine_epsilon_for_gamma()
    #generate_grid_sample_points()
    
    #generate_LHS_sample_points()
    #perform_BP_NG_experiment(grid=False)
    #distance_matrix = compute_bottleneck_distances(h_dim=1)
    #print(distance_matrix)
    for h_dim in [2]:
        distance_matrix = compute_bottleneck_distances(h_dim=h_dim, grid=False)
        print(distance_matrix)
    