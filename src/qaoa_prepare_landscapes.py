from concurrent.futures import ProcessPoolExecutor, as_completed
from time import sleep
import json
import math
import os
import sys
from typing import Dict
from datetime import datetime

SCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
sys.path.append(os.path.dirname(SCRIPT_DIR))

import networkx as nx
import numpy as np
from orquestra.integrations.qulacs.simulator import QulacsSimulator
from orquestra.opt.problems.maxcut import MaxCut
from orquestra.quantum.operators import PauliSum, PauliTerm

from qaoa.data_generation import generate_data, prepare_cost_function, save_hamiltonians, perform_2D_scan, generate_landscape
from qaoa.hamiltonian_generation import assign_random_weights, assign_weight_for_term
from qaoa.plots import create_plot, create_tda_plot
from qaoa.utils import generate_timestamp_str
from utils.sampling_utils import get_latin_hypercube_samples, get_grid_samples
from utils.file_utils import save_landscape, save_persistence_diagrams

import orqviz
from orqviz.scans import Scan2DResult
from typing import Any, Dict, List, Optional, Tuple
from ripser import ripser
from persim import plot_diagrams
from matplotlib import pyplot as plt
from scipy.stats import qmc


backend = QulacsSimulator()
scan_resolution = 101 # 31^2 approx 1000, was 101, but too much for persistene
num_runs=5
N = 10000


def generate_LHS_sample_points_for_qaoa(number_of_samples=10000, min_gamma=-np.pi, max_gamma=np.pi, min_beta=-np.pi/2, max_beta=np.pi/2):
    """
    Generates 5 sets of LHS sample points for QAOA for each p=1,2,3 (i.e. 2D, 4D and 6D sample points)

    :param number_of_samples: number of sample points, default: 10 000
    :param min_gamma: minimum value for gamma parameter, default: -np.pi
    :param max_gamma: maximum value for gamma parameter, default: np.pi
    :param min_beta: minimum value for beta parameter, default: -np.pi/2
    :param max_beta: maximum value for beta parameter, default: np.pi/2
    """
    # save path
    directory = os.path.join(os.path.join(os.path.join(os.getcwd(), "resources"), "sample_points"), "QAOA")
    # strings for file name to save sample points
    gamma_value = str(np.round(max_gamma, 2))
    beta_value = str(np.round(max_beta, 2))
    if max_gamma==np.pi:
        gamma_value = "pi"
    if max_beta==np.pi/2:
        beta_value = "0.5pi"

    for p in range(1,4):
        n = 2*p
        save_dir = os.path.join(directory, f"{n}D")
        #determine correct lower left and upper right corner of parameter space
        lowerleft = np.concatenate((np.ones(p)*min_gamma, np.ones(p)*min_beta))
        upperright = np.concatenate((np.ones(p)*max_gamma, np.ones(p)*max_beta))

        # generate 20 different sets of sample points per dimension (i.e. per p)
        samples = []
        discrepancies = []
        for run in range(20):
            # generate LHS samples
            sample_points = get_latin_hypercube_samples(lowerleft=lowerleft, upperright=upperright, dim=n, number_of_samples=number_of_samples)
            sampler = qmc.LatinHypercube(d=n)
            sample_points = sampler.random(n=number_of_samples)
            discrepancy = qmc.discrepancy(sample_points)
            sample_points = qmc.scale(sample_points,l_bounds=lowerleft, u_bounds=upperright)
            
            samples.append(sample_points)
            discrepancies.append(discrepancy)

        # determine sample points with lowest discrepancy and save them
        # sort sample point sets according to discrepancy
        sorted_samples = [s for _, s in sorted(zip(discrepancies, samples))]
        sorted_discrepancies = [d for d, _ in sorted(zip(discrepancies, samples))]
        # save best 5 sample point sets as json files (best according to discrepancy, lower discrepancy = better)
        for i in range(5):
            s = sorted_samples[i]
            d = sorted_discrepancies[i]
            file_name = f"samples_LHS_{number_of_samples}_gamma_{gamma_value}_beta_{beta_value}_set_{i}_discrepancy_{d}.json" 
            with open(os.path.join(save_dir, file_name), "w") as f:
                json.dump(s.tolist(), f)


def main():
    timestamp = generate_timestamp_str()

    # generate and save hamiltonians
    hamiltonians_dict: Dict[str, PauliSum] = {}
    ham_config_dict = {}
    id = 0
    start = datetime.now()
    for num_qubits in [3, 6, 9, 12, 15, 18]: #war: [], aber alle schon erledigt
        for run in range(num_runs):
            G_complete_graph = nx.complete_graph(num_qubits)
            weighted_MaxCut_hamil = MaxCut().get_hamiltonian(G_complete_graph)
            weighted_MaxCut_hamil = assign_weight_for_term(
                weighted_MaxCut_hamil, PauliTerm("I0"), 0
            )
            ham_file_label = f"ham_MAXCUT_weighted_-10_to_10_qubits_{num_qubits}_run_{run}"
            hamiltonian = assign_random_weights(weighted_MaxCut_hamil)
            ham_config_dict[ham_file_label] = {
                "num_qubits": num_qubits,
                "run": run,
                "ham_file_label": ham_file_label,
                "weight_limits": [-10, 10],
                "hamiltonian": hamiltonian
            }
    #save_hamiltonians(hamiltonians_dict, timestamp_str=timestamp)

    config_dict = {}
    cpu_count = os.cpu_count() 
    print(cpu_count)
    assert cpu_count is not None
    for p in range(1,4):
        for set in range(5): # each dimension has 5 different sets of sample points
            for num_qubits in [3,6,9,12,15,18]:
                #compute same-dimensional landscapes concurrently
                with ProcessPoolExecutor(max_workers=5) as exe:
                    futures = [exe.submit(generate_qaoa_landscape,ham_config_dict, f"ham_MAXCUT_weighted_-10_to_10_qubits_{num_qubits}_run_{run}", p, set) for run in range(num_runs)]            
                    # await results & save them:
                    for future in as_completed(futures):
                        id, run, samples_file_name, elapsed_time, landscape, ham_file_label = future.result()
                        config_dict[f"qaoa_{id}"] = ham_config_dict[ham_file_label]
                        config_dict[f"qaoa_{id}"]["p"] = p
                        config_dict[f"qaoa_{id}"]["sample_set"] = set
                        config_dict[f"qaoa_{id}"]["sample_file_label"] = samples_file_name
                        config_dict[f"qaoa_{id}"]["runtime"] = str(elapsed_time).split(".",1)[0]

                        # save landscape
                        landscape_directory = os.path.join(os.path.join(os.path.join(os.getcwd(), "resources"), "QAOA"), "landscapes")
                        landscape_file_name = f"landscape_qaoa_{id}_qubits_{num_qubits}_run_{run}_p_{p}_set_{set}_{timestamp}"
                        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        print(f"[DONE] {now}: {landscape_file_name}")
                        save_landscape(landscape, meta_data = config_dict[f"qaoa_{id}"], directory=landscape_directory, file_name=landscape_file_name, id=f"qaoa_{id}")

def generate_qaoa_landscape(ham_config_dict, ham_file_label, p, set):
    #print("[INFO] started", ham_file_label)
    n=2*p
    num_qubits = ham_config_dict[ham_file_label]["num_qubits"]
    run = ham_config_dict[ham_file_label]["run"]
    id = (p-1)*150+set*25+(num_qubits//3-1)*5+run # compute config id for later
    print(p, num_qubits, run, id)
    # get hamiltonian
    hamiltonian = ham_config_dict[ham_file_label]["hamiltonian"]

    # prepare cost function
    start = datetime.now()
    cost_function = prepare_cost_function(hamiltonian, backend)

    # load correct sample points
    s = f"samples_LHS_10000_gamma_pi_beta_0.5pi_set_{set}" 
    s_dir = os.path.join(os.getcwd(), f"resources/sample_points/QAOA/{n}D")
    all_sample_sets = os.listdir(s_dir)
    for file in all_sample_sets:
        if file.startswith(s):
            file_path = os.path.join(s_dir, file)
            with open(file_path) as f:
                sample_points =  np.asarray(json.load(f))
    
    # generate loss landscape
    landscape = generate_landscape(cost_function, sample_points)
    #print(len(landscape))
    elapsed_time = datetime.now()-start
    #print(f"Loss Landscape:", elapsed_time.strftime("%H:%M:%S"))
    return id, run, s, elapsed_time, landscape, ham_file_label


if __name__ == "__main__":
    #generate_LHS_sample_points_for_qaoa() #done
    main()
