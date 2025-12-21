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

from qaoa.data_generation import generate_data, prepare_cost_function, save_hamiltonians, perform_2D_scan
from qaoa.hamiltonian_generation import assign_random_weights, assign_weight_for_term
from qaoa.utils import calculate_plot_extents, generate_timestamp_str


from ripser import ripser
from persim import plot_diagrams
from matplotlib import pyplot as plt
from scipy.stats import qmc

from utils.file_utils import save_landscape, save_persistence_diagrams
from qaoa.data_generation import generate_landscape


backend = QulacsSimulator()
scan_resolution = 101 # 31^2 approx 1000, was 101, but too much for persistene
num_runs=1

def generate_loss_landscapes_for_test():
    """
    Generates 9 QAOA loss landscapes. Number of qubits is always 3. 
    p = 1,2,3 (number of repititions of cost and mixer operator in QAOA circuit)
    Samples points are 5k, 10k, 15k generated using Latin Hypercube sampling in 
    the hypercube [-0.5pi, 0.5pi]^2p (i.e. Dimension is 2, 4, 6).
    """
    #generate and save hamiltonian
    num_qubits = 3
    hamiltonians_dict = {}
    G_complete_graph = nx.complete_graph(num_qubits)
    weighted_MaxCut_hamil = MaxCut().get_hamiltonian(G_complete_graph)
    weighted_MaxCut_hamil = assign_weight_for_term(
        weighted_MaxCut_hamil, PauliTerm("I0"), 0
    )
    file_label = f"ham_MAXCUT_weighted_-10_to_10_qubits_{num_qubits}_run_0"
    hamiltonian = assign_random_weights(weighted_MaxCut_hamil)
    hamiltonians_dict[file_label] = hamiltonian
    timestamp = generate_timestamp_str()
    save_hamiltonians(hamiltonians_dict, timestamp_str=timestamp)

    for N in [5000, 10000, 15000]:    
        for p in range(1,4):  
            n=2*p

            # load correct sample points
            s = f"samples_LHS_{N}_-0.5pi_0.5pi.json"
            samples_file_name = f"resources\sample_points_{n}D\{s}" 
            directory_name = os.path.join(
                os.path.join(os.getcwd(), samples_file_name)
            )
            with open(directory_name) as f:
                sample_points =  np.asarray(json.load(f))

            # compute and save loss landscape
            cost_function = prepare_cost_function(hamiltonian, backend)
            landscape = generate_landscape(cost_function, sample_points)
            save_landscape(landscape, ham_file_label=file_label, sample_points_file_label=s, timestamp=timestamp, p=p)



def main():
    """
    Perform TDA on QAOA loss landscape for N=5k, 10k, 15k (number of sample points) and
    dim = 2, 4, 6 (dimension of sample points), i.e. 3D, 5D and 7D data points that are analyzed.
    Computes homologies up to dimension 2 (H_0, H_1, H_2).
    """
    # load loss landscape
    output = ""
    for N in [5000, 10000, 15000]:    
        for p in range(1,4): 
            start = datetime.now()
            # load landscape
            ham_file_label = "ham_MAXCUT_weighted_-10_to_10_qubits_3"
            sample_file_label = f"LHS_{N}"
            landscape_file_name = f"landscape_{ham_file_label}_{sample_file_label}_p_{p}.json"
            landscape_dir = os.path.join(
                os.path.join(os.getcwd(), os.path.join("resources\QAOA_H2_test\landscapes", landscape_file_name))
            )
            with open(landscape_dir) as f:
                landscape_dict =  json.load(f)
            landscape = np.asarray(landscape_dict["landscape"])

            # analyze landscape (TDA)
            ripser_result = ripser(landscape, maxdim=2) # todo: change maxdim to 2
            # save tda result
            save_persistence_diagrams(ripser_result, ham_file_label, sample_file_label, timestamp=generate_timestamp_str(), p=p, id="")

            elapse_time = datetime.now() - start
            output_line = f"N={N}, p={p}, time(H2): {elapse_time}"
            print(output_line)
            output = output + "\n" + output_line





if __name__ == "__main__":
    main()

    #generate_loss_landscapes_for_test()
    

