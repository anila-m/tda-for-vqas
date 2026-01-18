import json
import math
import os
import sys
from typing import Dict
from datetime import datetime

from utils.sampling_utils import get_latin_hypercube_samples

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
from utils.data_utils import get_interval_transformation
from qaoa.data_generation import generate_landscape

from qaoa_prepare_landscapes import generate_LHS_sample_points_for_qaoa


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

def qaoa_transformed_main():
    """
    Perform TDA on QAOA loss landscape
    number of sample points N= 10k (latin hypercube sampling)
    number of qubits: 15
    p = 1,2,3 (i.e. final dimension von points in point cloud: 3D, 5D, 7D)
    transformed cost values to [50,100] 
    """
    
    dir = os.path.join(
        os.path.join(
            os.path.join(
                os.path.join(
                    os.path.join(os.getcwd(),"experiment_results")
                    ,"Initial_QAOA_tests")
                ,"H1")
            ,"variable_p")
        ,"landscapes_2025_12_20_17_45_07")
    for p in range(1,4):
        # load landscapes
        file_name = f"landscape_ham_MAXCUT_weighted_-10_to_10_qubits_15_run_0_p_{p}.json"
        landscape_dir = os.path.join(dir, file_name)
        with open(landscape_dir) as f:
            landscape_dict =  json.load(f)
        landscape = np.asarray(landscape_dict["landscape"])
        min = landscape_dict["min"]
        max = landscape_dict["max"]
        sample_points = landscape[:,:-1]
        costs = landscape[:,-1]

        # transform loss values & put landscape back together
        f = get_interval_transformation(a=50, b=100, min=min, max=max)
        tranformed_costs = np.array([f(c) for c in costs])
        transformed_landscape = np.column_stack((sample_points, tranformed_costs))
        # perform TDA: persistence diagrams up tp H1
        start = datetime.now()
        ripser_result = ripser(transformed_landscape, maxdim=1) 

        # create persistence diagram (plot) and save it
        result_directory = os.path.join(os.path.join(os.path.join(os.path.join(os.getcwd(), "experiment_results"), "Initial_QAOA_tests"), "H1"), "transformed_cost")
        diagrams = ripser_result["dgms"]
        plot_diagrams(diagrams, show=False) # axis limits fit transformed loss values (to [50,100])
        if not os.path.exists(result_directory):
            os.mkdir(result_directory)
        file_name = f"persistence_transformed_landscape_50_100_15qubits_p_{p}.png"
        plt.savefig(os.path.join(result_directory, file_name))
        plt.close()
        elapse_time = datetime.now() - start
        output_line = f"p={p}, time(H1): {elapse_time}"
        print(output_line)
        # save persistence diagram (plot)
        

def qaoa_h2_test():
    p = 3
    min_gamma = -np.pi
    max_gamma = np.pi
    min_beta = -np.pi/2
    max_beta = np.pi/2
    N = 2000
    num_qubits = 3
    n = 2*p
    timestamp = generate_timestamp_str()


    # sample points
    lowerleft = np.concatenate((np.ones(p)*min_gamma, np.ones(p)*min_beta))
    upperright = np.concatenate((np.ones(p)*max_gamma, np.ones(p)*max_beta))
    sample_points = get_latin_hypercube_samples(lowerleft=lowerleft, upperright=upperright, dim=n, number_of_samples=N)

    # generate landscape
    G_complete_graph = nx.complete_graph(num_qubits)
    weighted_MaxCut_hamil = MaxCut().get_hamiltonian(G_complete_graph)
    weighted_MaxCut_hamil = assign_weight_for_term(
        weighted_MaxCut_hamil, PauliTerm("I0"), 0
    )
    ham_file_label = f"ham_MAXCUT_weighted_-10_to_10_qubits_{num_qubits}"
    hamiltonian = assign_random_weights(weighted_MaxCut_hamil)
    
    cost_function = prepare_cost_function(hamiltonian, backend)

    landscape = generate_landscape(cost_function, sample_points)
    
    # make persistence diagram
    start = datetime.now()

    # analyze landscape (TDA)
    ripser_result = ripser(landscape, maxdim=2) # todo: change maxdim to 2
    # save tda result
    save_persistence_diagrams(ripser_result,N, ham_file_label, f"{N}_LHS", timestamp=timestamp, p=p, id="")
    plot_diagrams(ripser_result["dgms"], show=False) # axis limits fit transformed loss values (to [50,100])
    dir = f"experiment_results\Initial_QAOA_tests\H2\persistence_{timestamp}"
    dir2 = os.path.join(dir, "plots")
    if not os.path.exists(dir2):
        os.mkdir(dir2)
    file_name = f"persistence_N_{N}_p_{p}.png"
    plt.savefig(os.path.join(dir2, file_name))
    plt.close()
    elapsed_time = datetime.now()-start
    print(f"N={N}, num_qubits={num_qubits}, p={p}, elapsed time: ", elapsed_time)


def main():
    """
    Perform TDA on QAOA loss landscape for N=5k, 10k, 15k (number of sample points) and
    dim = 2, 4, 6 (dimension of sample points), i.e. 3D, 5D and 7D data points that are analyzed.
    Computes homologies up to dimension 2 (H_0, H_1, H_2).
    """
    # load loss landscape
    timestamp = generate_timestamp_str()
    # create directory
    dir = os.path.join(os.path.join(os.path.join(os.path.join(os.getcwd(), f"experiment_results"),f"Initial_QAOA_tests"),"H2"),f"persistence_{timestamp}")
    if not os.path.exists(dir):
        os.mkdir(dir)
    # create txt file (overview over runtimes)
    dir2 = os.path.join(dir, f"persistence_runtimes_{timestamp}.txt")
    with open(dir2, "w") as text_file:
        text_file.write("Persistence runtimes (H2, ripser package) for 3qubit QAOA \nN = Number of sample points, Sampling method: Latin Hypercube Sampling")
    # compute persistence diagrams
    for N in [5000, 10000, 15000]:    
        for p in range(1,4): 
            start = datetime.now()
            # load landscape
            ham_file_label = "ham_MAXCUT_weighted_-10_to_10_qubits_3"
            sample_file_label = f"LHS_{N}"
            landscape_file_name = f"landscape_{ham_file_label}_{sample_file_label}_p_{p}.json"
            landscape_dir = os.path.join(
                os.path.join(os.getcwd(), os.path.join("resources/QAOA_H2_test/landscapes", landscape_file_name))
            )
            with open(landscape_dir) as f:
                landscape_dict =  json.load(f)
            landscape = np.asarray(landscape_dict["landscape"])
            

            # analyze landscape (TDA)
            ripser_result = ripser(landscape, maxdim=1) # todo: change maxdim to 2
            # save tda result
            save_persistence_diagrams(ripser_result,N, ham_file_label, sample_file_label, timestamp=timestamp, p=p, id="")

            # create persistence diagram (plot) and save it
            diagrams = ripser_result["dgms"]
            plot_diagrams(diagrams, show=False) # axis limits fit transformed loss values (to [50,100])
            dir2 = os.path.join(dir, "plots")
            if not os.path.exists(dir2):
                os.mkdir(dir2)
            file_name = f"persistence_N_{N}_p_{p}.png"
            plt.savefig(os.path.join(dir2, file_name))
            plt.close()

            # save runtime to txt file
            elapse_time = datetime.now() - start
            output_line = f"N={N}, p={p}, time(H2): {elapse_time}"
            print(output_line)
            dir2 = os.path.join(dir, f"persistence_runtimes_{timestamp}.txt")
            with open(dir2, "a") as text_file:
                text_file.write("\n" + output_line)






if __name__ == "__main__":
    #qaoa_transformed_main()
    #main()
    #generate_loss_landscapes_for_test()
    qaoa_h2_test()
    

