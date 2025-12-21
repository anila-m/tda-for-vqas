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

from src.qaoa.data_generation import generate_data, prepare_cost_function, save_hamiltonians, perform_2D_scan, get_grid_samples
from src.qaoa.hamiltonian_generation import assign_random_weights, assign_weight_for_term
from src.qaoa.plots import create_plot, create_tda_plot
from src.utils import calculate_plot_extents, generate_timestamp_str

import orqviz
from orqviz.scans import Scan2DResult
from typing import Any, Dict, List, Optional, Tuple
from ripser import ripser
from persim import plot_diagrams
from matplotlib import pyplot as plt


backend = QulacsSimulator()
scan_resolution = 101 # 31^2 approx 1000, was 101, but too much for persistene
num_runs=1


def main():
    cost_period = [np.pi, np.pi]
    fourier_period = [2 * np.pi, 2 * np.pi]
    timestamp = generate_timestamp_str()

    # generate and save grid sample points to use later
    for p in range(1,4):
        # create correct directory if it doesn't exist
        n=2*p
        directory_name = os.path.join(
            os.path.join(os.getcwd(), f"resources\sample_points_{n}D")
        )
        if not os.path.exists(directory_name):
            os.mkdir(directory_name)

        # generate sample points
        min = np.ones(2*p)*(-0.5*np.pi)
        max = np.ones(2*p)*(0.5*np.pi)
        sample_points = get_grid_samples(min, max, dim=2*p, number_of_samples=10000)
        N = len(sample_points)
        
        # save sample points as json
        file_name = f"samples_grid_10000_-0.5pi_0.5pi.json" 
        #10 000 is not the actual number of sample points generated, see get_grid_samples, what actual number of sample points ist
        with open(os.path.join(directory_name, file_name), "w") as f:
            json.dump(sample_points.tolist(), f)

    # generate and save hamiltonians
    hamiltonians_dict: Dict[str, PauliSum] = {}
    for num_qubits in [3, 8, 12, 15]:
        for run in range(num_runs):
            G_complete_graph = nx.complete_graph(num_qubits)
            weighted_MaxCut_hamil = MaxCut().get_hamiltonian(G_complete_graph)
            weighted_MaxCut_hamil = assign_weight_for_term(
                weighted_MaxCut_hamil, PauliTerm("I0"), 0
            )

            hamiltonians_dict[
                f"ham_MAXCUT_weighted_-10_to_10_qubits_{num_qubits}_run_{run}"
            ] = assign_random_weights(weighted_MaxCut_hamil)

    save_hamiltonians(hamiltonians_dict, timestamp_str=timestamp)

    for file_label, hamiltonian in hamiltonians_dict.items():
        print(file_label)
        for p in range(1,4):
            print(f"Dimension: {2*p}")
            # save 
            start = datetime.now()
            cost_function = prepare_cost_function(hamiltonian, backend)

            # load correct sample points
            n=2*p
            s = "samples_grid_10000_-0.5pi_0.5pi"
            directory_name = os.path.join(
                os.path.join(os.getcwd(), f"resources\sample_points_{n}D\{s}.json")
            )
            with open(directory_name) as f:
                sample_points =  np.asarray(json.load(f))
        
            # generate loss landscape
            landscape = generate_landscape(cost_function, sample_points)
            #print(len(landscape))
            elapsed_time = datetime.now()-start
            print(f"Loss Landscape:", elapsed_time)

            # save landscape
            save_landscape(landscape, ham_file_label=file_label, sample_points_file_label=s, timestamp=timestamp, p=p)

            # TDA: compute persistence diagram and save it, generate persistence diagram plot
            start = datetime.now()
            diagrams = ripser(landscape, maxdim=1)['dgms']
            print(diagrams)
            elapsed_time = datetime.now()-start
            print(f"Persistence Diagram:", elapsed_time)
            save_persistence_diagrams(diagrams, ham_file_label=file_label, sample_points_file_label=s, timestamp=timestamp, p=p)
            plot_diagrams(diagrams, show=False) # axis limits fit transformed loss values (to [50,100])
            dir = os.path.join(
                os.path.join(os.getcwd(), f"results\persistence_{timestamp}\plots")
            )
            if not os.path.exists(directory_name):
                os.mkdir(dir)
            plt.savefig(f"results\persistence_{timestamp}\plots\persistence_{file_label}_p_{p}.png")

            # period, fourier_res_x, fourier_res_y = calculate_plot_extents(hamiltonian)
            # scan2D_result, fourier_result, metrics_dict = generate_data(
            #     cost_function,
            #     origin=origin,
            #     dir_x=dir_x,
            #     dir_y=dir_y,
            #     file_label=file_label,
            #     scan_resolution=scan_resolution,
            #     cost_period=cost_period,
            #     fourier_period=fourier_period,
            #     timestamp_str=timestamp,
            # )
            # scan2D_result = generate_2D_scan(cost_function, origin, dir_x, dir_y, file_label, scan_resolution, cost_period)
            # print("[INFO] generate 2D scans done")
            
            # cost_landscape = np.vstack(np.dstack((scan2D_result.params_grid, scan2D_result.values)))
            # diagrams = ripser(scan2D_result.values, maxdim=2)['dgms']
            # #title=f"QAOA {filelabel}"
            # plot_diagrams(diagrams, show=False) # axis limits fit transformed loss values (to [50,100])
            # plt.savefig(f"results/other/onlyCostValues_{file_label}.png")
            # print(file_label, "Min/Max Cost: ", np.min(scan2D_result.values), np.max(scan2D_result.values))
            # create_tda_plot(
            #     scan2D_result,
            #     label=file_label,
            #     fourier_res_x=fourier_res_x,
            #     fourier_res_y=fourier_res_y,
            #     timestamp_str=timestamp,
            #     unit="pi",
            #     remove_constant=True,
            #     include_all_metrics=False,
            # )


def save_landscape(landscape, ham_file_label, sample_points_file_label, timestamp, p):
    directory_name = os.path.join(
        os.path.join(os.getcwd(), f"results\landscapes_{timestamp}")
    )
    # check if directory exists and create it if not

    if not os.path.exists(os.path.join(os.getcwd(), "results")):
        os.mkdir(os.path.join(os.getcwd(), "results"))
    if not os.path.exists(directory_name):
        os.mkdir(directory_name)
    file_name = f"landscape_{ham_file_label}_p_{p}.json"
    dict = {
        "hamiltonian": ham_file_label,
        "sample_points": sample_points_file_label,
        "timestamp": timestamp,
        "P": p,
        "landscape": landscape.tolist()
    }
    with open(os.path.join(directory_name, file_name), "w") as f:
        json.dump(dict, f)

def save_persistence_diagrams(diagrams, ham_file_label, sample_points_file_label, timestamp, p):
    directory_name = os.path.join(
        os.path.join(os.getcwd(), f"results\persistence_{timestamp}")
    )
    # check if directory exists and create it if not

    if not os.path.exists(os.path.join(os.getcwd(), "results")):
        os.mkdir(os.path.join(os.getcwd(), "results"))
    if not os.path.exists(directory_name):
        os.mkdir(directory_name)
    file_name = f"persistence_{ham_file_label}_p_{p}.json"
    d = {k:v.tolist() for k,v in diagrams.items()}
    dict = {
        "hamiltonian": ham_file_label,
        "sample_points": sample_points_file_label,
        "timestamp": timestamp,
        "P": p,
        "persistence diagram": d
    }
    with open(os.path.join(directory_name, file_name), "w") as f:
        json.dump(dict, f)

def generate_landscape(cost_func, sample_points):
    landscape = []
    for _, s in enumerate(sample_points):
        cost = cost_func(s)
        data_point = np.append(s, cost)
        landscape.append(data_point)
    return np.asarray(landscape)

def generate_2D_scan(
        cost_function: orqviz.aliases.LossFunction,
        origin: orqviz.aliases.ParameterVector,
        dir_x: orqviz.aliases.DirectionVector,
        dir_y: orqviz.aliases.DirectionVector,
        file_label: str,  # perhaps a string representing the operator
        scan_resolution: int,
        cost_period: List[float],
        end_points: Tuple[float, float] = (-np.pi, np.pi),
        ) -> Scan2DResult:
    dir_x = dir_x / np.linalg.norm(dir_x)
    dir_y = dir_y / np.linalg.norm(dir_y)

    scan2D_result = perform_2D_scan(
        cost_function, origin, scan_resolution, dir_x, dir_y, cost_period, end_points
    )
    #print(scan2D_result.params_grid)
    #print(scan2D_result.values)
    return scan2D_result

if __name__ == "__main__":
    main()
