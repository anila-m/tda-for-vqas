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

from metrics_experiment import get_qaoa_ids, get_qnn_ids
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


backend = QulacsSimulator()
# min_gamma=-np.pi
max_gamma=np.pi
# min_beta=-np.pi/2
max_beta=np.pi/2
min_gamma= - np.pi #-3.05
min_beta= - np.pi/2 #-1.5
qaoa_grid_size = [707,26,9]
#qaoa_grid_size = [1000, 33, 10] # grid sizes for QAOA, keys are p values, all grids have roughly 1M grid points
#qaoa_grid_size = [707, 26, 9] # lower grid sized for QAPA, all grids have roughly 500K grid points
cpu_count = os.cpu_count() 


def create_roughness_boxplot_for_QAOA_per_p(roughness_metric):
    assert roughness_metric in [
        "average absolute scalar curvature",
        "total variation",
        "fourier density",
    ]

    # load file
    file = (
        BASE_DIR
        / "experiment_results"
        / "QAOA"
        / "roughness_metrics_results"
        / "QAOA_roughness_metrics_2026_04_28_06_31_25_cleaned.json"
    )

    metric_dict = json.load(open(file))

    SAVE_DIR = (
        BASE_DIR
        / "experiment_results"
        / "QAOA"
        / "roughness_metrics_results"
    )
    # boxplot per number of qubits and p
    for p in [1,2,3]:
        i = 0
        value_dict = {}
        for num_qubits in [3,6,9,12,15,18]:
            id_list = get_qaoa_ids(p=p, num_qubits=num_qubits)
            values = []
            for id in id_list:
                values.append(metric_dict[str(id)][roughness_metric])
            value_dict[num_qubits] = values
            i += 1

        # create boxplot
        fig, ax = plt.subplots(figsize=(8, 5))

        x_labels = list(value_dict.keys())
        data = [value_dict[k] for k in x_labels]

        ax.boxplot(data, tick_labels=x_labels)

        ax.set_xlabel("Number of qubits")
        ax.set_ylabel(roughness_metric.capitalize())

        # no title

        plt.tight_layout()

        filename = f"QAOA_{roughness_metric.replace(' ', '_')}_boxplot_for_p_{p}.pdf"

        plt.savefig(SAVE_DIR / filename, bbox_inches="tight")
        plt.close()
  
def create_roughness_boxplot_for_QAOA_per_num_qubits(roughness_metric):
    assert roughness_metric in [
        "average absolute scalar curvature",
        "total variation",
        "fourier density",
    ]

    # load file
    file = (
        BASE_DIR
        / "experiment_results"
        / "QAOA"
        / "roughness_metrics_results"
        / "QAOA_roughness_metrics_2026_04_28_06_31_25_cleaned.json"
    )

    metric_dict = json.load(open(file))

    SAVE_DIR = (
        BASE_DIR
        / "experiment_results"
        / "QAOA"
        / "roughness_metrics_results"
    )

    # one figure per number of qubits
    for num_qubits in [3, 6, 9, 12, 15, 18]:
        value_dict = {}
        for p in [1, 2, 3]:
            id_list = get_qaoa_ids(p=p, num_qubits=num_qubits)
            values = []
            for id in id_list:
                values.append(metric_dict[str(id)][roughness_metric])
            value_dict[p] = values

        # create boxplot
        fig, ax = plt.subplots(figsize=(8, 5))
        x_labels = list(value_dict.keys())
        data = [value_dict[k] for k in x_labels]

        ax.boxplot(data, tick_labels=x_labels)

        ax.set_xlabel("Number of QAOA layers $p$")
        ax.set_ylabel(roughness_metric.capitalize())

        # no title

        plt.tight_layout()

        filename = (f"QAOA_{roughness_metric.replace(' ', '_')}_boxplot_for_num_qubits_{num_qubits}.pdf")

        plt.savefig(SAVE_DIR / filename, bbox_inches="tight")
        plt.close()
  

def create_roughness_boxplot_for_QNN(roughness_metric):
    directory = BASE_DIR / "experiment_results" / "QNN" / "transformed_50_100" / "roughness_metrics_results"
    filename = "QNN_roughness_metrics_2026_03_21_20_23_46_cleaned.json"
    file = directory / filename
    metric_dict = json.load(open(file))
    value_dict = {}
    for s in [1,2,3,4]:
        ids = get_qnn_ids(s_rank=s)
        print(ids)
        values = []
        for id in ids:
            values.append(metric_dict[str(id)][roughness_metric])
        value_dict[s] = values

    # create boxplot
    fig, ax = plt.subplots(figsize=(8, 5))
    x_labels = list(value_dict.keys())
    data = [value_dict[k] for k in x_labels]

    ax.boxplot(data, tick_labels=x_labels)

    ax.set_xlabel("Schmidt Rank")
    ax.set_ylabel(roughness_metric.capitalize())

    # no title

    plt.tight_layout()

    filename = (f"QNN_{roughness_metric.replace(' ', '_')}_boxplot_for_Schmidt_rank.pdf")

    plt.savefig(directory / filename, bbox_inches="tight")
    plt.close()

def diagrams_QAOA_BP_NG():
  # Load json file
  directory = BASE_DIR / "experiment_results" / "BP_NG" / "small_excerpt" / "LHS_samples" / "roughness_metrics"
  filename = "qaoa_id_20_roughness_metric_BP_NG_grid.json"
  file = directory / filename
  data = json.load(open(file))

  k_list = []
  total_variation = []
  fourier_density = []
  curvature = []

  # Extract data points from indices '0' to '3' (not 4, since its outside the narrow gorge)
  for key in ['0', '1', '2', '3']:
      if key in data:
          entry = data[key]
          k_list.append(entry['k'])
          total_variation.append(entry['total variation'])
          fourier_density.append(entry['fourier density'])
          curvature.append(entry['average absolute scalar curvature'])

  k_list = np.array(k_list)

  # Sort by k to ensure sequential order (0, 1, 2, 3, 4)
  sort_idx = np.argsort(k_list)
  k_values = k_list[sort_idx]
  total_variation = np.array(total_variation)[sort_idx]
  fourier_density = np.array(fourier_density)[sort_idx]
  curvature = np.array(curvature)[sort_idx]

  # x-axis tick labels
  tick_labels = [r'$1$', r'$1/2$', r'$1/4$', r'$1/8$']


  # Diagram 1: Total Variation and Fourier Density (Linear-spaced K Axis)
  fig, ax1 = plt.subplots(figsize=(4.5, 3.8))

  color_tv = 'tab:blue'
  ax1.set_xticks(k_values)
  ax1.set_xticklabels(tick_labels)
  ax1.set_ylabel('Total Variation', color=color_tv, fontsize=12)
  line1 = ax1.plot(k_values, total_variation, color=color_tv, marker='o', label='Total Variation')
  ax1.tick_params(axis='y', labelcolor=color_tv)

  # second y-axis  for FD
  ax2 = ax1.twinx()  
  color_fd = 'tab:orange'
  ax2.set_ylabel('Fourier Density', color=color_fd, fontsize=12)
  line2 = ax2.plot(k_values, fourier_density, color=color_fd, marker='s', label='Fourier Density')
  ax2.tick_params(axis='y', labelcolor=color_fd)

  # Combine legends from both axes into a single legend box
  lines = line1 + line2
  labels = [l.get_label() for l in lines]
  #ax1.legend(lines, labels, loc='upper right')

  fig.tight_layout()
  fig.savefig(directory / 'diagram_TV_FD.pdf', bbox_inches="tight")
  plt.close(fig)

  # diagram 2: ASC
  fig, ax = plt.subplots(figsize=(4.5, 3.8))

  ax.set_xticks(k_values)
  ax.set_xticklabels(tick_labels)
  ax.set_ylabel('Mean Absolute SC', fontsize=12)
  ax.plot(k_values, curvature, color='tab:blue', marker='^', label='Scalar Curvature')

  # Format y-axis with scientific notation
  ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0)) 

  fig.tight_layout()
  fig.savefig(directory / 'diagram_MASC.pdf', bbox_inches="tight")
  plt.close(fig)
      

if __name__ == "__main__":
    diagrams_QAOA_BP_NG()
        