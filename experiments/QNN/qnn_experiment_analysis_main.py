from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import itertools
import json
from pathlib import Path
from persim import bottleneck, bottleneck_matching
from gtda.diagrams import PairwiseDistance
import os
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

from src.qaoa.utils import generate_timestamp_str
from src.utils.file_utils import ripser_list_to_giotto

max_workers = 4
cpu_count = os.cpu_count()
timestamp = generate_timestamp_str()

# Directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
QNN_BASE_DIR = BASE_DIR / "experiment_results" / "QNN"
ANALYSIS_BASE_DIR = BASE_DIR / "experiment_results" / "QNN" / "analysis" / timestamp
METRIC_BASE_DIR = BASE_DIR / "experiment_results" / "QNN" / "analysis" 

def get_qnn_ids(set, unitary=None):
    """
    Determines correct QNN experiment ID list corresponding to number of qubits (if given), p and sample set number
    
    :param num_qubits: number of qubits in [3,6,9,12,15,18]
    :param p: in [1,2,3]
    :param set: in [0,...,4]
    """
    assert set in range(5)
    assert unitary in [0, 1,2,3,4,None]
    if unitary is not None:
        id_list = [set + 125*j + 25*unitary + 5*i for j in range(4) for i in range(5)]
    else:
        id_list = [set + 5*i for i in range(100)]
    return id_list

def test_get_qnn_ids():
    for set in range(5):
        for unitary in range(5):
            id_list = get_qnn_ids(set, unitary)
            unitaries = []
            sets = []
            for id in id_list:
                # read file
                RIPSER_BASE_DIR = QNN_BASE_DIR / "transformed_50_100" / "ripser_results"
                file_ending = "transformed_50_100_H1.json"
                
                file = RIPSER_BASE_DIR / f"persistence_qnn_{id}_{file_ending}"
                r_dict = json.load(open(file))
                unitaries.append(r_dict["unitary"])
                sets.append(r_dict["sample_set"])
            print("unitaries the same:", all(x == unitaries[0] for x in unitaries))
            print("sets the same:", all(x == sets[0] for x in sets))
    

def compute_distance_for_all_using_giotto(set, unitary, metric = "bottleneck", transformed=False):
    assert set in range(5)
    id_list = get_qnn_ids(set, unitary)
    if transformed:
        RIPSER_BASE_DIR = QNN_BASE_DIR / "transformed_50_100" / "ripser_results"
        file_ending = "transformed_50_100_H1.json"
    else:
        RIPSER_BASE_DIR = QNN_BASE_DIR / "not_transformed" / "ripser_results"
        file_ending = "not_transformed_H1.json"
    
    persistence_diagram_list = []
    for id in id_list:
        file = RIPSER_BASE_DIR / f"persistence_qnn_{id}_{file_ending}"
        r_dict = json.load(open(file))
        d = r_dict["persistence diagram"]["dgms"]
        dgm = [np.asarray(v) for v in d]
        persistence_diagram_list.append(dgm)
    giotto_diagrams = ripser_list_to_giotto(persistence_diagram_list)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[INFO] {now}: Starting set={set}, unitary={unitary} ...")
    PD = PairwiseDistance(metric = metric, n_jobs=cpu_count-1, order=None)
    dist_matrices = PD.fit_transform(giotto_diagrams)
    del giotto_diagrams
    del persistence_diagram_list
    return dist_matrices
    
def main(transformed=False, metric="bottleneck"):
    if transformed:
        RIPSER_BASE_DIR = QNN_BASE_DIR / "transformed_50_100" / "ripser_results"
        file_name = "transformed_50_100"
    else:
        RIPSER_BASE_DIR = QNN_BASE_DIR / "not_transformed" / "ripser_results"
        file_name = "not_transformed"
    dist_dict_H0 = {"info": f"{metric} distance between qnn persistence diagrams (homology dimension 0) per sample set.", "source files": str(RIPSER_BASE_DIR), "homology dimension": 0, "python library": f"giotto TDA, approximation of {metric} distance"}
    dist_dict_H1 = {"info": f"{metric} distance between qnn persistence diagrams (homology dimension 1) per sample set.", "source files": str(RIPSER_BASE_DIR), "homology dimension": 1, "python library": f"giotto TDA, approximation of {metric} distance"}
    for set in range(5):
        dist_dict_H0[set] = {}
        dist_dict_H1[set] = {}
        for unitary in range(5):
            id_list = get_qnn_ids(set)
            distance_matrices = compute_distance_for_all_using_giotto(set, unitary, metric=metric, transformed=transformed)
            dist_dict_H0[set][unitary] = {"set": set, "unitary": unitary, "id list": id_list, "distances matrix": distance_matrices[:,:,0].tolist()}
            dist_dict_H1[set][unitary] = {"set": set, "id list": id_list, "distances matrix": distance_matrices[:,:,1].tolist()}
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[DONE] {now}: set={set}, unitary={unitary}")
    # save results
    ANALYSIS_BASE_DIR.mkdir(exist_ok=True)
    file_H0 = ANALYSIS_BASE_DIR / f"QNN_{metric}_{file_name}_H0.json"
    file_H1 = ANALYSIS_BASE_DIR / f"QNN_{metric}_{file_name}_H1.json"
    file_H0.write_text(json.dumps(dist_dict_H0, indent=4))
    file_H1.write_text(json.dumps(dist_dict_H1, indent=4))


def compute_statistics_of_distance_per_p_and_set(metric, homology_dim):
    """
    Compute mean, median, std of metric values between persistence diagrams of different qaoa instances
    
    :param metric: metric used to analyze persistence diagrams, can be bottleneck or wasserstein
    """
    assert metric in ["bottleneck", "wasserstein"]
    assert homology_dim in [0,1]

    result_dict = {"info": "results are matrices of values, lines and columns correspond to # qubits, i.e. [3,6,9,12,15,18]. Distances between same qubit counts only take distances between different runs of same qubit count into account.", "metric": metric, "homology dimension": homology_dim}
    file = METRIC_BASE_DIR / f"QAOA_{metric}_H{homology_dim}.json"
    all_values = json.load(open(file))
    for p in [1,2,3]:
        result_dict[p] = {} 
        for set in range(5):
            result_dict[p][set] = {"p": p, "set": set} 
            curr_values = np.asarray(all_values[str(p)][str(set)]["distances matrix"])
            means = np.zeros((6,6))
            medians = np.zeros((6,6))
            stds = np.zeros((6,6))
            for num_qubits_1 in range(6):
                for num_qubits_2 in range(num_qubits_1,6):
                    # Get distance values
                    min_index1 = num_qubits_1*5
                    max_index1 = min_index1+5
                    min_index2 = num_qubits_2*5
                    max_index2 = min_index2+5
                    block1 = curr_values[min_index1:max_index1, min_index2:max_index2] 
                    block2 = curr_values[min_index2:max_index2, min_index1:max_index1]

                    # Flatten and combine into one pool of values
                    combined_values = np.concatenate([block1.ravel(), block2.ravel()])
                    if(num_qubits_1 == num_qubits_2):
                        # remove zeros from combined values
                        combined_values = combined_values[combined_values != 0]

                    # Compute statistics
                    means[num_qubits_1, num_qubits_2] = np.mean(combined_values)
                    means[num_qubits_2, num_qubits_1] = np.mean(combined_values)
                    medians[num_qubits_1, num_qubits_2] = np.median(combined_values)
                    medians[num_qubits_2, num_qubits_1] = np.median(combined_values)
                    stds[num_qubits_1, num_qubits_2] = np.std(combined_values)
                    stds[num_qubits_2, num_qubits_1] = np.median(combined_values)
            result_dict[p][set]["mean"] = means.tolist()
            result_dict[p][set]["median"] = medians.tolist()
            result_dict[p][set]["std"] = stds.tolist()
    
    # compute statistics over all sets
    for p in [1,2,3]:
        means = np.zeros((6,6))
        medians = np.zeros((6,6))
        stds = np.zeros((6,6))
        result_dict[p]["all"] = {}
        for num_qubits_1 in range(6):
            for num_qubits_2 in range(num_qubits_1,6):
                # Get distance matrix index values
                min_index1 = num_qubits_1*5
                max_index1 = min_index1+5
                min_index2 = num_qubits_2*5
                max_index2 = min_index2+5
                values = []
                for set in range(5):
                    curr_values = np.asarray(all_values[str(p)][str(set)]["distances matrix"])
            
                    block1 = curr_values[min_index1:max_index1, min_index2:max_index2] 
                    block2 = curr_values[min_index2:max_index2, min_index1:max_index1]

                    # Flatten and combine into one pool of values
                    combined_values = np.concatenate([block1.ravel(), block2.ravel()])
                    if(num_qubits_1 == num_qubits_2):
                        # remove zeros from combined values
                        combined_values = combined_values[combined_values != 0]
                    values.append(combined_values)
                # Compute statistics
                means[num_qubits_1, num_qubits_2] = np.mean(values)
                means[num_qubits_2, num_qubits_1] = np.mean(values)
                medians[num_qubits_1, num_qubits_2] = np.median(values)
                medians[num_qubits_2, num_qubits_1] = np.median(values)
                stds[num_qubits_1, num_qubits_2] = np.std(values)
                stds[num_qubits_2, num_qubits_1] = np.std(values)
        result_dict[p]["all"]["mean"] = means.tolist()
        result_dict[p]["all"]["median"] = medians.tolist()
        result_dict[p]["all"]["std"] = stds.tolist()
    # save results
    ANALYSIS_BASE_DIR.mkdir(exist_ok=True)
    file = ANALYSIS_BASE_DIR / f"QAOA_{metric}_H{homology_dim}_statistics.json"
    file.write_text(json.dumps(result_dict, indent=4))

def plot_heatmaps_all_sets(metric, homology_dim, statistic):
    # Load the matrix data
    file = METRIC_BASE_DIR / f"QAOA_{metric}_H{homology_dim}_statistics.json"
    data = json.load(open(file))
    HEATMAPS_DIR = ANALYSIS_BASE_DIR / f"{statistic}_distances"
    ANALYSIS_BASE_DIR.mkdir(exist_ok=True)
    HEATMAPS_DIR.mkdir(exist_ok=True)

    # Configuration for the plot
    sets_to_plot = ['all', '0', '1', '2', '3', '4']
    sup_title = "{statistic} of {metric} distance"
    titles = ["all sample sets", "sample set 0", "sample set 1", 
            "sample set 2", "sample set 3", "sample set 4"]
    qubits = [3, 6, 9, 12, 15, 18]

    # Create a 3x2 grid of subplots
    for p in [1,2,3]:
        fig, axes = plt.subplots(3, 2, figsize=(12, 15))
        axes = axes.flatten()
        for i, set_key in enumerate(sets_to_plot):
            matrix = np.array(data[str(p)][set_key][statistic])
            # Plotting the heatmap
            sns.heatmap(matrix, annot=True, fmt=".2f", ax=axes[i], 
                        xticklabels=qubits, yticklabels=qubits, cmap='viridis')
            
            axes[i].set_title(titles[i])
            axes[i].set_xlabel('Number of Qubits')
            axes[i].set_ylabel('Number of Qubits')

        plt.tight_layout()
        plt.savefig(HEATMAPS_DIR / f'QAOA_heatmap_{metric}_H{homology_dim}_p_{p}_{statistic}.pdf')
        plt.close()

def plot_heatmaps_all_p_all_homologies(metric, statistic):
    # Load the matrix data
    file_H0 = METRIC_BASE_DIR / f"QAOA_{metric}_H0_statistics.json"
    data = {}
    data[0] = json.load(open(file_H0))
    file_H1 = METRIC_BASE_DIR / f"QAOA_{metric}_H1_statistics.json"
    data[1] = json.load(open(file_H1))
    HEATMAPS_DIR = ANALYSIS_BASE_DIR / f"{statistic}_distances"
    ANALYSIS_BASE_DIR.mkdir(exist_ok=True)
    HEATMAPS_DIR.mkdir(exist_ok=True)

    # Configuration for the plot
    sets_to_plot = ['all']
    sup_title = "{statistic} of {metric} distance"
    titles = ["H0, p = 1", "H1, p = 1", "H0, p = 2", "H1, p = 2", "H0, p = 3", "H1, p = 3"]
    qubits = [3, 6, 9, 12, 15, 18]

    # Create a 3x2 grid of subplots
    i = 0
    fig, axes = plt.subplots(3, 2, figsize=(12, 15))
    axes = axes.flatten()
    for p in [1,2,3]:
        for H in [0,1]:
            matrix = np.array(data[H][str(p)]['all'][statistic])
            # Plotting the heatmap
            sns.heatmap(matrix, annot=True, fmt=".2f", ax=axes[i], 
                xticklabels=qubits, yticklabels=qubits, cmap='viridis')
                
            axes[i].set_title(titles[i])
            axes[i].set_xlabel('Number of Qubits')
            axes[i].set_ylabel('Number of Qubits')
            i += 1

    plt.tight_layout()
    plt.savefig(HEATMAPS_DIR / f'QAOA_heatmap_{metric}_all_{statistic}.pdf')
    plt.close()
    del data

if __name__=="__main__":
    for metric in ["wasserstein", "bottleneck"]:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[INFO] {now}: Starting {metric}, transformed")
        main(transformed=True, metric=metric)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[INFO] {now}: Starting {metric}, not transformed")
        main(transformed=False, metric=metric)