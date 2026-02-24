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


def compute_statistics_of_distance_set(metric, homology_dim, transformed=True):
    """
    Compute mean, median, std of metric values between persistence diagrams of different qnn instances
    
    :param metric: metric used to analyze persistence diagrams, can be bottleneck or wasserstein
    """
    assert metric in ["bottleneck", "wasserstein"]
    assert homology_dim in [0,1]
    if transformed: file_name = "transformed_50_100"
    else: file_name = "not_transformed"

    result_dict = {"info": "results are matrices of values, lines and columns correspond to Schmidt rank, i.e. [1,2,3,4]. Distances between same schmid ranks only take distances between different runs of same qubit count into account.", "metric": metric, "homology dimension": homology_dim, "transformed": file_name}
    file = METRIC_BASE_DIR / f"QNN_{metric}_{file_name}_H{homology_dim}.json"
    all_values = json.load(open(file))
    
    for unitary in range(5):
        result_dict[unitary] = {}
        for set in range(5):
            result_dict[unitary][set] = {"unitary": unitary, "set": set} 
            curr_values = np.asarray(all_values[str(set)][str(unitary)]["distances matrix"])
            means = np.zeros((4,4))
            medians = np.zeros((4,4))
            stds = np.zeros((4,4))
            for srank_1 in range(4):
                for srank_2 in range(srank_1,4):
                    # Get distance values
                    min_index1 = srank_1*5
                    max_index1 = min_index1+5
                    min_index2 = srank_2*5
                    max_index2 = min_index2+5
                    block1 = curr_values[min_index1:max_index1, min_index2:max_index2] 
                    block2 = curr_values[min_index2:max_index2, min_index1:max_index1]

                    # Flatten and combine into one pool of values
                    combined_values = np.concatenate([block1.ravel(), block2.ravel()])
                    if(srank_1 == srank_2):
                        # remove zeros from combined values
                        combined_values = combined_values[combined_values != 0]

                    # Compute statistics
                    means[srank_1, srank_2] = np.mean(combined_values)
                    means[srank_2, srank_1] = np.mean(combined_values)
                    medians[srank_1, srank_2] = np.median(combined_values)
                    medians[srank_2, srank_1] = np.median(combined_values)
                    stds[srank_1, srank_2] = np.std(combined_values)
                    stds[srank_2, srank_1] = np.median(combined_values)
            result_dict[unitary][set]["mean"] = means.tolist()
            result_dict[unitary][set]["median"] = medians.tolist()
            result_dict[unitary][set]["std"] = stds.tolist()
    
    # compute statistics over all sets
    for unitary in range(5):
        means = np.zeros((4,4))
        medians = np.zeros((4,4))
        stds = np.zeros((4,4))
        result_dict[unitary]["all"] = {}
        for srank_1 in range(4):
            for srank_2 in range(srank_1,4):
                # Get distance matrix index values
                min_index1 = srank_1*5
                max_index1 = min_index1+5
                min_index2 = srank_2*5
                max_index2 = min_index2+5
                values = []
                for set in range(5):
                    curr_values = np.asarray(all_values[str(set)][str(unitary)]["distances matrix"])
            
                    block1 = curr_values[min_index1:max_index1, min_index2:max_index2] 
                    block2 = curr_values[min_index2:max_index2, min_index1:max_index1]

                    # Flatten and combine into one pool of values
                    combined_values = np.concatenate([block1.ravel(), block2.ravel()])
                    if(srank_1 == srank_2):
                        # remove zeros from combined values
                        combined_values = combined_values[combined_values != 0]
                    values.append(combined_values)
                # Compute statistics
                means[srank_1, srank_2] = np.mean(values)
                means[srank_2, srank_1] = np.mean(values)
                medians[srank_1, srank_2] = np.median(values)
                medians[srank_2, srank_1] = np.median(values)
                stds[srank_1, srank_2] = np.std(values)
                stds[srank_2, srank_1] = np.std(values)
        result_dict[unitary]["all"]["mean"] = means.tolist()
        result_dict[unitary]["all"]["median"] = medians.tolist()
        result_dict[unitary]["all"]["std"] = stds.tolist()
    
    
    # compute statistics over all unitaries
    means = np.zeros((4,4))
    medians = np.zeros((4,4))
    stds = np.zeros((4,4))
    result_dict["all"] = {}
    
    for srank_1 in range(4):
        for srank_2 in range(srank_1,4):
            values = []
            for unitary in range(5):
                # Get distance matrix index values
                min_index1 = srank_1*5
                max_index1 = min_index1+5
                min_index2 = srank_2*5
                max_index2 = min_index2+5
                for set in range(5):
                    curr_values = np.asarray(all_values[str(set)][str(unitary)]["distances matrix"])
            
                    block1 = curr_values[min_index1:max_index1, min_index2:max_index2] 
                    block2 = curr_values[min_index2:max_index2, min_index1:max_index1]

                    # Flatten and combine into one pool of values
                    combined_values = np.concatenate([block1.ravel(), block2.ravel()])
                    if(srank_1 == srank_2):
                        # remove zeros from combined values
                        combined_values = combined_values[combined_values != 0]
                    values.append(combined_values)
            
            # Compute statistics
            means[srank_1, srank_2] = np.mean(values)
            means[srank_2, srank_1] = np.mean(values)
            medians[srank_1, srank_2] = np.median(values)
            medians[srank_2, srank_1] = np.median(values)
            stds[srank_1, srank_2] = np.std(values)
            stds[srank_2, srank_1] = np.std(values)
        result_dict["all"]["mean"] = means.tolist()
        result_dict["all"]["median"] = medians.tolist()
        result_dict["all"]["std"] = stds.tolist()


    # save results
    ANALYSIS_BASE_DIR.mkdir(exist_ok=True)
    file = ANALYSIS_BASE_DIR / f"QNN_{file_name}_{metric}_H{homology_dim}_statistics.json"
    file.write_text(json.dumps(result_dict, indent=4))

def plot_heatmaps_all_unitaries(metric, homology_dim, statistic, transformed=True):
    assert metric in ["bottleneck", "wasserstein"]
    assert homology_dim in [0,1]
    if transformed: file_name = "transformed_50_100"
    else: file_name = "not_transformed"

    # Load the matrix data
    file = METRIC_BASE_DIR / f"QNN_{file_name}_{metric}_H{homology_dim}_statistics.json"
    data = json.load(open(file))
    HEATMAPS_DIR = ANALYSIS_BASE_DIR / f"{statistic}_distances"
    ANALYSIS_BASE_DIR.mkdir(exist_ok=True)
    HEATMAPS_DIR.mkdir(exist_ok=True)



    # Configuration for the plot
    unitaries_to_plot = ['all', '0', '1', '2', '3', '4']
    sup_title = "{statistic} of {metric} distance"
    titles = ["all unitaries", "unitary 0", "unitary 1", 
            "unitary 2", "unitary 3", "unitary 4"]
    sranks = [1,2,3,4]

    # determine colorbar limits
    all_values = []
    for i, set_key in enumerate(unitaries_to_plot):
        if set_key == "all": all_values.extend(np.array(data[set_key][statistic]))
        else:
            all_values.extend(np.array(data[set_key]["all"][statistic]))
    global_min = np.min(all_values)
    global_max = np.max(all_values)

    # Create a 3x2 grid of subplots
    fig, axes = plt.subplots(3, 2, figsize=(12, 15))
    axes = axes.flatten()
    for i, set_key in enumerate(unitaries_to_plot):
        if set_key == "all": matrix = np.array(data[set_key][statistic])
        else:
            matrix = np.array(data[set_key]["all"][statistic])
        # Plotting the heatmap
        sns.heatmap(matrix, annot=True, fmt=".2f", ax=axes[i], 
                        xticklabels=sranks, yticklabels=sranks, cmap='viridis',
                        vmin=global_min, vmax=global_max)
            
        axes[i].set_title(titles[i])
        axes[i].set_xlabel('Schmidt Rank')
        axes[i].set_ylabel('Schmidt Rank')

    plt.tight_layout()
    plt.savefig(HEATMAPS_DIR / f'QNN_{file_name}_heatmap_{metric}_H{homology_dim}_{statistic}.png')
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
            axes[i].set_xlabel('Schmidt Rank')
            axes[i].set_ylabel('Schmidt Rank')
            i += 1

    plt.tight_layout()
    plt.savefig(HEATMAPS_DIR / f'QAOA_heatmap_{metric}_all_{statistic}.pdf')
    plt.close()
    del data

if __name__=="__main__":
    for transformed in [True, False]:
        for hdim in [0,1]:
            plot_heatmaps_all_unitaries(metric="bottleneck", homology_dim=hdim, statistic="mean", transformed=transformed)
            plot_heatmaps_all_unitaries(metric="bottleneck", homology_dim=hdim, statistic="median", transformed=transformed)
            plot_heatmaps_all_unitaries(metric="bottleneck", homology_dim=hdim, statistic="std", transformed=transformed)