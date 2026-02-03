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
RIPSER_BASE_DIR = BASE_DIR / "experiment_results" / "QAOA" / "ripser_results"
ANALYSIS_BASE_DIR = BASE_DIR / "experiment_results" / "QAOA" / "analysis" / timestamp
METRIC_BASE_DIR = BASE_DIR / "experiment_results" / "QAOA" / "analysis" 

def get_qaoa_ids(p, set, num_qubits = None):
    """
    Determines correct QAOA experiment ID list corresponding to number of qubits (if given), p and sample set number
    
    :param num_qubits: number of qubits in [3,6,9,12,15,18]
    :param p: in [1,2,3]
    :param set: in [0,...,4]
    """
    assert p in [1,2,3]
    assert num_qubits in [3,6,9,12,15,18, None]
    assert set in range(5)
    if num_qubits == None:
        id_list = [(p-1)*(5*6*5)+set*(6*5)+i for i in range(30)]
    else:
        id_list = [(p-1)*(5*6*5)+set*(6*5)+(num_qubits//3-1)*5+run for run in range(5)]
    return id_list


def compute_distances_between_persistence_diagrams(num_qubits1, num_qubits2, p, set, h_dim, metric="bottleneck", library="giotto"):
    """
    Compute list of distances between persistencediagrams of loss landscapes of QAOA instances with two different qubit numbers,
    but fixed p and sample set (used to compute the loss landscape).
    
    :param num_qubits1: Description
    :param num_qubits2: Description
    :param p: Description
    :param set: Description
    :param distance: Description
    :param library: python library used to compute bottleneck distance, can be "scikit-tda" or "giotto", default: giotto
    """
    #assert h_dim in [0,1]
    assert num_qubits1 in [3,6,9,12,15,18]
    assert num_qubits2 in [3,6,9,12,15,18]
    assert p in [1,2,3]
    assert set in range(5)

    id_list1 = get_qaoa_ids(p, set, num_qubits1)
    id_list2 = get_qaoa_ids(p, set, num_qubits2)
    ANALYSIS_BASE_DIR.mkdir(exist_ok=True)

    distance_list = []
    
    if(library == "scikit-tda"):
        if(num_qubits1 == num_qubits2):
            id_tuples = []
            for k, id1 in enumerate(id_list1):
                for id2 in id_list1[k+1:]:
                    id_tuples.append(tuple((id1, id2)))
            if(metric == "bottleneck"):
                with ProcessPoolExecutor(max_workers=max_workers) as exe:
                    futures = [exe.submit(compute_bottleneck_distance, id_1, id_2, h_dim) for (id_1,id_2) in id_tuples]            
                    # await results & save them:
                    for future in as_completed(futures):
                        dist = future.result()
                        distance_list.append(dist)
        else:
            if(metric == "bottleneck"):
                with ProcessPoolExecutor(max_workers=max_workers) as exe:
                    futures = [exe.submit(compute_bottleneck_distance, id_1, id_2, h_dim) for (id_1,id_2) in list(itertools.product(id_list1, id_list2))]            
                    # await results & save them:
                    for future in as_completed(futures):
                        dist = future.result()
                        distance_list.append(dist)
    elif (library == "giotto"):
        if(num_qubits1 == num_qubits2):
            id_tuples = []
            for k, id1 in enumerate(id_list1):
                for id2 in id_list1[k+1:]:
                    distance = compute_distance_using_giotto(id1, id2, metric=metric)
                    distance_list.append(distance)

        else:
            for (id_1,id_2) in list(itertools.product(id_list1, id_list2)):
                distance = compute_distance_using_giotto(id_1, id_2, metric=metric)
                distance_list.append(distance)


    return distance_list

def compute_distance_using_giotto(id_1, id_2, metric):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    PD = PairwiseDistance(metric = metric, n_jobs=cpu_count-1, order=None)
    print(f"[INFO] {now}: {id_1}, {id_2}, {metric}")
    file_1 = RIPSER_BASE_DIR / f"persistence_qaoa_{id_1}_not_transformed_H1.json"
    r_dict = json.load(open(file_1))
    d = r_dict["persistence diagram"]["dgms"]
    d1 = [np.asarray(v) for v in d]
    # convert to giotto compatible diagram
    file_2 = RIPSER_BASE_DIR / f"persistence_qaoa_{id_2}_not_transformed_H1.json"
    r_dict = json.load(open(file_2))
    d = r_dict["persistence diagram"]["dgms"]
    d2  = [np.asarray(v) for v in d]
    diagrams = ripser_list_to_giotto([d1, d2])
    #compute distance
    distance = PD.fit_transform(diagrams)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[DONE] {now}: {id_1}, {id_2}")
    return distance

def compute_bottleneck_distance(id_1, id_2, h_dim):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[INFO] {now}: {id_1}, {id_2}")
    file_1 = RIPSER_BASE_DIR / f"persistence_qaoa_{id_1}_not_transformed_H1.json"
    r_dict = json.load(open(file_1))
    d = r_dict["persistence diagram"]["dgms"]
    dgm_1 = [np.asarray(v) for v in d]

    file_2 = RIPSER_BASE_DIR / f"persistence_qaoa_{id_2}_not_transformed_H1.json"
    r_dict = json.load(open(file_2))
    d = r_dict["persistence diagram"]["dgms"]
    dgm_2 = [np.asarray(v) for v in d]
    dist = bottleneck(dgm_1[h_dim], dgm_2[h_dim])
    #dist = 0
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[DONE] {now}: {id_1}, {id_2}")
    return dist


def compute_distance_for_all_using_giotto(p, set, metric = "bottleneck"):
    assert p in [1,2,3]
    assert set in range(5)
    id_list = get_qaoa_ids(p, set)
    
    persistence_diagram_list = []
    for id in id_list:
        file = RIPSER_BASE_DIR / f"persistence_qaoa_{id}_not_transformed_H1.json"
        r_dict = json.load(open(file))
        d = r_dict["persistence diagram"]["dgms"]
        dgm = [np.asarray(v) for v in d]
        persistence_diagram_list.append(dgm)
    giotto_diagrams = ripser_list_to_giotto(persistence_diagram_list)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[INFO] {now}: Starting p={p}, set={set} ...")
    PD = PairwiseDistance(metric = metric, n_jobs=cpu_count-2, order=None)
    dist_matrices = PD.fit_transform(giotto_diagrams)
    del giotto_diagrams
    del persistence_diagram_list
    return dist_matrices
    
def main():
    metric = "wasserstein"
    dist_dict_H0 = {"info": f"{metric} distance between qaoa persistence diagrams (homology dimension 0) per p and sample set.", "source files": str(RIPSER_BASE_DIR), "homology dimension": 0, "python library": f"giotto TDA, approximation of {metric} distance"}
    dist_dict_H1 = {"info": f"{metric} distance between qaoa persistence diagrams (homology dimension 1) per p and sample set.", "source files": str(RIPSER_BASE_DIR), "homology dimension": 1, "python library": f"giotto TDA, approximation of {metric} distance"}
    for p in [1, 2, 3]:
        dist_dict_H0[p] = {}
        dist_dict_H1[p] = {}
        for set in range(5):
            id_list = get_qaoa_ids(p, set)
            distance_matrices = compute_distance_for_all_using_giotto(p, set, metric=metric)
            dist_dict_H0[p][set] = {"p": p, "set": set, "id list": id_list, "distances matrix": distance_matrices[:,:,0].tolist()}
            dist_dict_H1[p][set] = {"p": p, "set": set, "id list": id_list, "distances matrix": distance_matrices[:,:,1].tolist()}
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[DONE] {now}: p={p}, set={set}")
    # save results
    ANALYSIS_BASE_DIR.mkdir(exist_ok=True)
    file_H0 = ANALYSIS_BASE_DIR / f"QAOA_{metric}_H0.json"
    file_H1 = ANALYSIS_BASE_DIR / f"QAOA_{metric}_H1.json"
    file_H0.write_text(json.dumps(dist_dict_H0, indent=4))
    file_H1.write_text(json.dumps(dist_dict_H1, indent=4))


def main_old():
    set = 0
    qubit_counts = [3,6,9,12,15,18]
    h_dim = "all"
    p=1
    distance_dict = {"set": 0, "h_dim": h_dim, "library": "giotto"}
    ANALYSIS_BASE_DIR.mkdir(exist_ok=True)
    for k, q1 in enumerate(qubit_counts):
        for q2 in qubit_counts[k:]:
            distance_list = compute_distances_between_persistence_diagrams(q1, q2, p, set, h_dim, metric="bottleneck")
            print(f"[DONE] {q1} vs. {q2}")

            distance_dict[f"{q1},{q2}"] = {"p": p, "bottleneck distances": distance_list}
            # distance_dict["mean"] = np.mean(distance_list)
            # distance_dict["median"] = np.median(distance_list)
            # distance_dict["min"] = np.min(distance_list)
            # distance_dict["max"] = np.max(distance_list)
            # distance_dict["std"] = np.std(distance_list)

            # save distances
                    
            text_file = ANALYSIS_BASE_DIR / f"bottleneck_dist_p_{p}_set_{set}_hdim_{h_dim}.txt"
            with open(text_file, "a") as f:
                f.write(f"{q1},{q2}:{distance_list}\n")
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[DONE] {now}: {q1} vs. {q2}, p={p}")
    

    json_file = ANALYSIS_BASE_DIR / f"bottleneck_dist_p_{p}_set_{set}_hdim_{h_dim}.json"
    json_file.write_text(json.dumps(distance_dict, indent=4))

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
    #main()
    for statistic in ["mean", "std", "median"]:
        for metric in ["bottleneck", "wasserstein"]:
            plot_heatmaps_all_p_all_homologies(metric, statistic)
            for H in [0,1]:
                plot_heatmaps_all_sets(metric, H, statistic)