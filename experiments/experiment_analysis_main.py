from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import itertools
import json
from pathlib import Path
from persim import bottleneck, bottleneck_matching
from gtda.diagrams import PairwiseDistance
import os

import numpy as np

from src.qaoa.utils import generate_timestamp_str
from src.utils.file_utils import ripser_list_to_giotto

max_workers = 4
cpu_count = os.cpu_count()
timestamp = generate_timestamp_str()

# Directories
BASE_DIR = Path(__file__).resolve().parent.parent
RIPSER_BASE_DIR = BASE_DIR / "experiment_results" / "QAOA" / "ripser_results"
ANALYSIS_BASE_DIR = BASE_DIR / "experiment_results" / "QAOA" / "analysis" / timestamp

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


if __name__=="__main__":
    #main()
    print(os.getcwd())
    generate_timestamp_str()