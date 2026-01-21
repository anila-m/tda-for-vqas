from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import itertools
import json
from pathlib import Path
from persim import bottleneck, bottleneck_matching

import numpy as np

max_workers = 4

# Directories
BASE_DIR = Path(__file__).resolve().parent.parent
RIPSER_BASE_DIR = BASE_DIR / "experiment_results" / "QAOA" / "ripser_results"
ANALYSIS_BASE_DIR = BASE_DIR / "experiment_results" / "QAOA" / "analysis"

def get_qaoa_ids(num_qubits, p, set):
    """
    Determines correct QAOA experiment ID list corresponding to number of qubits, p and sample set number
    
    :param num_qubits: number of qubits in [3,6,9,12,15,18]
    :param p: in [1,2,3]
    :param set: in [0,...,4]
    """
    assert p in [1,2,3]
    assert num_qubits in [3,6,9,12,15,18]
    assert set in range(5)

    id_list = [(p-1)*(5*6*5)+set*(6*5)+(num_qubits//3-1)*5+run for run in range(5)]
    return id_list

def compute_distances_between_persistence_diagrams(num_qubits1, num_qubits2, p, set, h_dim, distance="bottleneck"):
    """
    Compute list of distances between persistencediagrams of loss landscapes of QAOA instances with two different qubit numbers,
    but fixed p and sample set (used to compute the loss landscape).
    
    :param num_qubits1: Description
    :param num_qubits2: Description
    :param p: Description
    :param set: Description
    :param distance: Description
    """
    assert h_dim in [0,1]
    assert num_qubits1 in [3,6,9,12,15,18]
    assert num_qubits2 in [3,6,9,12,15,18]
    assert p in [1,2,3]
    assert set in range(5)

    id_list1 = get_qaoa_ids(num_qubits1, p, set)
    id_list2 = get_qaoa_ids(num_qubits2, p, set)
    ANALYSIS_BASE_DIR.mkdir(exist_ok=True)

    distance_list = []
    
    if(num_qubits1 == num_qubits2):
        id_tuples = []
        for k, id1 in enumerate(id_list1):
            for id2 in id_list1[k+1:]:
                id_tuples.append(tuple((id1, id2)))
        if(distance == "bottleneck"):
            with ProcessPoolExecutor(max_workers=max_workers) as exe:
                futures = [exe.submit(compute_bottleneck_distance, id_1, id_2, h_dim) for (id_1,id_2) in id_tuples]            
                # await results & save them:
                for future in as_completed(futures):
                    dist = future.result()
                    distance_list.append(dist)
    else:
        if(distance == "bottleneck"):
            with ProcessPoolExecutor(max_workers=max_workers) as exe:
                futures = [exe.submit(compute_bottleneck_distance, id_1, id_2, h_dim) for (id_1,id_2) in list(itertools.product(id_list1, id_list2))]            
                # await results & save them:
                for future in as_completed(futures):
                    dist = future.result()
                    distance_list.append(dist)
    return distance_list

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


def main():
    set = 0
    qubit_counts = [3,6,9,12,15,18]
    h_dim = 1
    p=3
    distance_dict = {"set": 0, "h_dim": h_dim}
    ANALYSIS_BASE_DIR.mkdir(exist_ok=True)
    for k, q1 in enumerate(qubit_counts):
        for q2 in qubit_counts[k:]:
            distance_list = compute_distances_between_persistence_diagrams(q1, q2, p, set, h_dim, distance="bottleneck")
            print(f"[DONE] {q1} vs. {q2}")

            distance_dict[f"{q1},{q2}"] = {"p": p, "bottleneck distances": distance_list}
            distance_dict["mean"] = np.mean(distance_list)
            distance_dict["median"] = np.median(distance_list)
            distance_dict["min"] = np.min(distance_list)
            distance_dict["max"] = np.max(distance_list)
            distance_dict["std"] = np.std(distance_list)

            # save distances
                    
            text_file = ANALYSIS_BASE_DIR / f"bottleneck_dist_p_{p}_set_{set}_hdim_{h_dim}.txt"
            with open(text_file, "a") as f:
                f.write(f"{q1},{q2}:{distance_list}\n")
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[DONE] {now}: {q1} vs. {q2}, p={p}")
    

    json_file = ANALYSIS_BASE_DIR / f"bottleneck_dist_p_{p}_set_{set}_hdim_{h_dim}.json"
    json_file.write_text(json.dumps(distance_dict, indent=4))


if __name__=="__main__":

    main()