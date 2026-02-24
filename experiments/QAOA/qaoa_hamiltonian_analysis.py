from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import itertools
import json
from pathlib import Path
from persim import bottleneck, bottleneck_matching
from gtda.diagrams import PairwiseDistance
import os
import matplotlib.pyplot as plt
from ripser import ripser
from persim import plot_diagrams
import seaborn as sns
from orquestra.quantum.operators import get_pauli_strings, convert_dict_to_op
import copy

import numpy as np

from src.qaoa.utils import generate_timestamp_str
from src.utils.file_utils import ripser_list_to_giotto
from qaoa_prepare_landscapes import generate_qaoa_landscape
from bp_ng_experiment_main import plot_loss_landscape_interpolated

max_workers = 4
cpu_count = os.cpu_count()
timestamp = generate_timestamp_str()

# Directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
RIPSER_BASE_DIR = BASE_DIR / "experiment_results" / "QAOA" / "ripser_results"
ANALYSIS_BASE_DIR = BASE_DIR / "experiment_results" / "hamiltonian_experiment" / "analysis" / timestamp

RESULTS_BASE_DIR = BASE_DIR / "experiment_results" / "hamiltonian_experiment" / timestamp
HAMILTONIAN_RIPSER_BASE_DIR = BASE_DIR / "experiment_results" / "hamiltonian_experiment" / "ripser_results"


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


def print_all_hamiltonians():
    for num_qubits in [3,6,9,12,15,18]:
        print(f"Num qubits = {num_qubits}")
        for run in range(5):
            # load hamiltonian
            id = (num_qubits//3-1)*5+run
            file_1 = RIPSER_BASE_DIR / f"persistence_qaoa_{id}_not_transformed_H1.json"
            r_dict = json.load(open(file_1))
            ham = r_dict["hamiltonian"]
            ham_string = convert_dict_to_op(ham)
            print(ham_string)
        print("---------------------------")

def flatten_hamiltonian(ham_dict):
    """
    Converts the nested hamiltonian dict into a flat dict: {'Z0Z1': -0.5, ...}. Values are coefficients.

    param: ham_dict
    """
    flat = {}
    for term in ham_dict["terms"]:
        ops = term["pauli_ops"]
        if not ops:
            key = "Identity"
        else:
            sorted_ops = sorted(ops, key=lambda x: x["qubit"])
            key = "".join([f"{op['op']}{op['qubit']}" for op in sorted_ops])
        
        coeff = complex(term["coefficient"]["real"], term["coefficient"]["imag"])
        
        # If the term already exists: Info message
        if(key in flat.keys()):
            print(f"coefficient for {key} already exists: {flat.get(key, 0)}")
        flat[key] = flat.get(key, 0) + coeff
        
        
    return flat

def compare_hamiltonians(h1_raw, h2_raw):
    h1 = flatten_hamiltonian(h1_raw)
    h2 = flatten_hamiltonian(h2_raw)
    
    # unique terms present in both hamiltonians
    all_keys = set(h1.keys()).union(set(h2.keys()))
    
    diffs = []
    no_diffs = 0
    for key in all_keys:
        val1 = h1.get(key, 0j)
        val2 = h2.get(key, 0j)
        
        if val1 != val2:
            diffs.append({
                "term": key,
                "h1_val": val1,
                "h2_val": val2,
                "delta": val1 - val2
            })
            no_diffs += 1
            
    return diffs, no_diffs

def get_all_hamiltonians(num_qubits):
    all_hams = {}
    for run in range(5):
        # load hamiltonian
        id = (num_qubits//3-1)*5+run
        file_1 = RIPSER_BASE_DIR / f"persistence_qaoa_{id}_not_transformed_H1.json"
        r_dict = json.load(open(file_1))
        ham = r_dict["hamiltonian"]
        all_hams[run] = ham
    return all_hams

def compare_all_hamiltonians():
    qubit_list = [3,6,9,12,15,18]
    #qubit_list = [3,6]
    for num_qubits in qubit_list:
        print(f"Num qubits = {num_qubits}")
        all_hams = get_all_hamiltonians(num_qubits=num_qubits)
        diff_matrix = np.zeros((5,5))
        for run1 in range(5):
            for run2 in range(run1+1, 5):
                diffs, no_diffs = compare_hamiltonians(all_hams[run1], all_hams[run2])
                diff_matrix[run1, run2] = no_diffs
                #print(run1, run2, diffs)
                #print("----------------------")
        print(diff_matrix)
        print("-----------------------------------------------------------------")

def find_and_flip_max_coeff(all_hams, flip=True):
    # if flip: flip sign of coefficient, else: set coefficient to zero
    if flip: factor = -1
    else: factor = 0

    max_val = -np.inf
    run = None
    target_term_idx = None
    
    # find hamiltonian with largest absolute coefficient
    for ham_key, ham_data in all_hams.items():
        for idx, term in enumerate(ham_data["terms"]):
            # Calculate absolute value
            r = term["coefficient"]["real"]
            i = term["coefficient"]["imag"]
            abs_coeff = (r**2 + i**2)**0.5
            
            if abs_coeff > max_val:
                max_val = abs_coeff
                run = ham_key
                target_term_idx = idx

    if run is None:
        return None, None

    # deep copy of hamiltonian
    updated_ham = copy.deepcopy(all_hams[run])
    
    # change largest absolute coefficient
    target_term = updated_ham["terms"][target_term_idx]
    target_term["coefficient"]["real"] *= factor
    target_term["coefficient"]["imag"] *= factor
    
    return run, updated_ham, target_term_idx

def compute_persistence_diagram_flipped_coefficient(flip=True):
    dim = 1
    for num_qubits in [15,18 ]:
        all_hams = get_all_hamiltonians(num_qubits=num_qubits)

        # get new hamiltonian
        run, flipped_ham, target_term_idx = find_and_flip_max_coeff(all_hams=all_hams, flip=flip)
        result_dict = {"num_qubits": num_qubits, "run": run, "target_term_idx": target_term_idx, "hamiltonian flipped": flip, "new hamiltonian": flipped_ham}

        # compute loss landscape
        ham = convert_dict_to_op(flipped_ham)
        ham_file_label = ""
        ham_config_dict = {ham_file_label: {}}
        ham_config_dict[ham_file_label]["num_qubits"] = num_qubits
        ham_config_dict[ham_file_label]["run"] = run
        ham_config_dict[ham_file_label]["hamiltonian"] = ham
        p = 1
        set = 0
        start = datetime.now()
        id, _, samples_file_name, elapsed_time, landscape, _ = generate_qaoa_landscape(ham_config_dict, ham_file_label, p, set)
        elapsed_time = datetime.now()-start
        result_dict["runtime landscape"] = str(elapsed_time).split(".",1)[0]

        # compute persistence diagram
        start = datetime.now()
        ripser_result = ripser(landscape, maxdim=dim)
        elapsed_time = datetime.now()-start
        result_dict["runtime persistence"] = str(elapsed_time).split(".",1)[0]

        # prep and save ripser result 
        


        # update results dict
        # loss landscape + meta data
        result_dict["config id"] = id
        result_dict["p"] = p
        result_dict["sample_set"] = set
        result_dict["sample_file_label"] = samples_file_name
        result_dict["runtime"] = str(elapsed_time).split(".",1)[0]
        min = np.min(landscape[:, -1])
        max = np.max(landscape[:, -1])
        result_dict.update({
            "min cost": min,
            "max cost": max,
            "landscape": landscape.tolist()
        })
        # persistence diagram
        d = [v.tolist() for v in ripser_result["dgms"]]
        c = [[v.tolist() for v in l] for l in ripser_result["cocycles"]]
        ripser_dict = {
            "dgms" : d,
            "cocycles": c,
            "num_edges": ripser_result["num_edges"],
            #"distance_matrix": ripser_result["dperm2all"].tolist(), #too large to save
            "r_cover": ripser_result["r_cover"]
         }
        if ripser_result["idx_perm"] is not None:
            ripser_dict["idx_perm"] = ripser_result["idx_perm"].tolist()

        result_dict["persistence diagram"] = ripser_dict

        # save ripser result, including landscape, etc.
        file_name = f"persistence_qaoa_{id}_flipped_{flip}_not_transformed_H{dim}.json"
        file_name_plot = f"persistence_diagram_qaoa_{id}_flipped_{flip}_not_transformed_H{dim}.png"
        ripser_path = RESULTS_BASE_DIR / "ripser_results" 
        RESULTS_BASE_DIR.mkdir(exist_ok=True)
        ripser_path.mkdir(exist_ok=True)
        plot_path = RESULTS_BASE_DIR / "persistence_diagrams"
        plot_path.mkdir(exist_ok=True)
        ripser_file = ripser_path / file_name
        ripser_file.write_text(json.dumps(result_dict, indent=4))

        # generate and save persistence diagram
        plot_diagrams(ripser_result["dgms"], show=False)
        plt.savefig(plot_path / file_name_plot)
        plt.close()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[DONE] {num_qubits} qubits at {now}: {file_name}")

def compute_metrics_between_persistence_diagrams(flip=True):
    """

    :param metric: string, "bottleneck" or "wasserstein". Metric between Persistence diagrams.
    :param flip: boolean, if true: sign of coefficient in hamiltonian was flipped (5 -> -5), if false: coefficient in hamiltonian was set to zero. default: true
    """
    ANALYSIS_BASE_DIR.mkdir(exist_ok=True)
    ids = [1,5,10,15,20,25]
    i=0
    result_dict = {"info": "distances  between persistence diagrams of qaoa instances with one flipped coefficient in the hamiltonian. hamiltonian flipped=true means the sign of the coefficient was flipped. ...flip=false means the coefficient was set to zero. Distances were approximated using giotto tda. Persistence diagrams were computed using ripser (scikit tda). Keys are number of qubits. Distances are the average of two values (persistence diagram 1 -> 2 and 2 -> 1)."}
    for num_qubits in [3,6,9,12,15,18]:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[INFO] {now}: Start {num_qubits} qubits...")
        # get persistence diagram of original hamiltonian
        file_name1 = f"persistence_qaoa_{ids[i]}_not_transformed_H1.json"
        file1 = RIPSER_BASE_DIR / file_name1
        r_dict1 = json.load(open(file1))
        d = r_dict1["persistence diagram"]["dgms"]
        d1 = [np.asarray(v) for v in d]

        # get persistence diagram of flipped hamiltonian
        file_name2 = f"persistence_qaoa_{ids[i]}_flipped_{flip}_not_transformed_H1.json"
        file2 = HAMILTONIAN_RIPSER_BASE_DIR / file_name2
        r_dict2 = json.load(open(file2))
        d = r_dict2["persistence diagram"]["dgms"]
        d2 = [np.asarray(v) for v in d]

        diagrams = ripser_list_to_giotto([d1, d2])

        result_dict[num_qubits] = {
            "num_qubits": num_qubits,
            "run": r_dict2["run"],
            "target_term_idx": r_dict2["target_term_idx"],
            "hamiltonian flipped": r_dict2["hamiltonian flipped"],
            "p": r_dict2["p"],
            "config id": r_dict2["config id"],
        }

        for metric in ["bottleneck", "wasserstein"]:
            #compute distance
            PD = PairwiseDistance(metric = metric, n_jobs=cpu_count-1, order=None)
            distance = PD.fit_transform(diagrams)
            dist_H0 = distance[:,:,0]
            dist_H1 = distance[:,:,1]
            result_dict[num_qubits][metric] = {"H0": (dist_H0[0,1]+dist_H0[1,0])/2, "H1": (dist_H1[0,1]+dist_H1[1,0])/2}

        # plot both loss landscapes
        loss = np.asarray(r_dict1["landscape"])
        loss_flipped = np.asarray(r_dict2["landscape"])
        plot_loss_landscape_interpolated(loss, ANALYSIS_BASE_DIR, f"loss_landscape_original_qaoa_{ids[i]}_interpolated")
        plot_loss_landscape_interpolated(loss_flipped, ANALYSIS_BASE_DIR, f"loss_landscape_flipped_qaoa_{ids[i]}_interpolated")

        i += 1
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[DONE] {now}: {num_qubits} qubits")
        
    # save results
    results_file_name =  f"flipped_hamiltonian_metric_anaylsis.json"
    
    results_file = ANALYSIS_BASE_DIR / results_file_name
    results_file.write_text(json.dumps(result_dict, indent=4))


if __name__=="__main__":
    compute_metrics_between_persistence_diagrams()