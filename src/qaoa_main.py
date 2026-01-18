import json
from datetime import datetime
from pathlib import Path

import networkx as nx
import numpy as np
from orquestra.integrations.qulacs.simulator import QulacsSimulator

from ripser import ripser
from persim import plot_diagrams
from matplotlib import pyplot as plt

backend = QulacsSimulator()
scan_resolution = 101 # 31^2 approx 1000, 101 is too much for persistence
num_runs=5
N=10000 #number of sample points
total_number_landscapes = 450
dim = 1 # homology dimension

# Directories
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_BASE_DIR = BASE_DIR / "experiment_results" / "QAOA"
LANDSCAPE_DIR = BASE_DIR / "resources" / "QAOA" / "landscapes"

def main_experiment():
    RESULTS_BASE_DIR.mkdir(exist_ok=True)
    all_landscapes = LANDSCAPE_DIR.iterdir()
    i=0
    for file in all_landscapes:
        with open(file) as f:
            qaoa_dict =  json.load(f)
            id = qaoa_dict["config id"]
            landscape = np.asarray(qaoa_dict["landscape"])
            
            ######################################
            # compute persistence diagram for landscape
            start = datetime.now()
            ripser_result = ripser(landscape, maxdim=dim)
            elapsed_time = datetime.now()-start
            qaoa_dict["runtime"] = str(elapsed_time).split(".",1)[0]

            # prep and save ripser result 
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

            qaoa_dict["persistence diagram"] = ripser_dict

            # save ripser result, including landscape, etc.
            file_name = f"persistence_qaoa_{id}_not_transformed_H{dim}.json"
            file_name_plot = f"persistence_diagram_qaoa_{id}_not_transformed_H{dim}.png"
            ripser_path = RESULTS_BASE_DIR / "ripser_results" 
            ripser_path.mkdir(exist_ok=True)
            plot_path = RESULTS_BASE_DIR / "persistence_diagrams"
            plot_path.mkdir(exist_ok=True)
            ripser_file = ripser_path / file_name
            ripser_file.write_text(json.dumps(qaoa_dict, indent=4))

            # generate and save persistence diagram
            plot_diagrams(ripser_result["dgms"], show=False)
            plt.savefig(plot_path / file_name_plot)
            plt.close()
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            i += 1
            print(f"[DONE] {i}/{total_number_landscapes} at {now}: {file_name}")


if __name__ == "__main__":
    main_experiment()
