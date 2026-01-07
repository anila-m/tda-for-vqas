import json
import os
from ripser import ripser
from persim import plot_diagrams
from matplotlib import pyplot as plt
import numpy as np
from datetime import datetime

from utils.file_utils import save_persistence_diagrams
from utils.data_utils import transform_loss_landscape

dim = 1 # homology dimension

def main_experiment(transformed_min = 50, transformed_max = 100):
  landscape_directory = os.path.join(os.path.join(os.path.join(os.path.join(os.getcwd(), "resources"), "QNN"), "landscapes"), "not_transformed")
  all_landscapes = os.listdir(landscape_directory)
  for file in all_landscapes:
    file_path = os.path.join(landscape_directory, file)
    with open(file_path) as f:
      qnn_dict =  json.load(f)
      id = qnn_dict["qnn_id"]
      landscape = np.asarray(qnn_dict["landscape"])
      ######################################
      # compute persistence diagram for not-transformed landscape
      start = datetime.now()
      ripser_result = ripser(landscape, maxdim=dim)
      elapsed_time = datetime.now()-start
      qnn_dict["runtime"] = str(elapsed_time).split(".",1)[0]

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

      qnn_dict["persistence diagram"] = ripser_dict
      
      save_path = os.path.join(os.path.join(os.path.join(os.getcwd(), "experiment_results"), "QNN"), "not_transformed")
      file_name = f"persistence_qnn_{id}_not_transformed_H{dim}"
      file_name_plot = f"persistence_diagram_qnn_{id}_not_transformed_H{dim}"

      # save ripser result, including landscape, etc.
      with open(os.path.join(os.path.join(save_path, "ripser_results"), file_name), "w") as f:
        json.dump(qnn_dict, f, indent=4)

      # generate and save persistence diagram
      plot_diagrams(ripser_result["dgms"], show=False)
      plt.savefig(os.path.join(os.path.join(save_path, "persistence_diagrams"), file_name_plot))
      plt.close()
      now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
      print(f"[DONE] {now}: {file_name}")
      #########################################################
      # compute persistence diagram for transformed landscape
      transformed_landscape = transform_loss_landscape(landscape, transformed_min, transformed_max)
      start = datetime.now()
      ripser_result = ripser(transformed_landscape, maxdim=dim)
      elapsed_time = datetime.now()-start
      qnn_dict["runtime"] = str(elapsed_time).split(".",1)[0]

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

      qnn_dict["persistence diagram"] = ripser_dict
      qnn_dict["landscape"] = transformed_landscape.tolist()
      qnn_dict["transformed"] = True
      qnn_dict["transformed_min"] = transformed_min
      qnn_dict["transformed_max"] = transformed_max

      save_path = os.path.join(os.path.join(os.path.join(os.getcwd(), "experiment_results"), "QNN"), f"transformed_{transformed_min}_{transformed_max}")
      file_name = f"persistence_qnn_{id}_transformed_{transformed_min}_{transformed_max}_H{dim}"
      file_name_plot = f"persistence_diagram_qnn_{id}_transformed_{transformed_min}_{transformed_max}_H{dim}"
      
      # save ripser result, including landscape, etc.
      with open(os.path.join(os.path.join(save_path, "ripser_results"), file_name), "w") as f:
        json.dump(qnn_dict, f, indent=4)

      # generate and save persistence diagram
      plot_diagrams(ripser_result["dgms"], show=False)
      plt.savefig(os.path.join(os.path.join(save_path, "persistence_diagrams"), file_name_plot))
      plt.close()
      now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
      print(f"[DONE] {now}: {file_name}")
      

if __name__=="__main__":
  # with open("experiment_results/QNN/not_transformed/ripser_results/persistence_qnn_320_not_transformed_H1", "r") as f:
  #   r_dict = json.load(f)
  #   dgms = r_dict["persistence diagram"]["dgms"]
  #   d = [np.asarray(v) for v in dgms]
  #   plot_diagrams(d, show=True)

  main_experiment()
  #main_experiment(transformed=True)
