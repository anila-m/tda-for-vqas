from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import json
import re
import numpy as np
import os
from scipy.stats import qmc
import torch
from ripser import ripser

from src.utils.sampling_utils import get_latin_hypercube_samples
from src.utils.file_utils import save_qnn_landscape
from src.qaoa.utils import generate_timestamp_str
from src.qnn.qnn_untils import generate_qnn_landscape

def generate_LHS_sample_points_for_qnn(number_of_samples=10000, min=0, max=2*np.pi):
    """
    Docstring for generate_LHS_sample_points_for_qaoa
    
    :param number_of_samples: Description
    :param min: Description
    :param max: Description
    """
    # save path
    save_dir = os.path.join(os.path.join(os.path.join(os.getcwd(), "resources"), "sample_points"), "QNN")
    n=6
    #determine correct lower left and upper right corner of parameter space
    lowerleft = np.ones(n)*min
    upperright = np.ones(n)*max

    # generate 20 different sets of sample points per dimension (i.e. per p)
    samples = []
    discrepancies = []
    for run in range(20):
        # generate LHS samples
        sample_points = get_latin_hypercube_samples(lowerleft=lowerleft, upperright=upperright, dim=n, number_of_samples=number_of_samples)
        sampler = qmc.LatinHypercube(d=n)
        sample_points = sampler.random(n=number_of_samples)
        discrepancy = qmc.discrepancy(sample_points)
        sample_points = qmc.scale(sample_points,l_bounds=lowerleft, u_bounds=upperright)
        
        samples.append(sample_points)
        discrepancies.append(discrepancy)

    # determine sample points with lowest discrepancy and save them
    # sort sample point sets according to discrepancy
    sorted_samples = [s for _, s in sorted(zip(discrepancies, samples))]
    sorted_discrepancies = [d for d, _ in sorted(zip(discrepancies, samples))]
    # save best 5 sample point sets as json files (best according to discrepancy, lower discrepancy = better)
    for i in range(5):
        s = sorted_samples[i]
        d = sorted_discrepancies[i]
        file_name = f"{n}D_samples_LHS_{number_of_samples}_min_0_max_2pi_set_{i}_discrepancy_{d}.json" 
        with open(os.path.join(save_dir, file_name), "w") as f:
            json.dump(s.tolist(), f)

def generate_landscapes(qnn_id, unitary, unitary_id, databatch, databatch_id, set, transformed=False, transformed_min=50, transformed_max=100):
    """
    Configurations based on Ülger (2024).
    fixed: number of traning samples = 4, data type = random
    variable: Schmidt rank (1 to 4), target unitaries (5 different ones), data batches (i.e. training sample sets) (5 different ones)
    --> correspond to configurations 60 to 79 (each configuration contains 5 different data batches)
    """
    start = datetime.now()

    # load correct sample points
    s = f"6D_samples_LHS_10000_min_0_max_2pi_set_{set}" 
    s_dir = os.path.join(os.getcwd(), f"resources/sample_points/QNN")
    all_sample_sets = os.listdir(s_dir)
    for file in all_sample_sets:
        if file.startswith(s):
            file_path = os.path.join(s_dir, file)
            with open(file_path) as f:
                sample_points =  np.asarray(json.load(f))
    
    # generate loss landscape
    landscape = generate_qnn_landscape(unitary=unitary, datapoints=databatch, sample_points=sample_points, transformed=transformed, transformed_min=transformed_min, transformed_max=transformed_max)
    #ripser_result = ripser(landscape, maxdim=1)
    #print(len(landscape))
    elapsed_time = datetime.now()-start
    #print(f"Loss Landscape:", elapsed_time.strftime("%H:%M:%S"))
    return s, elapsed_time, landscape


def main(transformed = False, transformed_min = 50, transformed_max = 100):
    timestamp = generate_timestamp_str()
    # load correct qnn configurations
    filename = "resources/QNN/configurations_16_6_4_10_13_3_14.txt"
    file = open(filename, 'r')
    Lines = file.readlines()
    n = 0
    conf_id = 0
    databatch_id = 0
    data_type = ""
    num_data_points = 0
    s_rank = 0
    unitary = []
    databatches = []
    
    config_dict = {}
    cpu_count = os.cpu_count() 
    print(cpu_count)
    assert cpu_count is not None

    for line in Lines:
        if conf_id > 79: # other configs not needed
            break
        if(line.strip() == "---"): # config has been fully read, generate loss landscape for each databatch (training samples) and sample point set
            if(conf_id >= 60):
                for batch in range(len(databatches)):
                    for set in range(5):
                        unitary_id = conf_id % 5
                        qnn_id = (s_rank -1) * 125 + unitary_id*25 + batch*5 + set
                        databatch = databatches[batch]
                        databatch_string = (
                            np.array2string(databatch, separator=",")
                            .replace("\n", "")
                            .replace(" ", "")
                        )
                        unitary_string = (
                            np.array2string(unitary, separator=",")
                            .replace("\n", "")
                            .replace(" ", "")
                        )
                        #print(conf_id, s_rank, unitary_id, qnn_id)
                        samples_file_name, elapsed_time, landscape = generate_landscapes(qnn_id, unitary, unitary_id, databatch, batch, set, transformed=transformed, transformed_min=transformed_min, transformed_max=transformed_max)
                        #print(f"[DONE] {qnn_id}: {s_rank}, {unitary_id}, {batch}, {set}")
                        config_dict["original conf_id"] = conf_id
                        config_dict["qnn_id"] = qnn_id
                        config_dict["unitary_id"] = unitary_id
                        config_dict["unitary"] = unitary_string
                        config_dict["databatch id"] = batch
                        config_dict["databatch"] = databatch_string
                        config_dict["schmidt rank"] = s_rank
                        config_dict["data type"] = data_type
                        config_dict["number of data points"] = num_data_points
                        config_dict["sampling method"] = "LHS"
                        config_dict["transformed loss"] = transformed
                        if transformed:
                            config_dict["transformed min"] = transformed_min
                            config_dict["transformed max"] = transformed_max
                        config_dict["sample_set"] = set
                        config_dict["sample_file_label"] = samples_file_name
                        config_dict["runtime"] = str(elapsed_time).split(".",1)[0]

                        # save landscape
                        if(transformed):
                            landscape_directory = os.path.join(os.path.join(os.path.join(os.path.join(os.getcwd(), "resources"), "QNN"), "landscapes"), f"transformed_{transformed_min}_{transformed_max}")
                            landscape_file_name = f"landscape_qnn_{qnn_id}_srank_{s_rank}_unitary_{unitary_id}_batch_{batch}_set_{set}_transformed_{transformed_min}_{transformed_max}_{timestamp}.json"
                        else:
                            landscape_directory = os.path.join(os.path.join(os.path.join(os.path.join(os.getcwd(), "resources"), "QNN"), "landscapes"), "not_transformed")
                            landscape_file_name = f"landscape_qnn_{qnn_id}_srank_{s_rank}_unitary_{unitary_id}_batch_{batch}_set_{set}_{timestamp}.json"
                        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        print(f"[DONE] {now}: {landscape_file_name}")
                        save_qnn_landscape(landscape, meta_data = config_dict, directory=landscape_directory, file_name=landscape_file_name)
                    config_dict = {}
            databatches = []
            

        else:
            var, val = line.split("=")
            if(var == "conf_id"): conf_id = int(val) #config ID: between 0 and 319
            elif(var == "data_type"): data_type = val # data type: random, orthogonal, non_lin_ind, var_s_rank
            elif(var == "num_data_points"): num_data_points = int(val)  # number of data points: 1, 2, 3, 4
            elif(var == "s_rank"): s_rank = int(val) # Schmidt-Rank: 1, 2, 3, 4
            elif(var == "unitary"): 
                val,_ = re.subn('\[|\]|\\n', '', val) 
                unitary = np.fromstring(val,dtype=complex,sep=',').reshape(-1,4) # unitary: 4x4 tensor
            elif(var.startswith("data_batch_")): 
                val,_ = re.subn('\[|\]|\\n', '', val)
                databatches.append(np.fromstring(val,dtype=complex,sep=',').reshape(-1,4,4)) # training samples: 1x4x4 tensor
    


if __name__=="__main__":
    #main(transformed=True)
    print("successful")