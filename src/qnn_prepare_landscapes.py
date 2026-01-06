import json
import numpy as np
import os
from scipy.stats import qmc

from utils.sampling_utils import get_latin_hypercube_samples

def generate_LHS_sample_points_for_qaoa(number_of_samples=10000, min=0, max=2*np.pi):
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

def generate_qnn_landscapes():
    """
    Configurations based on Ülger (2024).
    fixed: number of traning samples = 4, data type = random
    variable: Schmidt rank (1 to 4), target unitaries (5 different ones), data batches (i.e. training sample sets) (5 different ones)
    --> correspond to configurations 0 to 19 (each configurations contains 5 different data batches)
    """