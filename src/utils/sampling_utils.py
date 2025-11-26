import numpy as np
import scipy as sp
from scipy.stats import qmc


"""
Several sampling strategies for n-dimensional hypercubes.
"""

def get_uniformly_random_samples(min: float, max: float, dim: int, number_of_samples: int = 1000) -> list[list[float]]: #TODO: Stimmt der return type?
    """
    Generates a list of uniformly random sampled points within a hypercube.
    The left corner of the hypercube is [min, ..., min] and upper right corner is [max, ..., max].
    
    :param min: defines lower left corner of hypercube [min, ..., min]
    :param max: defines upper right corner of hypercube [max, ..., max]
    :param dim: dimension of hypercube
    :param number_of_samples: number of sample points to be generated (default: 1000). 
    """
    lowerleft = np.ones(dim)*min
    upperright = np.ones(dim)*max
    rng = np.random.default_rng()
    sample_points = rng.uniform(low = lowerleft, high = upperright, size = (number_of_samples, dim))
    return sample_points

#TODO: implement latin hypercube sampling
def get_latin_hypercube_samples(min, max, dim, number_of_samples=1000):
    """
    Uses Latin Hypercube Sampling to generate a list of sample points (with specified dimension) within a hypercube. 
    The left corner of the hypercube is [min, ..., min] and upper right corner is [max, ..., max].
    Latin Hypercube Sampling generates more evenly distributed sample points compared to uniformly_random_sampling
    by taking previously sampled points into account.
    More info: https://en.wikipedia.org/wiki/Latin_hypercube_sampling
    
    :param min: defines lower left corner of hypercube [min, ..., min]
    :param max: defines upper right corner of hypercube [max, ..., max]
    :param dim: dimension of hypercube
    :param number_of_samples: number of sample points to be generated (default: 1000). 
    """
    sampler = qmc.LatinHypercube(d=dim)
    sample_points = sampler.random(n=number_of_samples)
    lowerleft = np.ones(dim)*min
    upperright = np.ones(dim)*max
    sample_points = qmc.scale(sample_points,l_bounds=lowerleft, u_bounds=upperright)
    return sample_points