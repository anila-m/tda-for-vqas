import numpy as np
import scipy as sp
from scipy.stats import qmc
import math

"""
Several sampling strategies for n-dimensional hypercubes.
"""

def get_uniformly_random_samples(min: float, max: float, dim: int, number_of_samples: int = 10000) -> list[list[float]]: #TODO: Stimmt der return type?
    """
    Generates a list of uniformly random sampled points within a hypercube.
    The left corner of the hypercube is [min, ..., min] and upper right corner is [max, ..., max].
    
    :param min: defines lower left corner of hypercube [min, ..., min]
    :param max: defines upper right corner of hypercube [max, ..., max]
    :param dim: dimension of hypercube
    :param number_of_samples: number of sample points to be generated (default: 10000). 
    """
    lowerleft = np.ones(dim)*min
    upperright = np.ones(dim)*max
    rng = np.random.default_rng()
    sample_points = rng.uniform(low = lowerleft, high = upperright, size = (number_of_samples, dim))
    return sample_points


def get_latin_hypercube_samples(lowerleft, upperright, dim, number_of_samples=10000):
    """
    Uses Latin Hypercube Sampling to generate a list of sample points (with specified dimension) within a hypercube. 
    The left corner of the hypercube is [min, ..., min] and upper right corner is [max, ..., max].
    Latin Hypercube Sampling generates more evenly distributed sample points compared to uniformly_random_sampling
    by taking previously sampled points into account.
    More info: https://en.wikipedia.org/wiki/Latin_hypercube_sampling
    
    :param min: defines lower left corner of hypercube [min, ..., min]
    :param max: defines upper right corner of hypercube [max, ..., max]
    :param dim: dimension of hypercube
    :param number_of_samples: number of sample points to be generated (default: 10000). 
    """
    sampler = qmc.LatinHypercube(d=dim)
    sample_points = sampler.random(n=number_of_samples)
    #lowerleft = np.ones(dim)*min
    #upperright = np.ones(dim)*max
    sample_points = qmc.scale(sample_points,l_bounds=lowerleft, u_bounds=upperright)
    return sample_points

def get_2D_grid_samples(min1, min2, max1, max2, grid_size1=10, grid_size2=10):
    x = np.linspace(min1, max1, grid_size1)
    y = np.linspace(min2, max2, grid_size2)
    g = np.meshgrid(x,y)
    sample_points = np.array(g).T.reshape(-1,2)
    return sample_points


def get_grid_samples(min, max, dim, number_of_samples=10000):
    """
    Generates n-dimensional grid sample points. The number of sample points N per dimension is chosen s.t. N^dim <= number_of_samples
    
    :param min: defines lower left corner of hypercube [min, ..., min]
    :param max: defines upper right corner of hypercube [max, ..., max]
    :param dim: dimension of hypercube
    :param number_of_samples: number of sample points to be generated (default: 10000). 
    """
    N = int(np.floor(number_of_samples**(1.0/dim)))
    print(number_of_samples, N**dim)
    # sample points per dimension
    values_per_dimension = []
    for i in range(dim):
        values = np.linspace(min[i], max[i], N)
        values_per_dimension.append(values)
    values_per_dimension = np.asarray(values_per_dimension)
    # list of all sample points
    print(values_per_dimension)
    landscape_shape = []
    for _ in range(dim):
        landscape_shape.append(N)
    landscape_shape = tuple(landscape_shape)
    landscape = np.zeros(landscape_shape)
    sample_points = []
    for idx, _ in np.ndenumerate(landscape):
        print(idx)
        sample = values_per_dimension[tuple(range(dim)),idx]
        sample_points.append(sample)
    return np.asarray(sample_points) 

def get_grid_landscapes_from_stepsize(min, grid_size, loss_func, step_size):
    '''
    Generates dim-dimensional grid sample points. The number of sample points N per dimension is dependant on the specified step size.
    Returns a dim-dimenionsal numpy array (loss landscape).
    
    :param min: defines lower left corner of hypercube [min, ..., min]
    :param gridsize: defines number of points in each dimension, i.e. numpy array of dimension dim
    :param dim: dimension of hypercube
    :param loss_func: loss function. Takes numpy array of dimension dim as input and ouputs a loss value
    :param stepsize: stepsize in each dimension, i.e. numpy array of dimension dim.
    
    '''
    assert min.shape == grid_size.shape
    assert min.shape == step_size.shape
    dim = min.shape[0]
    # calculate the parameter values for the grid size, evenly spread from 0 to 2 pi
    coordinates = []
    for dir in range(dim):
        low = min[dir]
        high = low + (grid_size[dir])*step_size[dir]
        coord = np.arange(low, high, step_size[dir])
        coordinates.append(coord)
    #step_size = lanscape_limit / (grid_size-1) # <- more evenly spread samples
    # generate landscape
    landscape_shape = []
    # 5, 9 [9][9][9][9][9]
    for dir in range(dim):
        landscape_shape.append(grid_size[dir])
    landscape_shape = tuple(landscape_shape)
    landscape = np.empty(tuple(grid_size.astype(np.int64)))
    # for every point
    for idx, _ in np.ndenumerate(landscape):  
        sample_point =  [] 
        # generate param array
        #print("idx", idx)
        i = 0
        for dimension in idx: # idx = [a,b,c,d,e,f] wobei alle zwischen 0 und 15 sind --> index eines Gitterpunkts, welcher Parameter für qnn is
            sample_point.append(coordinates[i][dimension]) 
            i += 1
        # calculate loss
        sample_point = np.asarray(sample_point) # Gitterpunkt x für objective(x)
        loss = loss_func(sample_point) 
        landscape[idx]=loss
    return coordinates, landscape



if __name__=="__main__":
    def f(x):
        return np.sum(x)
    l = get_grid_landscapes_from_stepsize(np.asarray([-3,-3,-1.5, -1.5]), np.asarray([3,3,2,2]), f, 1)
    print(l)

