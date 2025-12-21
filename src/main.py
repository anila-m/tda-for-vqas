import os 
from utils.sampling_utils import get_grid_samples

if __name__ == '__main__':
    get_grid_samples([0,0,0], [3,3,3], dim=3, number_of_samples=27)
