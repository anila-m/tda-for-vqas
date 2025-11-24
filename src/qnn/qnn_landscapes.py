import torch
import numpy as np
from qnn.data import *

def generate_data_points(type_of_data, schmidt_rank, num_data_points, U, num_qubits):
    """generates data points given a configuration consisting of the type of data point,
    the schmidt rank (level of entanglement)
    and the number of data points, as well as a unitary for reshaping purposes

    Args:
        type_of_data (int): describes what type of data to use (1=random, 2=orthogonal, 3=linearly dependent in H_x, 4= variable schmidt rank)
        schmidt_rank (int): what level of entanglement should the data have
        num_data_points (int): how many data points you want
        U (unitary): a unitary for reshaping of the data points
        num_qubits (int): the amount of wires/x_qubits for the chosen ansatz

    Returns:
        tensor: a tensor of data points that can be used for the experiments
    """

    raw_input = 0
    x_qubits = num_qubits
    r_qubits = x_qubits
    if type_of_data == 1:
        raw_input = torch.from_numpy(
            np.array(
                uniform_random_data(schmidt_rank, num_data_points, x_qubits, r_qubits)
            )
        )
    elif type_of_data == 2:
        raw_input = torch.from_numpy(
            np.array(
                uniformly_sample_orthogonal_points(
                    schmidt_rank, num_data_points, x_qubits, r_qubits
                )
            )
        )
    elif type_of_data == 3:
        raw_input = torch.from_numpy(
            np.array(
                sample_non_lihx_points(
                    schmidt_rank, num_data_points, x_qubits, r_qubits
                )
            )
        )
    elif type_of_data == 4:
        raw_input = torch.from_numpy(
            np.array(
                uniform_random_data_average_evenly(
                    schmidt_rank, num_data_points, x_qubits, r_qubits
                )
            )
        )
    return raw_input.reshape(
        (raw_input.shape[0], int(raw_input.shape[1] / U.shape[0]), U.shape[0])
    ).permute(0, 2, 1)