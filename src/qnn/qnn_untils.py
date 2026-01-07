from utils.data_utils import get_interval_transformation
import numpy as np
from qnn.qnn_cost_func import CostFunction

def generate_qnn_landscape(unitary, datapoints, sample_points, transformed=False, transformed_min=0, transformed_max=1):
    num_qubits=2
    dim = 3*num_qubits
    x = np.asarray(datapoints,dtype=complex)
    loss_func = CostFunction(num_qubits=num_qubits, unitary=unitary, inputs=x)
    landscape = []

    for _, point in enumerate(sample_points):
        loss = loss_func(point)
        if transformed:
            f = get_interval_transformation(a=transformed_min, b=transformed_max)
            loss = f(loss)
        data_point = np.append(point, loss)
        landscape.append(data_point)
    return np.asarray(landscape, dtype=np.float32)

