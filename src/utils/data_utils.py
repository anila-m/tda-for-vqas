import numpy as np
import torch
import networkx as nx
import math
from math import *

# Utility functions for QNN simulations

# tolerance for numpy =0 compares for orthonormality checks
tolerance = 1E-10

# Encodings

def get_interval_transformation(a, b, min=0, max=1):
    """
        Returns a function that transforms values x in [min,max] to values f(x) in [a,b]
    """
    def f(x):
        return (x-min)*(b-a)/(max-min)+a
    return f

def int_to_bin(num, num_bits):
    """
    Convert integer to binary with padding
    (e.g. (num=7, num_bits = 5) -> 00111)
    """
    b = bin(num)[2:]
    return [0 for _ in range(num_bits - len(b))] + [int(el) for el in b]

def one_hot_encoding(num, num_bits):
    """
    Returns one-hot encoding of a number
    (e.g. (num=4, num_bits=7) -> 0000100)
    """
    result = [0]*num_bits
    result[num] = 1
    return result

def normalize(point):
    """Normalizes vector"""
    return point / np.linalg.norm(point)

def num_li_hx(vectors, dim_x, dim_r):
    """number of linear independent vectors in hx"""
    hx_vectors_total = []
    for vec in vectors:
        coeffs, lefts, rights = schmidt_decomp(vec, dim_r, dim_x)
        hx_vectors = np.array(rights)[np.array(coeffs) > 1E-10] # tolerance based
        for vec in hx_vectors:
            hx_vectors_total.append(vec)
    return num_lin_ind(hx_vectors_total)

def tensor_product(state1: np.ndarray, state2: np.ndarray):
    result = np.zeros(len(state1)*len(state2), dtype=np.complex128)
    for i in range(len(state1)):
        result[i*len(state2):i*len(state2)+len(state2)] = state1[i] * state2
    return result

def torch_tensor_product(matrix1: torch.Tensor, matrix2: torch.Tensor, device='cpu'):
    result = torch.zeros((matrix1.shape[0]*matrix2.shape[0], matrix1.shape[1]*matrix2.shape[1]), dtype=torch.complex128, device=device)
    for i in range(matrix1.shape[0]):
        for j in range(matrix1.shape[1]):
            result[i*matrix2.shape[0]:i*matrix2.shape[0]+matrix2.shape[0], j*matrix2.shape[1]:j*matrix2.shape[1]+matrix2.shape[1]] = matrix1[i, j] * matrix2
    return result

# Data integrity checks

def all_ortho(*vecs):
    """All points orthogonal?"""
    for a in vecs:
        for b in vecs:
            product = np.abs(np.vdot(a, b))
            if np.any(a != b) and product > tolerance:
                #print("Non ortho: ")
                #print(product)
                #print(a)
                #print(b)
                return False
    return True


def all_non_ortho(*vecs):
    """All non-orthogonal?"""
    found_ortho = 0
    max_prod = 0
    for a in vecs:
        for b in vecs:
            product = np.abs(np.vdot(a, b))
            if np.any(a != b) and product > max_prod:
                max_prod = product
            if np.any(a != b) and product < tolerance:
                found_ortho += 1
    return found_ortho == 0

def num_ortho(*vecs):
    """Number of orthogonal pts"""
    found_ortho = 0
    max_prod = 0
    for a in vecs:
        for b in vecs:
            product = np.abs(np.vdot(a, b))
            if np.any(a != b) and product > max_prod:
                max_prod = product
            if np.any(a != b) and product < tolerance:
                found_ortho += 1
                
    return found_ortho/float(2)

def comp_basis(dim, i):
    """Computational basis vector with dimension 'dim'"""
    e = np.zeros((dim))
    e[i] = 1
    return e

def get_coeff(input, compbasis_entry):
    """Coeffiction for comp basis vector in state"""
    return np.vdot(compbasis_entry, input) 

def schmidt_decomp(v, dim_a, dim_b):
    """Schmidt decomposition by SVD using numpy"""
    # start with computational basis for H_A and H_B
    # therefore i can just read out the values of M
    M = np.zeros((dim_a, dim_b), dtype=np.complex128)
    for i in range(dim_a):
        e_i = comp_basis(dim_a, i)
        for j in range(dim_b):
            f_j = comp_basis(dim_b, j)
            M[i,j] = get_coeff(v, np.kron(e_i, f_j))

    U, sig, V = np.linalg.svd(M) 

    lefts = [0] * len(sig)
    rights = [0] * len(sig)
    coeffs = [0] * len(sig)
    for i in range(0, len(sig)):
        coeffs[i] = sig[i]
        lefts[i] = U[:, i] #U.col(i)
        rights[i] = V[i, :] # is already conj conjugate(V).col(i)

    return coeffs, lefts, rights

def get_schmidt_rank(v, dim_a, dim_b):
    """Computes schmidt rank using decomposition"""
    return len([coeff for coeff in schmidt_decomp(v, dim_a, dim_b)[0] if not np.isclose(coeff, 0)]) # extracts the length of the list of coefficients

def randnonzero():
    val = np.random.random()
    if val == 0: # if we really manage to get 0, we try again
        return randnonzero()
    else: 
        return val

def num_lin_ind(*vecs):
    """Number of linearly independent vectors"""
    M = np.row_stack(vecs)
    _, S, _ = np.linalg.svd(M)#
    # Tolerance:
    S[S < tolerance] = 0
    return np.count_nonzero(S)

def orthogonality_graph(*vecs):
    """Generates graph with edge for each non-orthogonal pair. 
    If graph not connected, non-orthogonality constraint is not satisfied."""
    # assigns an edge if 2 vectors are non-ortho
    g = nx.Graph()
    g.add_nodes_from(list(range(0, len(vecs))))
    for i, a in enumerate(vecs):
        for j, b in enumerate(vecs):
            if np.any(a != b):
                product = np.abs(np.vdot(a, b))
                if product > tolerance:
                    # non-ortho = assign edge
                    g.add_edge(i,j,weight = product)
    return g


def calc_combined_std(list_of_std):
    """calculates combined stdv of multiple stdv values

    Args:
        list_of_std (list): a list of standard deviations

    Returns:
        float: combined standard deviation
    """
    sq_sum = 0
    for std in list_of_std:
        sq_sum += std ** 2
    return np.sqrt(sq_sum / len(list_of_std))


def get_meta_for_mode(mode, data, min_val, max_val, titles, o, gate_name, ansatz):
    """function which calculates parameters for a 2d pyplot representing the data in whatever mode you want

    Args:
        mode (string): describes what kind of data you want to represent (default -> normal landscape, grad -> gradient magnitudes, log_scale -> logarithmic scale for coloring)

    Returns:
        different parameters for the pyplot
    """
    low_threshold = 0.000000001
    if mode == "default":
        c_map = "plasma"
        sup_title = f"Loss Landscapes for {ansatz}($\\phi,\\lambda)$ Approximating {gate_name} for Different Datasets"
        title = titles[o]
        v_min = min(min_val, 0)
        v_max = max(max_val, 1)
    elif mode == "grad":
        c_map = "winter"
        sup_title = "Gradient Magnitudes"
        # average gradient magnitude adjusted for sample frequency
        title = f"GM Score: {np.round(np.average(data) * len(data), 2)}"
        v_min = min(min_val, 0)
        v_max = math.ceil(max_val * 100.0) / 100.0
    elif mode == "log_scale":
        v_max = 1
        v_min = low_threshold
        c_map = "Greys"
        if min_val < low_threshold:
            min_text = "< 0.000000001"
        else:
            min_text = f"= {np.round(min_val, 10)}"
        sup_title = f"Logarithmic Loss (min. {min_text})"
        title = titles[o]
    return c_map, sup_title, title, v_min, v_max


def print_expected_output(U, x, name):
    """convience function to help print expected outputs of a unitary and input data points

    Args:
        U (tensor): a simple unitary matrix in tensor form
        x (tensor): data points
        name (string): name of the unitary/type of data
    """
    print("====")
    expected_output = torch.matmul(U, x)
    np_arr = expected_output.detach().cpu().numpy()
    print("expected output for ", name, ":\n", np_arr)
    print("====")


def print_datapoints(points, title):
    """a little helper to print data points to console more conveniently

    Args:
        points (torch tensor): tensor containing the data used to train a qnn
        title (string): describes what kind of data points (i.e. entangled..)
    """
    print("", title, " data points:")
    np_arr = points.detach().cpu().numpy()
    for i, row in enumerate(np_arr):
        print("---")
        for j, point in enumerate(row):
            print("", i, " - ", j, ":", point)


def get_k_norm(arr, k):
    """this function calculates the entry wise k-norm for an n-dimensional array https://en.wikipedia.org/wiki/Matrix_norm

    Args:
        arr (array): n-dimensional array
        k (int > 0): indicator of which norm you want to use (i.e. 1-norm, 2-norm, ...)

    Returns:
        int: the k-norm
    """
    arr = np.array(arr)
    inner_sum = 0
    for num in np.nditer(arr):
        inner_sum += np.absolute(num) ** k
    return inner_sum ** (1.0 / k)


def get_first_order_gradient_of_point(derivative_direction, target_point, landscape):
    """given an n dimensional array, this calculates the first oder gradient of a single point in a single direction

    Args:
        derivative_direction (int): id of the direction of the gradient
        target_point (list): the point you want to calculate the gradient of
        landscape (array): landscape in which the point can be found

    Returns:
        float: the first order gradient of the target point in the landscape in the target direction
    """
    grid_size = len(landscape[0])
    if target_point[derivative_direction] == 0:
        # forward diff
        leftid = list(target_point)
        leftid[derivative_direction] = leftid[derivative_direction] + 1
        rightid = list(target_point)
        leftidx = tuple(leftid)
        rightidx = tuple(rightid)
        return (landscape[leftidx] - landscape[rightidx])
    if target_point[derivative_direction] >= grid_size - 1:
        # backward diff
        leftid = list(target_point)
        rightid = list(target_point)
        rightid[derivative_direction] = rightid[derivative_direction] - 1

        leftidx = tuple(leftid)
        rightidx = tuple(rightid)
        return (landscape[leftidx] - landscape[rightidx])

    leftid = list(target_point)
    rightid = list(target_point)
    leftid[derivative_direction] = leftid[derivative_direction] + 1
    rightid[derivative_direction] = rightid[derivative_direction] - 1
    leftidx = tuple(leftid)
    rightidx = tuple(rightid)
    return (landscape[leftidx] - landscape[rightidx]) / 2


def get_second_order_gradient_of_point(first_order_direction, second_order_direction, target_point, landscape):
    """calculates the second order derivative in the specified directions

    Args:
        first_order_direction (int): the id of the direction of the first order derivative
        second_order_direction (int): the id of the direction of the second order derivative
        target_point (list): the point in the landscape to derive
        landscape (array): loss landscape

    Returns:
        float: the second order derivative at the point of the landscape specified
    """
    grid_size = len(landscape[0])
    if target_point[second_order_direction] == 0:
        # forward diff
        leftid = list(target_point)
        leftid[second_order_direction] = leftid[second_order_direction] + 1
        rightid = list(target_point)
        leftidx = tuple(leftid)
        rightidx = tuple(rightid)
        return (get_first_order_gradient_of_point(first_order_direction, leftidx,
                                                  landscape) - get_first_order_gradient_of_point(first_order_direction,
                                                                                                 rightidx, landscape))
    if target_point[second_order_direction] >= grid_size - 1:
        # backward diff
        leftid = list(target_point)
        rightid = list(target_point)
        rightid[second_order_direction] = rightid[second_order_direction] - 1
        leftidx = tuple(leftid)
        rightidx = tuple(rightid)
        return (get_first_order_gradient_of_point(first_order_direction, leftidx,
                                                  landscape) - get_first_order_gradient_of_point(first_order_direction,
                                                                                                 rightidx, landscape))
    leftid = list(target_point)
    rightid = list(target_point)
    leftid[second_order_direction] = leftid[second_order_direction] + 1
    rightid[second_order_direction] = rightid[second_order_direction] - 1
    leftidx = tuple(leftid)
    rightidx = tuple(rightid)
    return (get_first_order_gradient_of_point(first_order_direction, leftidx,
                                              landscape) - get_first_order_gradient_of_point(first_order_direction,
                                                                                             rightidx, landscape)) / 2
