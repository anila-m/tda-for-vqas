import torch
import orqviz
import numpy as np
import scipy as sp
import math
#from classic_training import cost_func

from src.utils.metric_utils import *
#from landscapes import *
from data_utils import calc_hessian, sample_n_ball_uniform, get_hypersphere_volume



# total absolute scalar curvature
# BA CurvaQ
def calc_total_absolute_scalar_curvature(function, r, c, sampling="uniform", N=1000, absolute=True):
    '''
    Calculates the total absolute scalar curvature of a function within a hypersphere of 
    radius r around a center point c.

    Args:
        function (array): function 
        r (float): radius of hypersphere, same in every dimension
        c (array): point within loss landscape, center of hypersphere, array with n entries (one for each dimension)
        sampling (String): sampling method, possible values: 
            "uniform" (uniformly random) (default)
            [so far no other sampling methods are implemented]
        N (int): number of sample points, default: 1000 
        absolute (bool): default: True, if False: non absolute scalar curvature values are used

    Returns:
        float: total absolute scalar curvature
        array: list of all scalar curvature values
        sample points: list of all sample points
    '''
    dimensions = len(c)
    # get sample points within hypersphere
    sample_points = sample_n_ball_uniform(dimensions, r, c, N)
    
    scalar_curvature_landscape, grad_summary, hess_summary = calc_scalar_curvature_for_function(function, sample_points)
    
    # get volume of hypersphere
    hypersphere_volume = get_hypersphere_volume(dimensions, r)
    # compute total absolute sc
    total_absolute_sc = np.sum(np.absolute(scalar_curvature_landscape))
    if not absolute: #if total scalar curvature is to be calculated, not absolute sc
        total_absolute_sc = np.sum(scalar_curvature_landscape)
    total_absolute_sc = total_absolute_sc * hypersphere_volume/N
    return np.round(total_absolute_sc, 3), scalar_curvature_landscape, sample_points

# BA CurvaQ
def calc_several_scalar_curvature_values(function, r, c, N=1000, absolute=True):
    '''
    Calculates the total (absolute) scalar curvature and mean (absolute) scalar curvature
    of a function within a hypersphere of radius r around a center point c.

    Args:
        function (array): function 
        r (float): radius of hypersphere, same in every dimension
        c (array): point within loss landscape, center of hypersphere, array with n entries (one for each dimension)
        sampling (String): sampling method, possible values: 
            "uniform" (uniformly random, Marsaglia method) (default)
        N (int): number of sample points, default: 1000 #TODO: andere Zahl?

    Returns:
        float: total absolute scalar curvature
        float: total scalar curvature
        float: mean absolute scalar curvature
        float: mean scalar curvature
        array: [median, mean, min, max] of scalar curvature values at sample points
        array: [median, mean, min, max] of gradient norms at sample points
        array: [median, mean, min, max] of hessian norms at sample points
    '''
    dimensions = len(c)
    # get sample points within hypersphere
    sample_points = sample_n_ball_uniform(dimensions, r, c, N)
    scalar_curvature_landscape, grad_summary, hess_summary = calc_scalar_curvature_for_function(function, sample_points)
    sc_summary = [np.median(scalar_curvature_landscape), np.mean(scalar_curvature_landscape), np.min(scalar_curvature_landscape), np.max(scalar_curvature_landscape)]
    # calculate total (absolute) scalar curvature
    # get volume of hypersphere
    hypersphere_volume = get_hypersphere_volume(dimensions, r)
    # compute total absolute sc
    total_absolute_sc = np.sum(np.absolute(scalar_curvature_landscape))
    total_sc = np.sum(scalar_curvature_landscape)
    total_absolute_sc = total_absolute_sc * hypersphere_volume/N
    total_sc = total_sc * hypersphere_volume/N

    # calculate mean (aboslute) scalar curvature
    mean_sc = np.mean(scalar_curvature_landscape)
    mean_asc = np.mean(np.absolute(scalar_curvature_landscape))
    return np.round(total_absolute_sc,3), np.round(total_sc,3), np.round(mean_asc,3), np.round(mean_sc,3), sc_summary, grad_summary, hess_summary

# n-dimensional scalar curvature, for an objective function and a list of points, not "just" a landscape
# BA CurvaQ
def calc_scalar_curvature_for_function(function, sample_points):
    """calculates the scalar curvature of a function at different points
    instead of calculating the whole n dimensional curvature array (same size as the input landscape)
    this function calculates the scalar curvature at each entry of the n dimensional landscape 
    and puts them back together into an output array

    Args:
        function (callable): callable function, which 
        sample_points (array): n dimensional sample_points array

    Returns:
        array: n dimensional scalar curvature array
        array: [median, mean, min, max] of gradient norms at sample points
        array: [median, mean, min, max] of hessian norms at sample points

    """
    sample_points = np.asarray(sample_points)
    scalar_curvature = np.ndarray(sample_points.shape[0])
    gradients = []
    hessians = []

    # iterate over all sample points 
    for idx in range(sample_points.shape[0]):
        # approximate dimsXdims sized hessian and dims sized vector of gradients for a specific point of the loss landscape
        # ------ ab hier anders --------
        gradient_vector = sp.optimize.approx_fprime(sample_points[idx], function) # funktioniert mit Kostenfunktion
        point_hessian = calc_hessian(function, sample_points[idx,:])
        gradients.append(gradient_vector)
        hessians.append(point_hessian)
        # ------ bis hier anders -------
        # calculate scalar curvature from here
        beta = 1 / (1 + np.linalg.norm(gradient_vector) ** 2)
        left_term = beta * (
            np.trace(point_hessian) ** 2
            - np.trace(np.matmul(point_hessian, point_hessian))
        )
        right_inner = np.matmul(point_hessian, point_hessian) - np.trace(
            point_hessian
        ) * point_hessian
        # order of matmul with gradient does not matter
        right_term = (
            2
            * (beta**2)
            * (np.matmul(np.matmul(gradient_vector.T, right_inner), gradient_vector))
        )
        point_curv = left_term + right_term
        scalar_curvature[idx] = point_curv
    grad_norm = np.linalg.norm(np.asarray(gradients), axis=1)
    hess_norm = np.linalg.norm(np.asarray(hessians), axis=(1,2))
    gradient_summary = [float(np.median(grad_norm)), float(np.mean(grad_norm)), float(np.min(grad_norm)), float(np.max(grad_norm))]
    hessian_summary = [float(np.median(hess_norm)), float(np.mean(hess_norm)), float(np.min(hess_norm)), float(np.max(hess_norm))]
    return scalar_curvature, gradient_summary, hessian_summary

# n-dimensional scalar curvature
def calc_scalar_curvature(landscape):
    """calculates the scalar curvature of a loss landscape
    instead of calculating the whole n dimensional curvature array (same size as the input landscape)
    this function calculates the scalar curvature at each entry of the n dimensional landscape 
    and puts them back together into an output array

    Args:
        landscape (array): n dimensional loss landscape array

    Returns:
        array: n dimensional scalar curvature array
    """
    landscape = np.asarray(landscape)
    scalar_curvature = np.ndarray(landscape.shape)
    dims = len(landscape.shape)
    # iterate over all landscape entries where idx is the exact position in the array (i.e: idx = (11, 2, 9, 10) -> arr[11][2][9][10] for a 4param qnn)
    for idx, _ in np.ndenumerate(scalar_curvature):
        # generate dimsXdims hessian and dims sized vector of gradients for a specific point of the loss landscape
        point_hessian = []
        gradient_vector = []
        for i in range(dims):
            #get gradient vector
            gradient_vector.append(get_first_order_gradient_of_point(i, idx, landscape))
            row = []
            for j in range(dims):
                # append e.g. [[0],[1]],[[2],[3]] for 2d
                row.append(get_second_order_gradient_of_point(i,j,idx,landscape))
            point_hessian.append(row)
        point_hessian = np.asarray(point_hessian)
        gradient_vector = np.asarray(gradient_vector)
        # calculate scalar curvature from here
        beta = 1 / (1 + np.linalg.norm(gradient_vector) ** 2)
        left_term = beta * (
            np.trace(point_hessian) ** 2
            - np.trace(np.matmul(point_hessian, point_hessian))
        )
        right_inner = np.matmul(point_hessian, point_hessian) - np.trace(
            point_hessian
        ) * point_hessian
        # order of matmul with gradient does not matter
        right_term = (
            2
            * (beta**2)
            * (np.matmul(np.matmul(gradient_vector.T, right_inner), gradient_vector))
        )
        point_curv = left_term + right_term
        scalar_curvature[idx] = point_curv
    return scalar_curvature


def calc_total_variation(landscape):
    """calculates the total variation of a landscape

    Args:
        landscape (array): n dimensional loss landscape as an n dimensional array
    """
    dimensions = len(np.array(landscape).shape)
    #print(dimensions)
    lanscape_limit = 2 * math.pi
    length = np.array(landscape).shape[0]
    step_size = lanscape_limit / length
    gradients = np.gradient(np.array(landscape))
    total_variation = np.sum(np.absolute(gradients))
    # normalize it by step size
    #using dimensions -1 gives more stable results w.r.t. the number of samples per dimension
    total_variation = total_variation * step_size**(dimensions)
    return np.round(total_variation, 3)


def calc_IGSD(landscape):
    """calculates the inverse gradient standard deviation of a landscape

    Args:
        landscape (array): n dimensional loss landscape array

    Returns:
        array: returns a list of IGSDs, one for each dimension 
    """
    gradients = np.gradient(np.array(landscape))
    # each array of the gradients encompasses the gradients for one dimension/direction/parameter
    gradient_standard_deviations = []
    for dimension in gradients:
        gradient_standard_deviations.append(np.std(dimension))

    inverse_gradient_standard_deviations = np.divide(1, gradient_standard_deviations)

    #print(landscape)
    return np.round(inverse_gradient_standard_deviations, 3)


def calc_fourier_density(landscape) -> float:
    """same as calculate_fourier_density below 
    but with custom k-norm function and rounded to 6 digits

    Args:
        landscape (array): n dimensional landscape array
    """
    fourier_result = np.fft.fftshift(np.fft.fftn(landscape, norm="forward")) #Fourier Coefficients (oder?)
    fourier_density = round(
        (get_k_norm(fourier_result, 1) ** 2) / (get_k_norm(fourier_result, 2) ** 2),
        6,
    )
    return fourier_density

# Alina
def calc_fourier_density_and_coefficients(landscape):
    """same as calc_fourier_density above 
    but also returns the calculated coefficients

    Args:
        landscape (array): n dimensional landscape array
    """
    fourier_result = np.fft.fftshift(np.fft.fftn(landscape, norm="forward")) #Fourier Coefficients (oder?)
    fourier_density = round(
        (get_k_norm(fourier_result, 1) ** 2) / (get_k_norm(fourier_result, 2) ** 2),
        6,
    )
    return fourier_density, fourier_result


# calculates the fourier density by reshaping the fourier result to get an vector of Fourier coefficients
def calculate_fourier_density(
    landscape,
) -> float:
    """calculates the fourier density of a given landscape

    Args:
        landscape (array): n-dim landscape

    """
    fourier_result = np.fft.fftshift(np.fft.fftn(landscape, norm="forward"))

    # reshape the fourier result into a vector according to the paper
    vector_fourier_result = fourier_result.reshape(-1)

    # sum the absolute values of each component of the vector
    one_norm = np.sum(np.abs(vector_fourier_result))

    # frobenius norm
    two_norm = np.linalg.norm(vector_fourier_result)
    return one_norm**2 / two_norm**2



def calc_grad_curv(landscape):
    """calculates the gradient curvature (custom metric consisting of the second order gradient magnitudes) for a given landscape

    Args:
        landscape (array): landscape of which you want to calculate the curvature

    Returns:
        array: array of curvature for every point in the landscape
    """
    first_order_gradients = np.gradient(np.array(landscape))
    second_order_gradients = []
    for grad in first_order_gradients:
        grads_of_grad = np.gradient(np.array(grad))
        for sec_grad in grads_of_grad:
            second_order_gradients.append(sec_grad)
    magnitude_sum = 0
    for g in second_order_gradients:
        magnitude_sum += g**2
    curv_mag = np.sqrt(magnitude_sum)
    return curv_mag
