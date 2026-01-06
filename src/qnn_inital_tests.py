from datetime import datetime
import time
import numpy as np
from ripser import ripser
from persim import plot_diagrams
import os
from matplotlib import pyplot as plt
from utils.resources import unitary, inputs_list, unitary_2, unitary_5, inputs_79, inputs_241
from qnn.qnn_cost_func import CostFunction
from utils.sampling_utils import get_uniformly_random_samples, get_latin_hypercube_samples
from scipy.stats import qmc
from scipy.optimize import approx_fprime
import json
from utils.data_utils import get_interval_transformation

#TODO: look at metrics of landscapes
# metric values for config IDs 0 and 15 taken from Victors master thesis
TV = {0: [15214.401, 16309.947, 18165.08, 13718.937, 19653.626],
      15: [7612.721, 6083.612, 7238.461, 6355.254, 7938.25]}

FD = {0: [25.954254, 28.435707, 31.539053, 18.779341, 25.873669],
      15: [7.695362, 6.516226, 7.516384, 6.904895, 8.725321]}

SC_avg_abs = {0: [0.004842119076112458, 0.0057642863343205, 0.007033948578070575, 0.003624772714807306, 0.008251599227975822],
      15: [0.0015106698578506775, 0.0010675523778842504, 0.00131834851764826, 0.0011552698634541118, 0.0016972122978039436]}

u_6D = np.asarray(unitary, dtype=complex)
u_6D_2 = np.asarray(unitary_2, dtype=complex) # for config 241
u_6D_5 = np.asarray(unitary_5, dtype=complex)# for config 79
in_79 = np.asarray(inputs_79, dtype=complex)
in_241 = np.asarray(inputs_241, dtype=complex)
#6D QNN cost function: Schmidt Rank 1, num data points 

def get_conf_id(s_rank, num_data_points, data_type):
  config_id = (data_type-1)*4*4 + (num_data_points-1)*4+(s_rank-1)
  return config_id

def generate_landscape(unitary=u_6D, s_rank=1, num_data_points=1, data_type=1, num_sample_points=100, random=True):
  num_qubits=2
  dim = 3*num_qubits
  config_id = get_conf_id(s_rank, num_data_points, data_type)
  print("Config ID", config_id, " ----- ", "Schmidt rank: ", s_rank, "  Number of data points: ", num_data_points, "  data type: ", data_type, "  Number Sample points: ", num_sample_points)
  x = np.asarray(inputs_list[config_id],dtype=complex)
  loss_func = CostFunction(num_qubits=num_qubits, unitary=u_6D, inputs=x)
  landscape = []

  landscape_limit = 2 * np.pi
  lowerleft = np.zeros(dim)
  upperright = np.ones(dim)*landscape_limit
  rng = np.random.default_rng()
  #TODO: use Latin hypercube sampling to generate sample points?
  sample_points = rng.uniform(low = lowerleft, high = upperright, size = (num_sample_points, dim))
  landscape = []
  for i, point in enumerate(sample_points):
      loss = loss_func(point)
      data_point = np.append(point, loss)
      landscape.append(data_point)
  return np.asarray(landscape, dtype=np.float32)

def generate_landscape_from_sample_points(sample_points, unitary=u_6D, s_rank=1, num_data_points=1, data_type=1, inputs =None, random=True, transform_loss=False, min=0, max=1): #TODO: Signatur anpassen, damit sie zur andere Funktion passt
  num_qubits=2
  dim = 3*num_qubits
  config_id = get_conf_id(s_rank, num_data_points, data_type)
  #print(config_id)
  #print("Config ID", config_id, " ----- ", "Schmidt rank: ", s_rank, "  Number of data points: ", num_data_points, "  data type: ", data_type)
  if inputs is None:
    x = np.asarray(inputs_list[config_id],dtype=complex)
  else:
    print("inputs")
    x = inputs
  loss_func = CostFunction(num_qubits=num_qubits, unitary=unitary, inputs=x)
  landscape = []
  landscape_limit = 2 * np.pi
  landscape = []
  f = get_interval_transformation(min,max)
  for i, point in enumerate(sample_points):
      loss = loss_func(point)
      if(transform_loss):
        loss = f(loss)
      data_point = np.append(point, loss)
      landscape.append(data_point)
  return np.asarray(landscape, dtype=np.float32)

# TODO: move to utils


def boxplot_loss_values(transform_loss=False):
  dim = 6
  landscape_limit = 2*np.pi
  num_sample_points = 10000
  sample_points = get_latin_hypercube_samples(min=0, max=landscape_limit, dim=dim, number_of_samples=num_sample_points)
  
  # rough landscape: s rank 1, ndp 1
  loss_values_111 = []
  loss_gradient_111 = []
  x_111 = np.asarray(inputs_list[0],dtype=complex)
  loss_func_111 = CostFunction(num_qubits=2, unitary=u_6D, inputs=x_111)
  # flat landscape: s rank 4, ndp 4
  loss_values_441 = []
  loss_gradient_441 = []
  x_441 = np.asarray(inputs_list[15],dtype=complex)
  loss_func_441 = CostFunction(num_qubits=2, unitary=u_6D, inputs=x_441)
  
  f = get_interval_transformation(50,100)
  
  for i, point in enumerate(sample_points):
      loss_111 = loss_func_111(point)
      loss_441 = loss_func_441(point)

      loss_values_111.append(loss_111)
      loss_values_441.append(loss_441)

      loss_grad_111 = approx_fprime(point, loss_func_111)
      loss_grad_441 = approx_fprime(point, loss_func_441)

      loss_gradient_111.append(np.linalg.norm(loss_grad_111)) # 2-norm of gradient
      loss_gradient_441.append(np.linalg.norm(loss_grad_441)) # 2-norm of gradient

  
  
  # boxplot of loss values
  data_dict = {"rough landscape (111)": loss_values_111, "flat landscape (441)": loss_values_441}
  title = "QNN Loss Landscape values \n (Schmidt Rank, size training set, data type)"
  fig, ax = plt.subplots()
  ax.boxplot(data_dict.values())
  ax.set_xticklabels(data_dict.keys(), fontsize=14)
  ax.set_title(title, fontsize=16)
  file_name = "boxplot_loss_values_latin"
  plt.savefig(f"experiment_results/plots/Initial_QNN_tests/{file_name}.pdf", format='pdf',bbox_inches='tight')

  # boxplot of loss values
  data_dict = {"rough landscape (111)": loss_gradient_111, "flat landscape (441)": loss_gradient_441}
  title = "QNN Loss Gradients (norm) \n (Schmidt Rank, size training set, data type)"
  fig, ax = plt.subplots()
  ax.boxplot(data_dict.values())
  ax.set_xticklabels(data_dict.keys(), fontsize=14)
  ax.set_title(title, fontsize=16)
  file_name = "boxplot_loss_gradients_latin"
  plt.savefig(f"experiment_results/plots/Initial_QNN_tests/{file_name}.pdf", format='pdf',bbox_inches='tight')


def scikit_tda_test(min=0, max=1):
  start = datetime.now()
  print("Start time", start.strftime('%Y-%m-%d %H:%M:%S'))
  s = 4
  ndp = 4
  dt = 1
  
  dim=6
  #landscape = generate_landscape(s_rank=s, num_data_points=ndp, data_type=dt, num_sample_points=nsp)
  with open("resources/sample_points_1000_6D_0_2pi.json") as f:
    sample_points =  json.load(f)
  nsp = len(sample_points)
  landscape = generate_landscape_from_sample_points(sample_points=sample_points, s_rank=s, num_data_points=ndp, transform_loss=False, min=min, max=max)
  print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "Generating loss landscape: DONE")
  diagrams = ripser(landscape, maxdim=2)['dgms']
  # Time
  elapsed_time = datetime.now()-start
  print("elapsed time", elapsed_time)
  title=f"6D QNN loss landscape\n Schmidt-Rank = {s}, #Trainings samples = {ndp}, Data type = {dt}\n #Sample points = {nsp}"
  a = plot_diagrams(diagrams, show=False, title=title)
  print(diagrams)

def scikit_tda_test2(min=0, max=1):
  """
  TDA of QNN cost landscape.

  :param min: lower bound of new interval (linear transformation of loss values), default: 0. 
  :param max: upper bound of new interval (linear transformation of loss values), default: 1.
  """
  start = datetime.now()
  print("Start time", start.strftime('%Y-%m-%d %H:%M:%S'))
  # config 79 (victors numbering) uses unitary_5, inputs_79 (inputs = training samples)
  # config 241 would be: s= 1, ndp=1, dt=4, unitary_2, inputs_241
  s = 4
  ndp = 4
  dt = 1
  dim=6
  #landscape = generate_landscape(s_rank=s, num_data_points=ndp, data_type=dt, num_sample_points=nsp)
  with open("resources/sample_points_1000_6D_0_2pi.json") as f:
    sample_points =  json.load(f)
  nsp = len(sample_points)
  landscape = generate_landscape_from_sample_points(sample_points=sample_points, unitary=unitary_5, inputs=in_79, transform_loss=True, min=min, max=max)
  print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "Generating loss landscape: DONE")
  diagrams = ripser(landscape, maxdim=2)['dgms']
  # Time
  elapsed_time = datetime.now()-start
  print("elapsed time", elapsed_time)
  title=f"6D QNN loss landscape\n Schmidt-Rank = {s}, #Trainings samples = {ndp}, Data type = {dt}\n #Sample points = {nsp}"
  plot_diagrams(diagrams, show=True, title=title, xy_range = [-0.5,10, -0.5, 10]) # axis limits fit transformed loss values (to [50,100])




if __name__ == "__main__":
  #scikit_tda_test()
  #print(os.getcwd())
  #print(generate_landscape(s_rank=1,num_data_points=2,data_type=1,num_grid_points=5))
  # s1 = get_latin_hypercube_samples(min = 0, max = 1, dim = 6, number_of_samples=1000)
  # s2 = get_uniformly_random_samples(min = 0, max = 1, dim = 6, number_of_samples=1000)
  # print("Latin Discrepancy: ", qmc.discrepancy(s1))
  # print("Uniform Discrepancy: ", qmc.discrepancy(s2))

  scikit_tda_test2(min=50, max=100) 