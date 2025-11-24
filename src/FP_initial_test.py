from datetime import datetime
import time
import numpy as np
from ripser import ripser
from persim import plot_diagrams
import os
from matplotlib import pyplot as plt
from utils.resources import unitary, inputs_list
from qnn.qnn_cost_func import CostFunction

#TODO: look at metrics of landscapes
# metric values for config IDs 0 and 15 taken from Victors master thesis
TV = {0: [15214.401, 16309.947, 18165.08, 13718.937, 19653.626],
      15: [7612.721, 6083.612, 7238.461, 6355.254, 7938.25]}

FD = {0: [25.954254, 28.435707, 31.539053, 18.779341, 25.873669],
      15: [7.695362, 6.516226, 7.516384, 6.904895, 8.725321]}

SC_avg_abs = {0: [0.004842119076112458, 0.0057642863343205, 0.007033948578070575, 0.003624772714807306, 0.008251599227975822],
      15: [0.0015106698578506775, 0.0010675523778842504, 0.00131834851764826, 0.0011552698634541118, 0.0016972122978039436]}



u_6D = np.asarray(unitary, dtype=complex)
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

def generate_landscape_from_sample_points(sample_points, unitary=u_6D, s_rank=1, num_data_points=1, data_type=1, random=True, transform_loss=False): #TODO: Signatur anpassen, damit sie zur andere Funktion passt
  num_qubits=2
  dim = 3*num_qubits
  config_id = get_conf_id(s_rank, num_data_points, data_type)
  print(config_id)
  print("Config ID", config_id, " ----- ", "Schmidt rank: ", s_rank, "  Number of data points: ", num_data_points, "  data type: ", data_type)
  x = np.asarray(inputs_list[config_id],dtype=complex)
  loss_func = CostFunction(num_qubits=num_qubits, unitary=u_6D, inputs=x)
  landscape = []
  landscape_limit = 2 * np.pi
  landscape = []
  f = get_interval_transformation(0,2*landscape_limit)
  for i, point in enumerate(sample_points):
      loss = loss_func(point)
      if(transform_loss):
        loss = f(loss)
      data_point = np.append(point, loss)
      landscape.append(data_point)
  return np.asarray(landscape, dtype=np.float32)

# TODO: move to utils
def get_interval_transformation(a, b):
  """
    Returns a function that transforms values x in [0,1] to values f(x) in [a,b]
  """
  def f(x):
    return x*(b-1)+a
  return f

def boxplot_loss_values(transform_loss=False):
  dim = 6
  landscape_limit = 2*np.pi
  num_sample_points = 10000
  lowerleft = np.zeros(dim)
  upperright = np.ones(dim)*landscape_limit
  rng = np.random.default_rng()
  sample_points = rng.uniform(low = lowerleft, high = upperright, size = (num_sample_points, dim))

  
  # rough landscape: s rank 1, ndp 1
  loss_values_111 = []
  x_111 = np.asarray(inputs_list[0],dtype=complex)
  loss_func_111 = CostFunction(num_qubits=2, unitary=u_6D, inputs=x_111)
  # flat landscape: s rank 4, ndp 4
  loss_values_441 = []
  x_441 = np.asarray(inputs_list[15],dtype=complex)
  loss_func_441 = CostFunction(num_qubits=2, unitary=u_6D, inputs=x_441)
  
  f = get_interval_transformation(0,2*landscape_limit)
  
  for i, point in enumerate(sample_points):
      loss_111 = loss_func_111(point)
      loss_441 = loss_func_441(point)
      if(transform_loss):
        loss_111 = f(loss_111)
        loss_441 = f(loss_441)
      loss_values_111.append(loss_111)
      loss_values_441.append(loss_441)
  
  print(loss_values_111)
  
  # boxplot of loss values
  data_dict = {"rough landscape (111)": loss_values_111, "flat landscape (441)": loss_values_441}
  title = "QNN Loss Landscape values \n (Schmidt Rank, size training set, data type)"
  fig, ax = plt.subplots()
  ax.boxplot(data_dict.values())
  ax.set_xticklabels(data_dict.keys(), fontsize=14)
  ax.set_title(title, fontsize=16)
  file_name = "loss_values_transformed_2"
  plt.savefig(f"plots/Second_QNN_tests/{file_name}.pdf", format='pdf',bbox_inches='tight')



def scikit_tda_test():
  start = datetime.now()
  print("Start time", start.strftime('%Y-%m-%d %H:%M:%S'))
  s = 1
  ndp = 1
  dt = 1
  nsp = 100
  dim=6
  landscape = generate_landscape(s_rank=s, num_data_points=ndp, data_type=dt, num_sample_points=nsp)
  print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "Generating loss landscape: DONE")
  diagrams = ripser(landscape, maxdim=2)['dgms']
  # Time
  elapsed_time = datetime.now()-start
  print("elapsed time", elapsed_time)
  title=f"6D QNN loss landscape\n Schmidt-Rank = {s}, #Trainings samples = {ndp}, Data type = {dt}\n #Sample points = {nsp}"
  plot_diagrams(diagrams, show=True, title=title)


if __name__ == "__main__":
  scikit_tda_test()
  #print(os.getcwd())
  #print(generate_landscape(s_rank=1,num_data_points=2,data_type=1,num_grid_points=5))
  