
import copy
import numpy as np
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from simulate import *


k_B = 8.617e-5 # Boltzmann constant, [eV K^-1]

def extreme_points(dim_range,L_range,x_range,T_range,
                   w_AA_range,w_BB_range,w_AB_range):
    # enumerates all vertices of hypercube spanned by extreme
    # values of parameters
    grid_tuple = (dim_range,L_range,x_range,T_range,
                   w_AA_range,w_BB_range,w_AB_range)
    arrays = np.meshgrid(*grid_tuple)
    # reshape into order 2 array of shape
    #   (n_vertices,number of parameters)
    vertices_flat = np.reshape(arrays,(7,2**7)).T 
    return vertices_flat


def sample_parameter_point(dim_range,L_range,x_range,T_range,
                           w_AA_range,w_BB_range,w_AB_range):
    # uniformly samples a random point in the given
    dim = np.random.randint(low=dim_range[0],high=dim_range[1]+1)
    L = np.random.randint(low=L_range[0],high=L_range[1]+1)
    x = np.random.uniform(low=x_range[0],high=x_range[1])
    T_exponent = np.random.uniform(low=np.log10(T_range[0]),
                                   high=np.log10(T_range[1]))
    T = np.power(10,T_exponent)
    w_AA = np.random.uniform(low=w_AA_range[0],high=w_AA_range[1])
    w_BB = np.random.uniform(low=w_BB_range[0],high=w_BB_range[1])
    w_AB = np.random.uniform(low=w_AB_range[0],high=w_AB_range[1])
    parameter_point = np.array([dim, L, x, T, w_AA, w_BB, w_AB])
    return parameter_point


def run_parameter_point(parameter_point,S_frac_tol):
    # wrapper to take parameter configuration and return U
    # by running Monte Carlo
    pp = parameter_point # alias
    max_it = np.inf
    initialization = 'random'
    print_conv = 'yes'
    stop_interval = 5000
    dim  = int(pp[0]) # required conversion to int
    L    = int(pp[1]) # required conversion to int
    x    = pp[2]
    T    = pp[3]
    w_AA = pp[4]
    w_BB = pp[5]
    w_AB = pp[6]
    beta = 1/(k_B*T)
    __, U_vec = lattice_mixture_monte_carlo(dim,L,x,beta,
                                            w_AA,w_BB,w_AB,
                                            max_it,S_frac_tol,
                                            initialization=initialization,
                                            check_interval=stop_interval,
                                            print_conv=print_conv)
    U = np.mean(U_vec)
    return U


def point_entry_string(parameter_point,U):
    pp = parameter_point # alias
    dim  = pp[0]
    L    = pp[1]
    x    = pp[2]
    T    = pp[3]
    w_AA = pp[4]
    w_BB = pp[5]
    w_AB = pp[6]
    out_string = f'{dim}, {L}, {x}, {T}, {w_AA}, {w_BB}, {w_AB}, {U}\n'
    return out_string


def generate_vertex_data(data_file_name,
                         dim_range,L_range,x_range,T_range,
                         w_AA_range,w_BB_range,w_AB_range,
                         S_frac_tol):
    # overwrite data (always same points) and write header
    with open(data_file_name,'w+') as f:
        header_string = 'd, L, x, T, w_AA, w_BB, w_AB, U\n'
        f.writelines([header_string])
    
    # run all extreme points
    extrema_flat = extreme_points(dim_range,L_range,x_range,T_range,
                                  w_AA_range,w_BB_range,w_AB_range)
    n_points = extrema_flat.shape[0]
    for i_point, point in enumerate(extrema_flat):
        print(f'running extreme point {i_point+1}/{n_points}')
        # generate random parameter point in parameter space
        parameter_point = sample_parameter_point(dim_range,L_range,
                                                 x_range,T_range,
                                                 w_AA_range,w_BB_range,
                                                 w_AB_range)
        # calculate U for mixture on lattice defined by parameters
        U_point = run_parameter_point(parameter_point,S_frac_tol)
        # write parameter point and energy to file
        entry_string = point_entry_string(parameter_point,U_point)
        with open(data_file_name,'a') as f:
            f.writelines([entry_string])


def generate_random_data(data_file_name,max_points,
                        dim_range,L_range,x_range,T_range,
                        w_AA_range,w_BB_range,w_AB_range,
                         S_frac_tol):
    # write header if file empty
    with open(data_file_name,'r') as f:
        lines = f.readlines()
    if (len(lines) == 0):
        header_string = 'd, L, x, T, w_AA, w_BB, w_AB, U\n'
        with open(data_file_name,'w+') as f:
            f.writelines([header_string])
    
    # run random points
    n_points = 0
    while (n_points < max_points):
        print(f'running random point {n_points+1}/{max_points}')
        # generate random parameter point in parameter space
        parameter_point = sample_parameter_point(dim_range,L_range,
                                                 x_range,T_range,
                                                 w_AA_range,w_BB_range,
                                                 w_AB_range)
        # calculate U for mixture on lattice defined by parameters
        U_point = run_parameter_point(parameter_point,S_frac_tol)
        # write parameter point and energy to file
        entry_string = point_entry_string(parameter_point,U_point)
        with open(data_file_name,'a') as f:
            f.writelines([entry_string])
        n_points += 1



if (__name__ == "__main__"):
    dim_range  = np.array([1,3])
    L_range    = np.array([5,100])
    x_range    = np.array([0.0,0.5])
    T_range    = np.array([1e-3,1e6]) # [K]
    w_AA_range = np.array([-1.0,1.0]) # [eV]
    w_BB_range = np.array([-1.0,1.0]) # [eV]
    w_AB_range = np.array([-1.0,1.0]) # [eV]
    S_frac_tol = 5e-5 # tolerance of error as fraction of mean

    generate_vertex_data('data/extreme_point_data.txt',
                         dim_range,L_range,x_range,T_range,
                         w_AA_range,w_BB_range,w_AB_range,
                         S_frac_tol)

    #generate_random_data('data/random_point_data.txt',10,
    #                     dim_range,L_range,x_range,T_range,
    #                     w_AA_range,w_BB_range,w_AB_range,
    #                     S_frac_tol)


     
