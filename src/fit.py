import numpy as np
import scipy as sp
from scipy import optimize
import pandas as pd

def organize_fitting_data(data_file_name):
    # creates independent variable matrix of shape (n_samples, 6)
    # and dependent variable vector of shape (n_samples,)
    df = pd.read_csv(data_file_name)
    z        = df["z"].to_numpy()
    x        = df["x"].to_numpy()
    beta     = df["beta"].to_numpy()
    w_AA     = df["w_AA"].to_numpy()
    w_BB     = df["w_BB"].to_numpy()
    w_AB     = df["w_AB"].to_numpy()
    m_AB_hat = df["m_AB_hat"].to_numpy()
    p_array = np.vstack((z,x,beta,w_AA,w_BB,w_AB)).T # shape (samples, 6)
    return p_array, m_AB_hat


def extract_independent_variables(data_file_name):
    df = pd.read_csv(data_file_name)
    z        = df["z"].to_numpy()
    x        = df["x"].to_numpy()
    beta     = df["beta"].to_numpy()
    w_AA     = df["w_AA"].to_numpy()
    w_BB     = df["w_BB"].to_numpy()
    w_AB     = df["w_AB"].to_numpy()
    return z, x, beta, w_AA, w_BB, w_AB
    


def fit_model_function(fun,data_file_name):
    m_AB_hat = pd.read_csv(data_file_name)["m_AB_hat"].to_numpy()
    v_optimal, __ = sp.optimize.curve_fit(fun,data_file_name,m_AB_hat)
    return v_optimal


def evaluate_model_error(fun,v_optimal,data_file_name):
    # extract "exact" values at data points
    m_AB_hat = pd.read_csv(data_file_name)["m_AB_hat"].to_numpy()
    # compute values of model at data points with optimal parameters
    f_m_AB_hat = fun(data_file_name,*v_optimal)
    r = m_AB_hat - f_m_AB_hat  # residual vector
    mean_r = np.mean(np.abs(r))
    return mean_r


# model functions
def example_model_function(data_file_name,alpha,betap,gamma):
    # just to test I/O and get familiar with scipy's curve_fit
    z, x, beta, w_AA, w_BB, w_AB = extract_independent_variables(data_file_name)
    f_eval = np.zeros(z.shape[0])
    for i_p, __ in enumerate(z): # loop over sample points
        f_eval[i_p] = alpha*(z[i_p] + x[i_p]) + betap**2*(beta[i_p]*w_AA[i_p]) \
            + gamma**3*(w_BB[i_p] + w_AB[i_p])
    return f_eval

def f_m_AB_hat_BW(data_file_name):
    # Bragg-Williams model function for m_AB_hat
    z, x, beta, w_AA, w_BB, w_AB = extract_independent_variables(data_file_name)
    f = np.zeros(z.shape[0]) # vector of model evaluations
    for i_p, __ in enumerate(z): # loop over sample points
        f[i_p] = z[i_p]*(1-x[i_p])*x[i_p]
    return f

def f_m_AB_hat_1(data_file_name):
    pass






if (__name__ == "__main__"):
    fitting_data_file = 'data/processed/fitting_data.txt'

    """
    v_optimal_emf = fit_model_function(example_model_function,
                                       fitting_data_file)
    r_norm_emf = evaluate_model_error(example_model_function,
                                      v_optimal_emf,
                                      fitting_data_file)
    print(r_norm_emf)
    """

    # evaluate error of Bragg-Williams model function
    mean_r_BW = evaluate_model_error(f_m_AB_hat_BW,(),
                                     fitting_data_file)
    print(f'mean residual under Bragg-Williams approximation : {mean_r_BW}')


     
