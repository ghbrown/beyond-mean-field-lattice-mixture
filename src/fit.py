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
    


def fit_model_function(fun,data_file_name,p0=None):
    m_AB_hat = pd.read_csv(data_file_name)["m_AB_hat"].to_numpy()
    v_optimal, __ = sp.optimize.curve_fit(fun,data_file_name,m_AB_hat,
                                          p0=p0)
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
    # model function 1
    z, x, beta, w_AA, w_BB, w_AB = extract_independent_variables(data_file_name)
    f = np.zeros(z.shape[0]) # vector of model evaluations
    for i_p, __ in enumerate(z): # loop over sample points
        f[i_p] = (2/(1 + np.exp(beta[i_p]*(w_AB[i_p]-0.5*(w_AA[i_p] + w_BB[i_p]))))) \
            *z[i_p]*(1-x[i_p])*x[i_p]
    return f


def f_m_AB_hat_2(data_file_name,a_2,b_2,c_2,d_2):
    # model function 2
    z, x, beta, w_AA, w_BB, w_AB = extract_independent_variables(data_file_name)
    f = np.zeros(z.shape[0]) # vector of model evaluations
    for i_p, __ in enumerate(z): # loop over sample points
        f[i_p] = ((a_2 + b_2) \
                  /(a_2 + b_2*np.exp(beta[i_p]*(c_2*w_AB[i_p]-d_2*0.5*(w_AA[i_p] + w_BB[i_p]))))) \
                  *z[i_p]*(1-x[i_p])*x[i_p]
    return f


def f_m_AB_hat_3(data_file_name,a_3):
    # model function 3
    z, x, beta, w_AA, w_BB, w_AB = extract_independent_variables(data_file_name)
    f = np.zeros(z.shape[0]) # vector of model evaluations
    for i_p, __ in enumerate(z): # loop over sample points
        f[i_p] = np.power(z[i_p],a_3)*(1-x[i_p])*x[i_p]
    return f


def f_m_AB_hat_4(data_file_name,a_4):
    # model function 4
    z, x, beta, w_AA, w_BB, w_AB = extract_independent_variables(data_file_name)
    f = np.zeros(z.shape[0]) # vector of model evaluations
    for i_p, __ in enumerate(z): # loop over sample points
        # exchange parameter
        psi_cur = beta[i_p]*(w_AB[i_p]-0.5*(w_AA[i_p] + w_BB[i_p]))
        g_cur = (2/np.pi)*(a_4-0.5)*np.arctan(psi_cur) + 0.5
        f[i_p] = (1/(1 + np.exp(psi_cur)) + g_cur) \
            *z[i_p]*(1-x[i_p])*x[i_p]
    return f




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

    # solve and evaluate models
    # Bragg-Williams model function
    mean_r_BW = evaluate_model_error(f_m_AB_hat_BW,(),
                                     fitting_data_file)

    # model function 1
    mean_r_1 = evaluate_model_error(f_m_AB_hat_1,(),
                                     fitting_data_file)

    # model function 2
    v_optimal_2 = fit_model_function(f_m_AB_hat_2,
                                     fitting_data_file)
    mean_r_2 = evaluate_model_error(f_m_AB_hat_2,v_optimal_2,
                                     fitting_data_file)

    # model function 3
    v_optimal_3 = fit_model_function(f_m_AB_hat_3,
                                     fitting_data_file,
                                     p0=np.array([1.0]))
    mean_r_3 = evaluate_model_error(f_m_AB_hat_3,v_optimal_3,
                                     fitting_data_file)

    # model function 4
    v_optimal_4 = fit_model_function(f_m_AB_hat_4,
                                     fitting_data_file)
    mean_r_4 = evaluate_model_error(f_m_AB_hat_4,v_optimal_4,
                                     fitting_data_file)


                        
    # print model summaries
    print(f'\n\n')

    print(f'mean residual under Bragg-Williams approximation : {mean_r_BW}')

    print(f'mean residual under model function 1 : {mean_r_1}')

    print(f'mean residual under model function 2 : {mean_r_2}')
    print(f'a_2: {v_optimal_2[0]} \nb_2: {v_optimal_2[1]} \nc_2: {v_optimal_2[2]} \nd_2: {v_optimal_2[3]}')

    print(f'mean residual under model function 3 : {mean_r_3}')
    print(f'a_3: {v_optimal_3[0]}')

    print(f'mean residual under model function 4 : {mean_r_4}')
    print(f'a_4: {v_optimal_4[0]}')




     
