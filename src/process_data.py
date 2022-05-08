import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from simulate import k_B

def join_extrema_and_random(extreme_file_name,random_file_name,
                            output_file_name):
    # join extreme points and random interior points
    # (and strip spaces from column names)
    df_extreme = pd.read_csv(extreme_file_name)
    df_random = pd.read_csv(random_file_name)
    frames = [df_extreme, df_random]
    df = pd.concat(frames,ignore_index=True)
    df.columns = df.columns.str.strip()
    df.to_csv(output_file_name,index=False)


def to_int(file_name):
    # make appropriate columns integers
    df = pd.read_csv(file_name)
    df["d"] = df["d"].to_numpy(dtype=np.int64)
    df["L"] = df["L"].to_numpy(dtype=np.int64)
    df.to_csv(file_name,index=False)


def add_z(file_name):
    # add coordination number
    df = pd.read_csv(file_name)
    d_vec = df["d"].to_numpy()
    z_vec = 2*d_vec
    df["z"] = z_vec
    df.to_csv(file_name,index=False)


def add_N(file_name):
    # add total number of atoms
    df = pd.read_csv(file_name)
    d_vec = df["d"].to_numpy()
    L_vec = df["L"].to_numpy()
    N_vec = np.power(L_vec,d_vec).astype(np.int64)
    df["N"] = N_vec
    df.to_csv(file_name,index=False)


def add_beta(file_name):
    # add thermodynamic coldness (beta)
    df = pd.read_csv(file_name)
    T_vec = df["T"].to_numpy()
    beta_vec = np.power(k_B*T_vec,-1.0)
    df["beta"] = beta_vec
    df.to_csv(file_name,index=False)


def add_x_actual(file_name):
    """
    add actual volume fraction to data set
    needed because lattice generator does integer rounding
    meaning N=5 x=0.21 results in a system like
    [1, 0, 0, 0 ,0]
    """
    df = pd.read_csv(file_name)
    N_vec = df["N"].to_numpy()
    x_vec = df["x"].to_numpy()
    # calculate real configurations based on ideal volume fraction
    N_B_vec = np.round(x_vec*N_vec).astype(np.int64)
    N_A_vec = N_vec - N_B_vec
    # compute realized (rather than ideal/specified) volume fraction
    x_actual_vec = N_B_vec/N_vec
    df["x_actual"] = x_actual_vec
    df.to_csv(file_name,index=False)


def add_U_hat(file_name):
    # add energy per site
    df = pd.read_csv(file_name)
    U_vec = df["U"].to_numpy()
    N_vec = df["N"].to_numpy()
    U_hat_vec = U_vec/N_vec
    df["U_hat"] = U_hat_vec
    df.to_csv(file_name,index=False)


def add_m_AB_hat(file_name):
    """
    add number of AB interactions per site
    mapping from U -> m_AB has singularity when w_AA=w_BB=w_AB
    giving m_AB_hat = inf (this is fixed in another step)
    """
    df = pd.read_csv(file_name)
    U_hat_vec = df["U"].to_numpy()
    z_vec = df["z"].to_numpy()
    x_vec = df["x_actual"].to_numpy()
    w_AA_vec = df["w_AA"].to_numpy()
    w_BB_vec = df["w_BB"].to_numpy()
    w_AB_vec = df["w_AB"].to_numpy()
    n_points = z_vec.shape[0]

    # compute m_AB_hat vector
    term_1 = (U_hat_vec -
              (z_vec/2)*((np.ones(n_points)-x_vec)*w_AA_vec +
                         x_vec*w_BB_vec))
    term_2 = np.power(w_AB_vec -
                      0.5*(w_AA_vec + w_BB_vec),
                      -1.0)
    m_AB_hat_vec = term_1*term_2
    df["m_AB_hat"] = m_AB_hat_vec
    df.to_csv(file_name,index=False)


def add_m_AB_hat_fixed(file_name):
    """
    fixes m_AB_hat = inf cases
    these cases occur when w_AA=w_BB=w_AB, which is consequently the
    same case when the lattice configuration doesn't matter (since
    all interactions have same strength) leading to random lattice
    the random lattice is the case described exactly by the mean
    field approximation, where
      m_AB_hat => z(1-x)x
    """
    df = pd.read_csv(file_name)
    m_AB_hat_fixed_vec = df["m_AB_hat"].to_numpy().copy()
    x_vec = df["x_actual"].to_numpy()
    for i_p,m_AB_hat in enumerate(m_AB_hat_fixed_vec):
        if (np.isinf(m_AB_hat)):
            m_AB_hat_fixed_vec[i_p] = (1 - x_vec[i_p])*x_vec[i_p]
    df["m_AB_hat_fixed"] = m_AB_hat_fixed_vec
    df.to_csv(file_name,index=False)


def build_fitting_data(processed_file_name,fitting_file_name):
    """
    restrict data with all computed quantities to only those
    necessary for fitting
    """
    # extract necessary columns
    df_proc = pd.read_csv(processed_file_name)
    x        = df_proc["x_actual"]
    beta     = df_proc["beta"]
    w_AA     = df_proc["w_AA"]
    w_BB     = df_proc["w_BB"]
    w_AB     = df_proc["w_AB"]
    U_hat    = df_proc["U_hat"]
    m_AB_hat = df_proc["m_AB_hat_fixed"]

    # construct fitting data frame
    df_fit = pd.DataFrame()
    df_fit["x"]        = x
    df_fit["beta"]     = beta
    df_fit["w_AA"]     = w_AA
    df_fit["w_BB"]     = w_BB
    df_fit["w_AB"]     = w_AB
    df_fit["U_hat"]    = U_hat
    df_fit["m_AB_hat"] = m_AB_hat

    df_fit.to_csv(fitting_file_name,index=False)



if (__name__ == "__main__"):
    # computation of additional quantities from raw data
    processed_data_file = 'data/processed/full_data.txt'

    join_extrema_and_random('data/raw/extreme_point_data.txt',
                            'data/raw/random_point_data.txt',
                            processed_data_file)
    to_int(processed_data_file)
    add_z(processed_data_file)
    add_N(processed_data_file)
    add_beta(processed_data_file)
    add_x_actual(processed_data_file)
    add_U_hat(processed_data_file)
    add_m_AB_hat(processed_data_file)
    add_m_AB_hat_fixed(processed_data_file)

    # streamlined and renamed data for fitting routines
    fitting_data_file   = 'data/processed/fitting_data.txt' 
    build_fitting_data(processed_data_file,fitting_data_file)
     
