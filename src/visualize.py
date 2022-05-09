
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from simulate import *
from generate_data import *
from process_data import plot_process

# global variables:
#  default parameter values when not being varied
d_default    = 2
L_default    = 50
z_default    = 2*d_default
x_default    = 0.5
T_default    = 300 # [K]
beta_default = 1/(k_B*300) # [eV^{-1}]
w_AA_default = -1.0 # [eV]
w_BB_default = 1.0
w_AB_default = -0.5
# default ranges (single point) for meshgrid
d_range_default = np.array([d_default])
L_range_default = np.array([L_default])
x_range_default = np.array([x_default])
T_range_default = np.array([T_default])
w_AA_range_default = np.array([w_AA_default])
w_BB_range_default = np.array([w_BB_default])
w_AB_range_default = np.array([w_AB_default])

# number of grid points to plot
n_grid = 10

#temporary file for CSVs when plotting
temp_file = 'data/working/temp_file'



def plot_m_AB_hat_versus_z(d_range,max_it,S_frac_tol):
    d_grid = np.array([1,2,3])
    grid_flat = generate_vertex_data(temp_file,
                                     d_grid,L_range_default,
                                     x_range_default,T_range_default,
                                     w_AA_range_default,w_BB_range_default,
                                     w_AB_range_default,
                                     max_it,S_frac_tol)
    plot_process(temp_file)
    df = pd.read_csv(temp_file)
    z_plot = df["z"].to_numpy()
    m_AB_hat_plot = df["m_AB_hat"].to_numpy()
    plt.plot(z_plot,m_AB_hat_plot,'ok')
    plt.xlabel(r'lattice coordination number, $z$')
    plt.ylabel(r'$A-B$ interactions per site, $\hat{m}_{AB}$')
    plt.show()
    os.remove(temp_file)


def plot_m_AB_hat_versus_x(x_range,max_it,S_frac_tol):
    x_grid = np.linspace(x_range[0],x_range[1],n_grid)
    grid_flat = generate_vertex_data(temp_file,
                                     d_range_default,L_range_default,
                                     x_grid,T_range_default,
                                     w_AA_range_default,w_BB_range_default,
                                     w_AB_range_default,
                                     max_it,S_frac_tol)
    plot_process(temp_file)
    df = pd.read_csv(temp_file)
    x_plot = df["x"].to_numpy()
    m_AB_hat_plot = df["m_AB_hat"].to_numpy()
    os.remove(temp_file)
    plt.plot(x_plot,m_AB_hat_plot,'ok')
    plt.xlabel(r'$B$ volume fraction, $x$')
    plt.ylabel(r'$A-B$ interactions per site, $\hat{m}_{AB}$')
    plt.show()


def plot_m_AB_hat_versus_beta(T_range,max_it,S_frac_tol):
    T_grid = np.logspace(np.log10(T_range[0]),np.log10(T_range[1]),
                         n_grid)
    T_grid = np.hstack((T_grid,np.logspace(3.0,4.5,10)))
    grid_flat = generate_vertex_data(temp_file,
                                     d_range_default,L_range_default,
                                     x_range_default,T_grid,
                                     w_AA_range_default,w_BB_range_default,
                                     w_AB_range_default,
                                     max_it,S_frac_tol)
    plot_process(temp_file)
    df = pd.read_csv(temp_file)
    beta_plot = df["beta"].to_numpy()
    m_AB_hat_plot = df["m_AB_hat"].to_numpy()
    os.remove(temp_file)
    plt.semilogx(beta_plot,m_AB_hat_plot,'ok')
    plt.xlabel(r'coldness, $\beta$ [eV$^{-1}$]')
    plt.ylabel(r'$A-B$ interactions per site, $\hat{m}_{AB}$')
    plt.show()


def plot_m_AB_hat_versus_w_AA(w_AA_range,max_it,S_frac_tol):
    w_AA_grid = np.linspace(w_AA_range[0],w_AA_range[1],n_grid)
    grid_flat = generate_vertex_data(temp_file,
                                     d_range_default,L_range_default,
                                     x_range_default,T_range_default,
                                     w_AA_grid,w_BB_range_default,
                                     w_AB_range_default,
                                     max_it,S_frac_tol)
    plot_process(temp_file)
    df = pd.read_csv(temp_file)
    w_AA_plot = df["w_AA"].to_numpy()
    m_AB_hat_plot = df["m_AB_hat"].to_numpy()
    plt.plot(w_AA_plot,m_AB_hat_plot,'ok')
    plt.xlabel(r'$A-A$ interaction strength, $w_{AA}$ [eV]')
    plt.ylabel(r'$A-B$ interactions per site, $\hat{m}_{AB}$')
    plt.show()
    os.remove(temp_file)


def plot_m_AB_hat_versus_w_BB(w_BB_range,max_it,S_frac_tol):
    w_BB_grid = np.linspace(w_BB_range[0],w_BB_range[1],n_grid)
    grid_flat = generate_vertex_data(temp_file,
                                     d_range_default,L_range_default,
                                     x_range_default,T_range_default,
                                     w_AA_range_default,w_BB_grid,
                                     w_AB_range_default,
                                     max_it,S_frac_tol)
    plot_process(temp_file)
    df = pd.read_csv(temp_file)
    w_BB_plot = df["w_BB"].to_numpy()
    m_AB_hat_plot = df["m_AB_hat"].to_numpy()
    os.remove(temp_file)
    plt.plot(w_BB_plot,m_AB_hat_plot,'ok')
    plt.xlabel(r'$B-B$ interaction strength, $w_{BB}$ [eV]')
    plt.ylabel(r'$A-B$ interactions per site, $\hat{m}_{AB}$')
    plt.show()


def plot_m_AB_hat_versus_w_AB(w_AB_range,max_it,S_frac_tol):
    w_AB_grid = np.linspace(w_AB_range[0],w_AB_range[1],n_grid)
    grid_flat = generate_vertex_data(temp_file,
                                     d_range_default,L_range_default,
                                     x_range_default,T_range_default,
                                     w_AA_range_default,w_BB_range_default,
                                     w_AB_grid,
                                     max_it,S_frac_tol)
    plot_process(temp_file)
    df = pd.read_csv(temp_file)
    w_AB_plot = df["w_AB"].to_numpy()
    m_AB_hat_plot = df["m_AB_hat"].to_numpy()
    os.remove(temp_file)
    plt.plot(w_AB_plot,m_AB_hat_plot,'ok')
    plt.xlabel(r'$A-B$ interaction strength, $w_{AB}$ [eV]')
    plt.ylabel(r'$A-B$ interactions per site, $\hat{m}_{AB}$')
    plt.show()



if (__name__ == "__main__"):
    max_it = int(200e3) # in case tolerance not met quickly
    S_frac_tol = 5e-5 # tolerance of error as fraction of mean

    # m_AB_hat versus z
    #plot_m_AB_hat_versus_z([1,3],max_it,S_frac_tol)

    # m_AB_hat versus x
    #plot_m_AB_hat_versus_x([0,0.5],max_it,S_frac_tol)

    # m_AB_hat versus beta
    #plot_m_AB_hat_versus_beta([0.001,1e6],max_it,S_frac_tol)

    # m_AB_hat versus w_AA
    plot_m_AB_hat_versus_w_AA([-1.0,1.0],max_it,S_frac_tol)

    # m_AB_hat versus w_BB
    #plot_m_AB_hat_versus_w_BB([-1.0,1.0],max_it,S_frac_tol)

    # m_AB_hat versus w_BB
    #plot_m_AB_hat_versus_w_AB([-1.0,1.0],max_it,S_frac_tol)
