import copy
import numpy as np
import scipy as sp
from scipy import ndimage
import matplotlib.pyplot as plt

# global variables: k_B, kernelNNList

k_B = 8.617e-5 # Boltzmann constant, [eV K^-1]

def latticeToInteger(lattice):
    """
    converts a lattice with 0s and 1s on sites to corresponding
    integer
    """
    latticeFlat = lattice.ravel()
    powersOf2Vec = np.power(np.full(latticeFlat.shape,2,dtype=int),
                            np.arange(0,latticeFlat.shape[0],dtype=int))
    powersOf2Vec = np.flip(powersOf2Vec)
    latticeInteger = np.dot(powersOf2Vec,latticeFlat)
    return latticeInteger


def createLattice(dim,N_cellPerEdge,vFrac):
    """
    generates a random lattice of volume fraction vFrac in dimension
    dim of size N_cellPerEdge
    ---Inputs---
    dim : {integer}
        number of spatial dimensions of square lattice
    N_cellPerEdge : {integer}
        number of lattice sites along one edge of lattice
    vFrac : {float}
        number of B sites divided by total number of sites
    ---Outputs---
    lattice : {numpy array}
        order dim tensor of uniform dimension N_cellPerEdge filled
        with 0s (A sites) and 1s (B sites)
    """
    shapeTuple = tuple([N_cellPerEdge]*dim)
    N_cellTotal=int(np.power(N_cellPerEdge,dim))
    N_cellB=int(np.round(vFrac*N_cellTotal))
    N_cellA=N_cellTotal-N_cellB
    filledCounter = 0
    latticeList = [0]*N_cellA
    #generate random insertion indices for each B site
    randVec = [np.random.randint(N_cellTotal-i) for i
               in range(N_cellB)]
    #insert B sites into initially A-only vector
    for i_pos in reversed(randVec):
        latticeList.insert(i_pos,1) #insert element 1 at i_pos
    lattice = np.array(latticeList).reshape(shapeTuple)
    return lattice


def nearestNeighborKernel(dim):
    """
    Generates a kernel to compute the number of nearest neighbors for
    a given lattice site. Kernel is essentially an N-D "plus sign"
    with a zero at the center
    ---Inputs---
    dim: dimensionality of the lattice (in one, two, etc. dimensions), scalar integer
    ---Outputs---
    nn_kernel: nearest neighbor kernel for dim dimensional lattice, [3]*dim integer array
    """
    nn_kernel=np.zeros([3]*dim,dtype=int) #preallocate nearest neighbor kernel
    kernelSetupList=[':']+['1']*(dim-1) #set list of indices used to create kernel
    for i_setup in range(len(kernelSetupList)):
        curSliceString='np.s_['+','.join(kernelSetupList)+']'
        curSlice=eval(curSliceString)
        nn_kernel[curSlice]=np.ones(3,dtype=int)
        kernelSetupList=np.roll(kernelSetupList,1)
    centerElementIndices=tuple([1]*dim)
    nn_kernel[centerElementIndices]=0 #set center element of kernel to zero
    return nn_kernel


# pregenerate nearest neighbor kernels (relatively expensive)
kernelNNList = [0] # dummy dim=0 case for sensible indexing of list later 
kernelNNList += [nearestNeighborKernel(dim) for dim in range(1,7)]


def getNearestNeighborEnvironment(lattice,center_indices):
    """
    gets lattice environment of site at center_indices from lattice
    NOTE: assumes periodic boundary conditions
    NOTE: only extracts values of sites that share an edge with
          site at center_indices
          for example:
              1 1 0    0 1 0
              1 X 1 -> 1 X 1
              1 0 1    0 0 0
    ---Inputs---
    center_indices : {iterable}
        indices defining site whose environment is sought
    ---Outputs---
    environment : {numpy array}
        order dim, dimension 3 array containing environment of site
        at center_indices
    """
    lattice_shape = lattice.shape
    dim = len(lattice_shape) # spatial dimension
    N_cellPerEdge = lattice_shape[0] # sites per edge of volume
    environment = np.zeros([3]*dim,dtype=int) # environment array
    for i_d in range(dim): # loop over spatial dimensions
        cur_center = center_indices[i_d]
        # create three indices
        cur_range = [0]*3
        if (cur_center == 0): #selected site on low edge
          cur_range[0] = N_cellPerEdge - 1 #wrap to high edge
        else:
            cur_range[0] = cur_center - 1
        cur_range[1] = cur_center
        if (cur_center == (N_cellPerEdge-1)): #selected site on high edge
          cur_range[2] = 0 #wrap to low edge
        else:
            cur_range[2] = cur_center + 1
        cur_lattice_indices = list(copy.deepcopy(center_indices))
        cur_lattice_indices[i_d] = cur_range
        cur_environment_indices = [1]*dim # center for all modes
        cur_environment_indices[i_d] = [0,1,2] # all indices for single mode 
        environment[tuple(cur_environment_indices)] = lattice[tuple(cur_lattice_indices)]
    return environment


def computeNumberOfInteractions(lattice):
    # A sites are represented by 0s, B sites are represented by 1s
    lattice_shape = lattice.shape
    dim = len(lattice_shape) # spatial dimension
    N_cellPerEdge = lattice_shape[0] # sites per edge of volume
    z = dim*2 # number of nearest neighbors sites for a single site
    N_cellTotal = int(np.power(N_cellPerEdge,dim))
    N_cellB = np.sum(lattice)
    N_cellA = N_cellTotal - N_cellB

    kernelNN = kernelNNList[dim] # retrieve kernel array
    # convolving the nearest neighbor kernel with 0,1 lattice then
    #   reports the number of B neighbors for each site
    numBNeighborArray = sp.ndimage.convolve(lattice,kernelNN,mode='wrap')
    # elementwise multiplying this by the original lattice reports the
    #   number of BB interactions for each site (since A sites with
    #   B neighbors are multiplied by 0
    numBBArray = lattice*numBNeighborArray
    # total number of BB interactions, don't double count
    m_BB = int(np.sum(numBBArray)/2)
    m_AB = z*N_cellB - 2*m_BB
    m_AA = int((z*N_cellA - m_AB)/2)
    # sanity checks
    #print(numBBArray)
    #print(f'm_AA : {m_AA}')
    #print(f'm_BB : {m_BB}')
    #print(f'm_AB : {m_AB}')
    #print(f'z N_A - 2*m_AA - m_AB : {z*N_cellA-2*m_AA-m_AB}')
    #print(f'z N_B - 2*m_BB - m_AB : {z*N_cellB-2*m_BB-m_AB}')
    return m_AA, m_BB, m_AB


def computeTotalEnergy(lattice,w_AA,w_BB,w_AB):
    m_AA, m_BB, m_AB = computeNumberOfInteractions(lattice)
    E_AA = m_AA*w_AA
    E_BB = m_BB*w_BB
    E_AB = m_AB*w_AB
    E_total = E_AA + E_BB + E_AB
    return  E_total


def computeNumberOfInteractionsEnvironment(environment):
    """
    compute number of AA, BB, AB interactions for the site at the
    center of the environment which is a subset of the whole lattice
    ---Inputs---
    environment : {numpy array}
        shape [3]*dim local environment of the site at the center,
        only including sites which share a side/face with said site 
    ---Outputs---
    m_<X><Y> : {float}
        number of XY interactions
        float because possibility of half interaction
    """
    dim = len(environment.shape) # spatial dimension
    z = dim*2 # number of nearest neighbors sites for a single site
    center_indices = tuple([1]*dim) # indices of central site of environment
    center_site = environment[center_indices]
    kernelNN = kernelNNList[dim] # retrieve kernel array

    # convolving the nearest neighbor kernel with 0,1 lattice reports
    #   the number of B neighbors for each site
    numBNeighborArray = sp.ndimage.convolve(environment,kernelNN,mode='wrap')
    numBNeighbor = numBNeighborArray[center_indices] # number of B neighbors of center site
    if (center_site == 0): # looking at environment of A site
        m_AB = numBNeighbor
        m_AA = z - m_AB
        m_BB = 0
    elif (center_site == 1): # looking at environment of B site
        m_BB = numBNeighbor
        m_AB = z - m_BB
        m_AA = 0
    # correct for double counting, since central atom only takes on
    # half of total interaction energy
    m_AA *= 0.5
    m_AB *= 0.5
    m_BB *= 0.5
    #print(f'm_AA, m_BB, m_AB : {(m_AA,m_BB,m_AB)}')
    return m_AA, m_BB, m_AB


def computeEnvironmentEnergy(environment,w_AA,w_BB,w_AB):
    m_AA_env, m_BB_env, m_AB_env = computeNumberOfInteractionsEnvironment(environment)
    E_AA_env = m_AA_env*w_AA
    E_BB_env = m_BB_env*w_BB
    E_AB_env = m_AB_env*w_AB
    E_env = E_AA_env + E_BB_env + E_AB_env
    return  E_env


def computeCurrentEnvironmentEnergy(lattice,center_indices_i,center_indices_j,
                                    w_AA,w_BB,w_AB):
    """
    computes the environment/local energy for two sites in a lattice
    """
    # get local environments of i, j
    environment_i = getNearestNeighborEnvironment(lattice,center_indices_i)
    computeEnvironmentEnergy
    environment_j = getNearestNeighborEnvironment(lattice,center_indices_j)

    # get local energies attributed to i, j, and local energy
    #   attributed to i and j
    E_i_cur = computeEnvironmentEnergy(environment_i,w_AA,w_BB,w_AB)
    E_j_cur = computeEnvironmentEnergy(environment_j,w_AA,w_BB,w_AB)
    E_env_cur = E_i_cur + E_j_cur
    return E_env_cur


def sitesAreAdjacent(indices_i,indices_j,max_index):
    """
    determines if two index lists correspond to those of adjacent
    elements
    """
    dim = np.array(indices_i).shape[0]
    adjacentIndices = [] # list of [mode, jj - ii = +/- 1]
    equalIndices = []
    for i_d in range(dim):
        ii_cur = indices_i[i_d]
        jj_cur = indices_j[i_d]
        if (np.abs(ii_cur-jj_cur) == 0): # indices in i_d mode same
            equalIndices.append(i_d)
        elif (np.abs(jj_cur - ii_cur) == 1): # indices in i_d mode off by 1
            adjacentIndices.append([i_d,jj_cur-ii_cur])
        elif ((ii_cur == 0) and (jj_cur == max_index-1)): # j left of i after wrapping
            adjacentIndices.append([i_d,-1])
        elif ((ii_cur == max_index-1) and (jj_cur == 0)): # j right of i after wrapping
            adjacentIndices.append([i_d,1])
    # I know you can just return the condition
    if ((len(adjacentIndices) == 1) and (len(equalIndices) == dim-1)):
        # indices are all the same except for single mode where adjacent
        areAdjacent = True
        relativeShift = np.zeros(dim,dtype=int)
        adjacentMode = adjacentIndices[0][0]
        differenceji = adjacentIndices[0][1]
        relativeShift[adjacentMode] = differenceji
    else: # indices not consistent with case of adjacent elements
        areAdjacent = False
        relativeShift = None
    return areAdjacent, relativeShift


def computeProposedEnvironmentEnergy(lattice,center_indices_i,center_indices_j,
                                     w_AA,w_BB,w_AB):
    """
    computes the environment/local energy when two sites in a lattice
    are swapped
    """
    # NOTE: ideally, you'd only call getNearestNeighborEnvironment
    #       twice total per Monte Carlo iteration (one for each site)
    #       the current implementation calls it four times
    lattice_shape = lattice.shape
    dim = len(lattice_shape)
    N_cellPerEdge = lattice_shape[0]
    # get local environments of i, j
    environment_i = getNearestNeighborEnvironment(lattice,center_indices_i)
    computeEnvironmentEnergy
    environment_j = getNearestNeighborEnvironment(lattice,center_indices_j)
    #print(f'center_indices_i : {center_indices_i}')
    #print(f'center_indices_j : {center_indices_j}')
    #print(f'original:\nenvironment_i :\n{environment_i}')
    #print(f'environment_j :\n{environment_j}')

    # swap center atoms of the environments, with special care if
    # i, j are adjacent
    areAdjacent, relativeShift = sitesAreAdjacent(center_indices_i,
                                                  center_indices_j,
                                                  N_cellPerEdge)
    # swap center sites of environments
    center_indices = tuple([1]*dim) # indices of central site of environment
    site_i_value = environment_i[center_indices]
    site_j_value = environment_j[center_indices]
    environment_i[center_indices] = site_j_value
    environment_j[center_indices] = site_i_value
    #print(f'after center swap:\nenvironment_i :\n{environment_i}')
    #print(f'environment_j :\n{environment_j}')
    if (areAdjacent): # correct if i, j adjacent
        #print('SITES ARE ADJACENT')
        #print(f'relative index shift : {relativeShift}')
        # get indices of j in i's environment and vice versa
        indices_i_in_i_env_new = tuple(np.array(center_indices) +
                                      np.array(relativeShift))
        indices_j_in_j_env_new = tuple(np.array(center_indices) -
                                      np.array(relativeShift))
        # put i, j where they should be after swap
        environment_i[indices_i_in_i_env_new] = site_i_value
        environment_j[indices_j_in_j_env_new] = site_j_value
        #print(f'after adjacent correction:\nenvironment_i :\n{environment_i}')
        #print(f'environment_j :\n{environment_j}')

    # get local energies for swapped environments
    E_i_new = computeEnvironmentEnergy(environment_i,w_AA,w_BB,w_AB)
    E_j_new = computeEnvironmentEnergy(environment_j,w_AA,w_BB,w_AB)
    E_env_new = E_i_new + E_j_new
    return E_env_new


def lattice_mixture_monte_carlo(dim,N_cellPerEdge,vFrac,beta,
                                w_AA,w_BB,w_AB,
                                max_it,S_frac_tol,
                                initialization='random',check_interval=1000,
                                print_conv='no'):
    """
    ---Inputs---
    S_frac_tol : {float}
        tolerance for error in mean, in interval (0,1)
    """
    if (vFrac > 0.5):
        print(f'ERROR: volume fraction of B (vFrac) must be smaller than 0.5, since A-B symmetry covers vFrac > 0.5')
        return
    if ((S_frac_tol < 0) or (S_frac_tol >1)):
        print(f'ERROR: S_frac_tol must be in range (0,1)')
        return

    N_cellTotal = int(np.power(N_cellPerEdge,dim))

    # generate intialization
    if (initialization == 'random'):
        lattice = createLattice(dim,N_cellPerEdge,vFrac)
    elif (initialization == 'sorted'):
        N_cellB = int(np.round(vFrac*N_cellTotal))
        N_cellA = N_cellTotal - N_cellB
        N_leftPad = int(N_cellA/2)
        N_rightPad = N_cellA - N_leftPad
        latticeFlat = np.array([0]*N_leftPad + [1]*N_cellB \
                               + [0]*N_rightPad)
        lattice = latticeFlat.reshape([N_cellPerEdge]*dim)

    # check if lattice has single microstate
    if (np.sum(lattice) in [0,1,N_cellTotal-1,N_cellTotal]):
        # all lattice microstates have equal energy, energy always
        # same, so don't run any Monte Carlo steps, just compute
        # said energy
        max_it = 0

    # energy in initial configuration
    E_0 = computeTotalEnergy(lattice,w_AA,w_BB,w_AB) 

    deltaEVec = np.array([0.0]) # array of E(cur_iteration) - E_0
    it = 0 # iteration counter
    S_frac = 100000 # percentage error in mean (dummy value)
    # get arrays of shape (N_x,dim) containing indices of site types
    A_sites = np.vstack(np.where(lattice == 0)).T
    B_sites = np.vstack(np.where(lattice == 1)).T
    N_A = A_sites.shape[0]
    N_B = B_sites.shape[0]
    N_cellTotal = N_A + N_B
    while ((it < max_it) and (S_frac > S_frac_tol)):
        # select site i (A type) and j (B type) at random, these are
        # environment centers
        # (always swapping sites of different types)
        # integer indices into A/B multi-index arrays
        site_index_A = np.random.randint(low=0,high=N_A)
        site_index_B = np.random.randint(low=0,high=N_B)
        center_indices_i = tuple(A_sites[site_index_A])
        center_indices_j = tuple(B_sites[site_index_B])
        E_env_cur = computeCurrentEnvironmentEnergy(lattice,
                                                    center_indices_i,center_indices_j,
                                                    w_AA,w_BB,w_AB)
        E_env_prop = computeProposedEnvironmentEnergy(lattice,
                                                      center_indices_i,center_indices_j,
                                                      w_AA,w_BB,w_AB)
        p_accept = min(1.0,np.exp(-beta*(E_env_prop-E_env_cur)))

        if (np.random.rand() < p_accept): # move accepted
            # swap multi-indices of sites
            A_sites[site_index_A] = np.array(center_indices_j)
            B_sites[site_index_B] = np.array(center_indices_i)
            # use updated multi-index arrays to update lattice
            lattice[tuple(A_sites[site_index_A])] = 0
            lattice[tuple(B_sites[site_index_B])] = 1

            E_new = deltaEVec[-1] - E_env_cur + E_env_prop
        else: # move not accepted
            E_new = deltaEVec[-1]
        deltaEVec = np.append(deltaEVec,E_new)
        it += 1
        if (it%check_interval == 0): # check if converged
            EVecTemp = E_0*np.ones(deltaEVec.shape) + deltaEVec
            E_mean = np.mean(EVecTemp) # mean energy
            sigma_E = np.std(EVecTemp) # standard deviation of energy
            S = sigma_E/np.sqrt(it) # standard error of mean
            S_frac = np.abs(S/E_mean) # approximate percentage error of mean
            if (print_conv == 'yes'):
                print(f'  S_frac : {S_frac}')
            
    # add deltas to original value
    EVec = E_0*np.ones(deltaEVec.shape) + deltaEVec
    A_site_values = [lattice[tuple(site)] for site in A_sites]
    B_site_values = [lattice[tuple(site)] for site in B_sites]
    return lattice, EVec

    

if (__name__ == "__main__"):
    testDim = 2
    testN_cellPerEdge = 100
    testvFrac = 0.5
    testbeta = 0.1
    testLat = createLattice(testDim,testN_cellPerEdge,testvFrac)

    latticeTest,E_testVec = mixture_monte_carlo(testDim,testN_cellPerEdge,
                                                testvFrac,testbeta,
                                                -10e-3,-30e-3,-10e-3,np.inf,5e-5,
                                                initialization='sorted')
    plt.spy(latticeTest)
    plt.show()

    plt.plot(np.arange(1,E_testVec.shape[0]+1),E_testVec,color='black')
    plt.xlabel(r'iteration count')
    plt.ylabel(r'internal energy, $U$')
    plt.show()

                                                 

