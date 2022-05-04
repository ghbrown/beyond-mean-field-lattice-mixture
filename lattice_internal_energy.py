import copy
import numpy as np
import scipy as sp
from scipy import ndimage
import matplotlib.pyplot as plt

"""
TO DO:
-check that signs of all physics quantities are correct (very important)
-implement a different algorithm to generate all acceptable lattice fillings,
    current method wastes time by generating many unacceptable lattice fillings
    and then throwing them away (try Chase's twiddle algorithm)
-SOLUTION:implement it as a Monte Carlo method
    -given a fixed volume fraction start with a random filling
    -generate a new random filling and determine if the system will switch over
        to that filling using the ratio of the Boltzmann factors
    -once a loop determine the number of nearest neighbors of the configuration
        and add one to the tally for that number
    -once all the looping is done take the ratio of the number of sampled configurations
        which had n nearest neighbors to the total number of sample configurations
        as probability p_n
    -use the array p_n and the energy for each number of nearest neighbors to
        compute <U>/epsilon or whatever it's called
    -confirm that this solution works by comparing it with the direct method
        for low total cell number lattices (confirm stochastic approach with
        exact/deterministic approach)
STUFF TO KEEP IN MIND (PERHAPS FOR WRITING UP THIS PAPER):
    E_microstate=N_nnpMicrostate*U(cellwidth)
    Q=sum_microstates[W_microstate*exp(-Beta*N_nnpMicrostate*U(cellwidth))]
    let Beta*U(cellWidth)=C, some constant which characterizes the ratio of the
        attraction between atoms to the thermal energy
    must represent the internal energy as a ratio of the internal energy to U(cellWidth)
        since U(cellWidth) is not specified, only C
"""

"""
THINGS TO CHECK:
- does it make a difference if proposed step always swaps B site and
  A site, or if proposed move can swap B with B or A with A, etc.
- convergence of initially random filling versus initially sorted
  filling (0..011..1 or 0..011..100..0)
"""

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

    kernelNN = nearestNeighborKernel(dim) # generate kernel array
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
    kernelNN = nearestNeighborKernel(dim) # generate kernel array

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


def monte_carlo(dim,N_cellPerEdge,vFrac,w_AA,w_BB,w_AB,beta,
                max_it,tol,initialization='random'):
    N_cellTotal = int(np.power(N_cellPerEdge,dim))
    if (w_BB > w_AA):
        print(f'ERROR: w_BB must be smaller than w_AA')
        return
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
    print(lattice)

    # energy in initial configuration
    # NOTE:
    # do rest of math relative to/as delta onto this quantity
    # since all that appears are differences between energy?
    E_0 = computeTotalEnergy(lattice,w_AA,w_BB,w_AB) 

    deltaEVec = np.array([0.0]) # array of E(cur_iteration) - E_0
    it = 0 # iteration counter
    var = 1000 # dummy value
    # make tolerance relative to system size?
    while ((it < max_it) and (var > tol)):
        # select site i and j at random, these are environment center
        center_indices_i = tuple(np.random.randint(low=0,high=N_cellPerEdge,
                                                   size=dim))
        center_indices_j = tuple(np.random.randint(low=0,high=N_cellPerEdge,
                                                   size=dim))
        E_env_cur = computeCurrentEnvironmentEnergy(lattice,
                                                    center_indices_i,center_indices_j,
                                                    w_AA,w_BB,w_AB)
        E_env_prop = computeProposedEnvironmentEnergy(lattice,
                                                      center_indices_i,center_indices_j,
                                                      w_AA,w_BB,w_AB)
        p_accept = min(1.0,np.exp(-beta*(E_env_prop-E_env_cur)))
        #print(f'E_env_cur : {E_env_cur}')
        #print(f'E_env_prop : {E_env_prop}')

        if (np.random.rand() < p_accept): # move accepted
            site_i_cur = lattice[center_indices_i] # temp variables
            site_j_cur = lattice[center_indices_j]
            lattice[center_indices_i] = site_j_cur # perform swap
            lattice[center_indices_j] = site_i_cur
            E_new = deltaEVec[-1] - E_env_cur + E_env_prop
        else: # move not accepted
            E_new = deltaEVec[-1]
        deltaEVec = np.append(deltaEVec,E_new)
        it += 1
    # add deltas to original value
    print(lattice)
    print(f'E_0 : {E_0}')
    print(f'deltaEVec : {deltaEVec}')
    EVec = E_0*np.ones(deltaEVec.shape) + deltaEVec
    return lattice, EVec

    

if (__name__ == "__main__"):
    testDim = 2
    testN_cellPerEdge = 100
    testvFrac = 0.5
    testLat = createLattice(testDim,testN_cellPerEdge,testvFrac)

    latticeTest,E_testVec = monte_carlo(testDim,testN_cellPerEdge,testvFrac,
                            -1,-3,-1,1,100000,0.0,
                            initialization='random')
    plt.spy(latticeTest)
    plt.show()

    print(np.mean(E_testVec[-1000:]))

    plt.plot(np.arange(1,E_testVec.shape[0]+1),E_testVec,color='black')
    plt.xlabel(r'iteration count')
    plt.ylabel(r'energy, E')
    plt.show()

                                                 

    """
    #some main stuff goes down here
    dim=2
    N_cellPerEdge=4
    vFrac=0.25
    edgeMode='vacuum'
    C=1 #ratio of attractive to thermal energy

    #computeMacroconfigurationNearestNeighborPairFrequencies(dim,N_cellPerEdge,vFrac,edgeMode)

    N_cellTotal=int(np.power(N_cellPerEdge,dim))
    vFracArray=np.arange(N_cellTotal+1)/N_cellTotal
    #vFracArray=np.ones((1,))
    CArray=np.linspace(0.01,100,10)

    U_byU_cellWidthArray=U_macro_versus_volume_fraction(dim,N_cellPerEdge,vFracArray,edgeMode,CArray)

    for i_c,UAtFixedC in enumerate(U_byU_cellWidthArray):
        plt.plot(vFracArray,UAtFixedC,label=f'C={CArray[i_c]}')
    plt.legend()
    plt.show()
    """
