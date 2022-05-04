
def computeMicroconfigurationNearestNeighborPairs(lattice,kernelNN,edgeMode):
    """
    Computes the number of nearest neighbors for an arbitrarily filled (square)
    lattice using a nearest neighbor kernel convolved with the lattice.
    The original lattice is then multiplied with the output of the convolution
    to ensure that unfilled sites with filled neighbors are mapped back to zero.
    ---Inputs---
    lattice: array which specifies the filling of the lattice, [N_cellPerEdge]*dim integer array
    dim: dimensionality of the lattice (in one, two, etc. dimensions), scalar integer
    N_cellPerEdge: number of lattice cells on the edge of the dim-D square lattice,
        scalar integer
    kernelNN: nearest neighbor kernel, [3]*dim integer array
    ---Outputs---
    N_nnp: number of nearest neighbor pairs, scalar integer
    """
    if edgeMode=='vacuum':
        nearestNeighborArray=lattice*sp.ndimage.convolve(lattice,kernelNN,mode='constant',cval=0)
    elif edgeMode=='periodic':
        nearestNeighborArray=lattice*sp.ndimage.convolve(lattice,kernelNN,mode='wrap')
    else:
        print('ERROR: invalid edge mode for scipy.ndimage.convolve, select vacuum or periodic')
    N_nnp = int(np.sum(nearestNeighborArray)/2) #compute number of nearest neighbor pairs
    return N_nnp


def computeMacroconfigurationNearestNeighborPairFrequencies(dim,N_cellPerEdge,vFrac,edgeMode):
    """
    Generates every possible lattice "microconfigurations" which satisfy the (dim,
    N_cellPerEdge,vFrac) "macroconfiguration", and computes the number of nearest
    neighbor pairs for every microconfiguration.
    ---Inputs---
    dim: dimensionality of the lattice (in one, two, etc. dimensions), scalar integer
    N_cellPerEdge: number of lattice cells on the edge of the dim-D square lattice,
        scalar integer
    vFrac: volume fraction (number of filled cells/number of cells), scalar float
    ---Outputs---
    N_nnpFrequencyArray: 1-D array which stores the number of microconfigurations
        which have each number of nearest neighbor pairs, [(N_cellPerEdge^dim)*dim]
        integer array
    """
    #Compute general metrics of "macroconfiguration"
    N_cellTotal=int(np.power(N_cellPerEdge,dim))
    N_cellB=int(np.round(vFrac*N_cellTotal))
    N_cellA=N_cellTotal-N_cellB
    N_configurations=int(sp.math.factorial(N_cellTotal)/(sp.math.factorial(N_cellB)*
        sp.math.factorial(N_cellA)))
    print(f'number of configurations : {N_configurations}')
    latticeDimensionString='['+' '.join([str(N_cellPerEdge)]*dim)+']'
    kernelNN=nearestNeighborKernel(dim) #use function to set appropriate kernel for lattice dimension

    #Generate microconfigurations and compute their number of nearest neigbor pairs,
    #storing the results in an array which counts the frequency of each number of
    #nearest neighbor pairs
    N_microconfigurationsFound=0
    trialDecimal=0
    shapeTuple=tuple([N_cellPerEdge]*dim)
    #array to store the frequency with which each number of nearest
    #neighbor pairs occurs,length of array equal to upper limit of
    #number of nearest neighbors pairs
    N_nnpFrequencyArray=np.zeros(N_cellTotal*dim+1,dtype=int) 
    microConfigurationIntegers = []
    N_microconfigurationsFound = 0
    # version generating trials be sweeping through integers (faster)
    curInteger = 0
    while (N_microconfigurationsFound < N_configurations):
        curIntegerBinary = bin(curInteger)[2:].zfill(N_cellTotal)
        latticeFlat = np.array([int(k) for k in curIntegerBinary])
        if (np.sum(latticeFlat) == N_cellB):
            trialLattice = latticeFlat.reshape(shapeTuple)
            N_nnpMicroconfiguration=computeMicroconfigurationNearestNeighborPairs(trialLattice,\
                                                                                kernelNN,\
                                                                                edgeMode)
            N_microconfigurationsFound += 1
            N_nnpFrequencyArray[N_nnpMicroconfiguration]+=1 #increase the frequency of that number of nearest neighbor pairs by one
        curInteger += 1
    # version generating trial using random realizations
    """
    while (N_microconfigurationsFound < N_configurations):
        trialLattice = createLattice(dim,N_cellPerEdge,vFrac)
        trialInteger = latticeToInteger(trialLattice)
        if (trialInteger not in microConfigurationIntegers):
            microConfigurationIntegers.append(trialInteger)
            N_microconfigurationsFound += 1
            N_nnpMicroconfiguration=computeMicroconfigurationNearestNeighborPairs(trialLattice,\
                                                                                kernelNN,\
                                                                                edgeMode)
            N_nnpFrequencyArray[N_nnpMicroconfiguration]+=1 #increase the frequency of that number of nearest neighbor pairs by one
    """

    return N_nnpFrequencyArray


def computeMacroconfigurationInternalEnergy(N_nnpFrequencyArray,C):
    """
    Computes the energy of a macroscopic configuration using the
    frequencies of it's constituent microstates
    ---Inputs---
    N_nnpFrequencyArray: 1-D array which stores the number of microconfigurations
        which have each number of nearest neighbor pairs, [(N_cellPerEdge^dim)*dim]
        integer array
    C: constant representing the product of the coldness*U(cellWidth)=
        U(cellWidth)/(k*T), or the ratio of the attractive energy to the thermal
        energy, scalar float
    ---Outputs---
    UbyU_cellWidth: ratio of the internal energy to the pairwise attraction energy
        between particles, scalar float
    """
    N_nnpArray=np.arange(len(N_nnpFrequencyArray))
    Q=np.sum(N_nnpFrequencyArray*np.exp(-C*N_nnpArray))
    # TODO: concerned above isn't actual partition function
    #       but maybe I'm using a slightly altered approach
    #       that doesn't require partition function directly
    microstateProbabilityArray=N_nnpFrequencyArray*np.exp(-C*N_nnpArray)/Q
    #print(microstateProbabilityArray)
    #TODO: sometimes the microstate probabilites are NaN,
    #      make sure to treat this properly
    UbyU_cellWidth=np.sum(microstateProbabilityArray*N_nnpArray) #internal energy divided by single pair
    #energy <E>/U(cellWidth)=U/U(cellWidth)
    return UbyU_cellWidth


def U_macro_versus_volume_fraction(dim,N_cellPerEdge,vFracArray,edgeMode,CArray):
    """
    Compute the internal energy of the specified lattice for a number of volume
    fractions.
    ---Inputs---
    dim: dimensionality of the lattice (in one, two, etc. dimensions), scalar integer
    N_cellPerEdge: number of lattice cells on the edge of the dim-D square lattice,
        scalar integer
    vFracArray: 1-D array specifying the volume fractions at which the internal
        energy should be computed, 1-D float
    CArray: 1-D array of constants representing the product coldness*U(cellWidth)=
        U(cellWidth)/(k*T), which is the ratio of the attractive energy to the thermal
        energy, 1-D float
    ---Outputs---
    UbyU_cellWidthArray: array containing the ratio of the internal energy to the
        pairwise attraction energy between particles at a range of volume fractions
        over a range of volume fraction and energetic ratios, [len(CArray) len(vFracArray)]
        float
    """
    N_C=CArray.shape[0] #number of energetic ratio constants to be sampled
    N_vFrac=vFracArray.shape[0] #number of volume fractions sampled
    UbyU_cellWidthArray=np.zeros([N_C,N_vFrac])
    for i_phi,vFrac in enumerate(vFracArray):
        N_nnpFrequencyArray=computeMacroconfigurationNearestNeighborPairFrequencies(dim,N_cellPerEdge,vFrac,edgeMode)
        for i_C,C in enumerate(CArray):
            UbyU_cellWidthArray[i_C,i_phi]=computeMacroconfigurationInternalEnergy(N_nnpFrequencyArray,C)
    return UbyU_cellWidthArray
