def GridSampler1D_func(NoSD_sca,dims_li):
    """
    A function for generating a grid sample across a 1D space.

    This function takes:
        NoSD_sca = scalar, number of samples desired across the space.
        dim_li = list, a list of dimensions involved (i.e. 1D for 1D problem)

    This function returns:
        s_arr = array, a set of values gridded across the space
    """
    import numpy as np
    delta_sca = dims_li[0].upper - dims_li[0].lower
    segments_sca = delta_sca / (NoSD_sca - 1)
    s_li = []
    for i in range(NoSD_sca):
        if i == 0:
            s_li.append(dims_li[0].lower)
        elif i == 1:
            s_li.append(dims_li[0].lower + segments_sca)
        elif i > 1:
            s_li.append(s_li[len(s_li)-1] + segments_sca)
    s_arr = np.array(s_li)
    return s_arr

def PseudorandomSampler1D_func(NoSD_sca,dims_li):
    """
    This is a function for generating pseudorandom sampling across a
    1D parameter space.
    
    This function takes:
        NoSD_sca = scalar, number of samples desired across the space.
        dim_li = list, a list of dimensions involved (i.e. 1D for 1D problem)

    This function returns:
        s_arr = array, a set of values pseudorandomly sampled from the space
    """
    import numpy as np
    s_li = np.random.uniform(dims_li[0].lower,dims_li[0].upper,NoSD_sca)
    s_arr = np.array(s_li)
    return s_arr

def DetectNoPI1D_func(array,low_sca,high_sca):
    """
    A function for detecting the number of points inside the parameter
    space of interest. In this case, a 1D line.

    This function takes:
        array = array, a set of points distributed a 1D space
        low_sca = scalar, the lower bound for a point to be considered within the range
        high_sca = scalar, the higher bound below which a point is considered within the range

    This function returns:
        NoPI_sca = scalar, the number of points considered within the space
        PointsIn_arr = array, the set of points considered within the space
    """
    import numpy as np
    NoPI_sca = 0
    PointsIn_arr = []
    for _ in array:
        if _ > low_sca and _ < high_sca:
            NoPI_sca += 1
            PointsIn_arr.append(_)
    PointsIn_arr = np.array(PointsIn_arr)
    return NoPI_sca,PointsIn_arr

def NearestSobolBaseFinder_func(NoSD_sca):
    """
    This function takes the number of samples desired and finds the
    nearest sobol base and the number of samples that base would yield.

    This function takes:
        NoSD = scalar, the number of samples desired within the space

    This function returns:
        base_sca = scalar, the base value required by a sobol sampler
                            deemed to yield the number of samples closest
                            to the desired number of samples.
        NoS_sca = scalar, the number of samples the base will yield
    """
    import numpy as np
    # Making an array of number of samples to be taken at each base_sca value in sobol sequence
    SobolSampleSeq_li = []
    for _ in range(100):
        SobolSampleSeq_li.append(2 ** _)
    # Find the index (base_sca) of the nearest value between sobol sequence number of samples array and the
    #  scalar for double the number of samples needed in the equilateral triangle
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx,array[idx]
    # Set the base_sca value for the sobol sampler, i.e. Set base_sca (which is x) where 2 ** x
    # is equal to the number of samples the sobol sequence can deal with for an attempt
    base_sca,NoS_sca = find_nearest(SobolSampleSeq_li,NoSD_sca)
    return base_sca,NoS_sca

def SobolFixer(abstract_arr,low_sca,high_sca):
    """
    Function takes an abstract set of sobol samples between 0 and 1
    and turns them into samples between a low and a high point.

    This function takes:
        abstract_arr = array, the sobol sampled array between 0 and 1
        low_sca = scalar, the lower boundary of the dimensions parameter space
        high_sca = scalar, the upper boundary of the dimensions parameter space
    This function returns:
        fixed_arr = array, the normalised array between lower and upper bounds of parameter space
    """
    import numpy as np
    delta_sca = high_sca - low_sca
    fixed_lis = []
    for number in abstract_arr:
        fixed_lis.append(low_sca + (number * delta_sca))
    fixed_arr = np.array(fixed_lis)
    return fixed_arr

def QuasirandomSampler1D_func(NoSD_sca,dims_li):
    """
    Function that samples quasirandomly from across the 1D parameter space
    in question. The method is sobolian in nature.

    This function takes:
        NoSD_sca = scalar, number of samples desired across the space.
        dim_li = list, a list of dimensions involved (i.e. 1D for 1D problem)
    
    This function returns:
        s_arr = array, a set of values quasirandomly sampled from across the space
    """
    import numpy as np
    from scipy.stats import qmc

    delta_sca = dims_li[0].upper - dims_li[0].lower
    step_sca = delta_sca / 100

    i = 0
    NoPI = 0
    MaxAttempts = 1000
    while i < MaxAttempts:
        if i == 0:
            low_sca = dims_li[0].lower
            high_sca = dims_li[0].upper
        if (i % 250) == 0:
            low_sca = dims_li[0].lower
            high_sca = dims_li[0].upper
        base_sca,NoS_sca = NearestSobolBaseFinder_func(NoSD_sca)
        if NoS_sca < NoSD_sca:
            base_sca,NoS_sca = NearestSobolBaseFinder_func(NoSD_sca * 2)
        sampler_obj = qmc.Sobol(d=len(dims_li))
        sample_arr = sampler_obj.random_base2(m=base_sca)
        abstract_arr = sample_arr[:,0]
        fixed_arr = SobolFixer(abstract_arr,low_sca,high_sca)
        NoPI_sca,PointsIn_arr = DetectNoPI1D_func(fixed_arr,dims_li[0].lower,dims_li[0].upper)
        if NoSD_sca == NoPI_sca:
            break
        low_sca = low_sca - step_sca
        high_sca = high_sca + step_sca
        i += 1

    s1_arr = np.array(PointsIn_arr)

    return s1_arr

def GridSampler3D_func(NoS_desired_sca,dimensions_arr):
    """
    This function takes an array of dimensions and the desired number of
    samples, synthesises these pieces of information, and returns the ideal
    number of samples required for a perfect grid sampling exercise as well as
    an array of arrays pertaining to the dimensions inputted in the first place.

    This function takes:
        dimensions_arr = array, the dimensions e.g. x1, x2, x3
        NoS_desired_sca = scalar, the number of samples desired
    
    This function returns:
        s1_arr = array, a set of points grid sampled from across the space in question
        s2_arr = array, a set of points grid sampled from across the space in question
        x1_arr = array, a set of points grid sampled from across the space in question
    """
    import numpy as np
    dims_sca = len(dimensions_arr)
    GridSampleSeq_arr = []
    for _ in range(100):
        GridSampleSeq_arr.append(_ ** dims_sca)
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return GridSampleSeq_arr[idx]
    NoS_sca = find_nearest(GridSampleSeq_arr,NoS_desired_sca)
    VerticePoints_sca = int(round(NoS_sca ** (1 / dims_sca),1))
    x_vars = []
    for i,dim_xx in zip(range(dims_sca),dimensions_arr):
        VarName_str = f"x_{i}_arr"
        value_arr = []
        delta_sca = dim_xx.upper - dim_xx.lower
        step_sca = delta_sca / (dims_sca - 1)
        start_sca = dim_xx.lower
        for _ in range(VerticePoints_sca):
            if _ == 0:
                value_arr.append(start_sca)
                start_sca = start_sca + step_sca
            elif _ > 0:
                value_arr.append(start_sca)
                start_sca = start_sca + step_sca
        globals()[VarName_str] = value_arr
        x_vars.append(globals()[VarName_str])
    x_arr = np.meshgrid(*x_vars)

    x_final_arr = []
    for i,array in zip(range(dims_sca),x_arr):
        VarName_str = f"x{i+1}_arr"
        value_arr = np.array(list(np.concatenate(array,axis=0).flat))
        globals()[VarName_str] = value_arr
        x_final_arr.append(globals()[VarName_str])
    
    print(f"Samples gridded = {NoS_sca}")
    s1_arr = np.array(x_final_arr[0])
    s2_arr = np.array(x_final_arr[1])
    x1_arr = np.array(x_final_arr[2])
    b1_arr = x1_arr

    return s1_arr,s2_arr,x1_arr,b1_arr

def PseudorandomSampler3D_func(NoSD_sca,dims_li):
    """
    This function pseudorandomly samples from across a 3D space.

    This function takes:
        NoSD_sca = scalar, number of samples desired across the space.
        dim_li = list, a list of dimensions involved (i.e. 3 for 3D problem)
    
    This function returns:
        s1_arr = array, a set of points pseudorandomly sampled from across the space in question
        s2_arr = array, a set of points pseudorandomly sampled from across the space in question
        x1_arr = array, a set of points pseudorandomly sampled from across the space in question
    """
    s1_arr = PseudorandomSampler1D_func(NoSD_sca,[dims_li[0]])
    s2_arr = PseudorandomSampler1D_func(NoSD_sca,[dims_li[1]])
    x1_arr = PseudorandomSampler1D_func(NoSD_sca,[dims_li[2]])
    b1_arr = x1_arr
    return s1_arr,s2_arr,x1_arr,b1_arr

def DetectNoPI3DCube_func(fixed_s1_arr,fixed_s2_arr,fixed_x1_arr,dims_li):
    """
    A function for detecting the number of points inside the parameter
    space of interest. In this case, a 3D cube.

    This function takes:
        fixed_s1_arr = array, a set of points generated in and around the parameter space
        fixed_s2_arr = array, a set of points generated in and around the parameter space
        fixed_x1_arr = array, a set of points generated in and around the parameter space
        dims_li = list, a list of dimensions involved (i.e. 3 for 3D problem)

    This function returns:
        NoPI_sca = scalar, the number of points in the parameter space
        s1PointsIn_arr = array, the points within the parameter space
        s2PointsIn_arr = array, the points within the parameter space
        x1PointsIn_arr = array, the points within the parameter space
    """
    NoPI_sca = 0
    s1PointsIn_arr = []
    s2PointsIn_arr = []
    x1PointsIn_arr = []
    for s1_sca,s2_sca,x1_sca in zip(fixed_s1_arr,fixed_s2_arr,fixed_x1_arr):
        if s1_sca > dims_li[0].lower and s1_sca < dims_li[0].upper and s2_sca > dims_li[1].lower and s2_sca < dims_li[1].upper and x1_sca > dims_li[2].lower and x1_sca < dims_li[2].upper:
            NoPI_sca += 1
            s1PointsIn_arr.append(s1_sca)
            s2PointsIn_arr.append(s2_sca)
            x1PointsIn_arr.append(x1_sca)
    return NoPI_sca,s1PointsIn_arr,s2PointsIn_arr,x1PointsIn_arr

def QuasirandomSampler3D_func(NoSD_sca,dims_li):
    """
    This function samples quasirandomly across a 3D parameter space.

    This function takes:
        NoSD_sca = scalar, number of samples desired across the space.
        dim_li = list, a list of dimensions involved (i.e. 3 for 3D problem)
    
        This function returns:
        s1_arr = array, a set of points quasirandomly sampled from across the space in question
        s2_arr = array, a set of points quasirandomly sampled from across the space in question
        x1_arr = array, a set of points quasirandomly sampled from across the space in question
    """
    import numpy as np
    from scipy.stats import qmc

    high_s1_sca = dims_li[0].upper
    high_s2_sca = dims_li[1].upper
    high_x1_sca = dims_li[2].upper
    low_s1_sca = dims_li[0].lower
    low_s2_sca = dims_li[1].lower
    low_x1_sca = dims_li[2].lower

    delta_s1_sca = dims_li[0].upper - dims_li[0].lower
    step_s1_sca = delta_s1_sca / 100
    delta_s2_sca = dims_li[1].upper - dims_li[1].lower
    step_s2_sca = delta_s2_sca / 100
    delta_x1_sca = dims_li[2].upper - dims_li[2].lower
    step_x1_sca = delta_x1_sca / 100

    i = 0
    NoPI = 0
    MaxAttempts = 1000
    while i < MaxAttempts:
        if i == 0 or (i % 250) == 0:
            c_low_s1_sca = low_s1_sca
            c_low_s2_sca = low_s2_sca
            c_low_x1_sca = low_x1_sca
            c_high_s1_sca = high_s1_sca
            c_high_s2_sca = high_s2_sca
            c_high_x1_sca = high_x1_sca
        base_sca,NoS_sca = NearestSobolBaseFinder_func(NoSD_sca)
        if NoS_sca < NoSD_sca:
            base_sca,NoS_sca = NearestSobolBaseFinder_func(NoSD_sca * 2)
        sampler_obj = qmc.Sobol(d=len(dims_li))
        sample_arr = sampler_obj.random_base2(m=base_sca)
        abstract_s1_arr = sample_arr[:,0]
        abstract_s2_arr = sample_arr[:,1]
        abstract_x1_arr = sample_arr[:,2]
        fixed_s1_arr = SobolFixer(abstract_s1_arr,c_low_s1_sca,c_high_s1_sca)
        fixed_s2_arr = SobolFixer(abstract_s2_arr,c_low_s2_sca,c_high_s2_sca)
        fixed_x1_arr = SobolFixer(abstract_x1_arr,c_low_x1_sca,c_high_x1_sca)

        NoPI_sca,s1PointsIn_arr,s2PointsIn_arr,x1PointsIn_arr = DetectNoPI3DCube_func(fixed_s1_arr,fixed_s2_arr,fixed_x1_arr,dims_li)

        if NoSD_sca == NoPI_sca:
            break
        c_low_s1_sca = c_low_s1_sca - step_s1_sca
        c_high_s1_sca = c_high_s1_sca + step_s1_sca
        c_low_s2_sca = c_low_s2_sca - step_s2_sca
        c_high_s2_sca = c_high_s2_sca + step_s2_sca
        c_low_x1_sca = c_low_x1_sca - step_x1_sca
        c_high_x1_sca = c_high_x1_sca + step_x1_sca
        i += 1
    
    s1_arr = np.array(s1PointsIn_arr)
    s2_arr = np.array(s2PointsIn_arr)
    x1_arr = np.array(x1PointsIn_arr)
    b1_arr = np.array(x1_arr)

    return s1_arr,s2_arr,x1_arr,b1_arr

def DetectNoPI2DTri_func(x_arr,y_arr):
    """
    This function takes two arrays corresponding to coordinates within a 2D
    space, works out which points are within an equilateral triangular space
    and returns all the points which are.

    This function takes a number of variables as inputs:
        x_arr = array, the values of x for the points in 2D space
        y_arr = array, the values of y for the points in 2D space
    
    This function returns a number of variables as outputs:
        x_reduced_arr = array, the values of x for points within the equilateral triangles bounds
        y_reduced_arr = array, the values of y for points within the equilateral triangles bounds
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Laying out a function that describes simple y = mx + c linear functions
    def SI_lin_func(x, m, c):
        y = m * x + c
        return y

    # Laying out a function to calculate distances between points and lines
    def dist_func(x, y, a, b, c):
        d = ((a * x) + (b * y) + c) / (((a ** 2) + (b ** 2)) ** (1 / 2))
        return d

    # Slope Intercept Form Variables (m, c)
    L1_SIm_sca = (((1 ** 2) - (0.5 ** 2)) ** (1/2)) / 0.5
    L1_SIc_sca = 0
    L1_SIVars_arr = [L1_SIm_sca,L1_SIc_sca]
    L2_SIm_sca = -1 * ((((1 ** 2) - (0.5 ** 2)) ** (1/2)) / 0.5)
    L2_SIc_sca = (((1 ** 2) - (0.5 ** 2)) ** (1/2)) * 2
    L2_SIVars_arr = [L2_SIm_sca,L2_SIc_sca]
    L3_SIm_sca = 0
    L3_SIc_sca = 0
    L3_SIVars_arr = [L3_SIm_sca,L3_SIc_sca]
    SIVars_arr = [L1_SIVars_arr,L2_SIVars_arr,L3_SIVars_arr]

    # Standard Form Variables (a, b, c) for straight lines
    L1_Sa_sca = -1 * ((((1 ** 2) - (0.5 ** 2)) ** (1/2)) / 0.5)
    L1_Sb_sca = 1
    L1_Sc_sca = 0
    L1_SVars_arr = [L1_Sa_sca,L1_Sb_sca,L1_Sc_sca]
    L2_Sa_sca = (((1 ** 2) - (0.5 ** 2)) ** (1/2)) / 0.5
    L2_Sb_sca = 1
    L2_Sc_sca = -1 * ((((1 ** 2) - (0.5 ** 2)) ** (1/2)) * 2)
    L2_SVars_arr = [L2_Sa_sca,L2_Sb_sca,L2_Sc_sca]
    L3_Sa_sca = 0
    L3_Sb_sca = 1
    L3_Sc_sca = 0
    L3_SVars_arr = [L3_Sa_sca,L3_Sb_sca,L3_Sc_sca]
    SVars_arr = [L1_SVars_arr,L2_SVars_arr,L3_SVars_arr]

    # Setting the lines bounding the equilateral triangle space
    L1_x_arr = np.linspace(0,0.5,100)
    L2_x_arr = np.linspace(0.5,1,100)
    L3_x_arr = np.linspace(0,1,100)
    L1_y_arr = SI_lin_func(L1_x_arr,L1_SIVars_arr[0],L1_SIVars_arr[1])
    L2_y_arr = SI_lin_func(L2_x_arr,L2_SIVars_arr[0],L2_SIVars_arr[1])
    L3_y_arr = SI_lin_func(L3_x_arr,L3_SIVars_arr[0],L3_SIVars_arr[1])

    # Obtaining distances between lines and points
    L1_dist_arr = []
    L2_dist_arr = []
    L3_dist_arr = []
    dist_arr = [L1_dist_arr,L2_dist_arr,L3_dist_arr]

    for x_sca,y_sca in zip(x_arr,y_arr):
        for L_SVars_arr,L_dist_arr in zip(SVars_arr,dist_arr):
            L_dist_arr.append(round(dist_func(x_sca,y_sca,L_SVars_arr[0],L_SVars_arr[1],L_SVars_arr[2]),4))
    x_reduced_arr = []
    y_reduced_arr = []

    for L1,L2,L3,x_sca,y_sca in zip(L1_dist_arr,L2_dist_arr,L3_dist_arr,x_arr,y_arr):
        if L1 <= 0 and L2 <= 0 and L3 >= 0:
            x_reduced_arr.append(x_sca)
            y_reduced_arr.append(y_sca)

    return x_reduced_arr,y_reduced_arr

def BalancePointFinder2DTri_func(x1_arr,y1_arr):
    """
    This function take coordinates within a parameter space x1,y1
    and converts them to balancing parameters which range between 0 and 1
    but which describe how to return to the aforementioned x1,y1 values within
    an equilateral triangle parameter space.

    This function takes:
        x1_arr = array, samples' x coordinates
        y1_arr = array, samples' y coordinates
    This function returns:
        b1_arr = array, b1 values describing movement up line 1
        b2_arr = array, b2 values describing crossing movement between lines 1 and 3
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Laying out a function that describes simple y = mx + c linear functions
    def SI_lin_func(x, m, c):
        y = m * x + c
        return y

    # Laying out a function to calculate distances between points and lines
    def dist_func(x, y, a, b, c):
        d = ((a * x) + (b * y) + c) / (((a ** 2) + (b ** 2)) ** (1 / 2))
        return d

    # Slope Intercept Form Variables (m, c)
    L1_SIm_sca = (((1 ** 2) - (0.5 ** 2)) ** (1/2)) / 0.5
    L1_SIc_sca = 0
    L1_SIVars_arr = [L1_SIm_sca,L1_SIc_sca]
    L2_SIm_sca = -1 * ((((1 ** 2) - (0.5 ** 2)) ** (1/2)) / 0.5)
    L2_SIc_sca = (((1 ** 2) - (0.5 ** 2)) ** (1/2)) * 2
    L2_SIVars_arr = [L2_SIm_sca,L2_SIc_sca]
    L3_SIm_sca = 0
    L3_SIc_sca = 0
    L3_SIVars_arr = [L3_SIm_sca,L3_SIc_sca]
    SIVars_arr = [L1_SIVars_arr,L2_SIVars_arr,L3_SIVars_arr]

    # Standard Form Variables (a, b, c) for straight lines
    L1_Sa_sca = -1 * ((((1 ** 2) - (0.5 ** 2)) ** (1/2)) / 0.5)
    L1_Sb_sca = 1
    L1_Sc_sca = 0
    L1_SVars_arr = [L1_Sa_sca,L1_Sb_sca,L1_Sc_sca]
    L2_Sa_sca = (((1 ** 2) - (0.5 ** 2)) ** (1/2)) / 0.5
    L2_Sb_sca = 1
    L2_Sc_sca = -1 * ((((1 ** 2) - (0.5 ** 2)) ** (1/2)) * 2)
    L2_SVars_arr = [L2_Sa_sca,L2_Sb_sca,L2_Sc_sca]
    L3_Sa_sca = 0
    L3_Sb_sca = 1
    L3_Sc_sca = 0
    L3_SVars_arr = [L3_Sa_sca,L3_Sb_sca,L3_Sc_sca]
    SVars_arr = [L1_SVars_arr,L2_SVars_arr,L3_SVars_arr]

    # Setting the lines bounding the equilateral triangle space
    L1_x_arr = np.linspace(0,0.5,100)
    L2_x_arr = np.linspace(0.5,1,100)
    L3_x_arr = np.linspace(0,1,100)
    L1_y_arr = SI_lin_func(L1_x_arr,L1_SIVars_arr[0],L1_SIVars_arr[1])
    L2_y_arr = SI_lin_func(L2_x_arr,L2_SIVars_arr[0],L2_SIVars_arr[1])
    L3_y_arr = SI_lin_func(L3_x_arr,L3_SIVars_arr[0],L3_SIVars_arr[1])

    # Obtaining distance between points and line 1 (leftmost line of the equilateral)
    d_arr = []
    for x1_sca,y1_sca in zip(x1_arr,y1_arr):
        d_arr.append(round(abs(dist_func(x1_sca,y1_sca,L1_SVars_arr[0],L1_SVars_arr[1],L1_SVars_arr[2])),8))

    b2_int_arr = []
    for d_sca in d_arr:
        b2_int_arr.append(d_sca/np.sin((60/360)*(2*np.pi)))

    origin_distances_arr = []
    for x1_sca,y1_sca in zip(x1_arr,y1_arr):
        origin_distances_arr.append(((x1_sca**2)+(y1_sca**2))**(1/2))
    b1_arr = []

    e_arr = []
    for d_sca,od_sca in zip(d_arr,origin_distances_arr):
        e_arr.append(((od_sca**2)-(d_sca**2))**(1/2))
    
    f_arr = []
    for b2_int_sca,d_sca in zip(b2_int_arr,d_arr):
        f_arr.append(((b2_int_sca**2)-(d_sca**2))**(1/2))

    b1_arr = []
    for e_sca,f_sca in zip(e_arr,f_arr):
        b1_arr.append(e_sca+f_sca)
    
    b2_arr = []
    for b2_int_sca,b1_sca in zip(b2_int_arr,b1_arr):
        b2_arr.append(b2_int_sca/b1_sca)

    b1_arr = np.array(b1_arr)
    b2_arr = np.array(b2_arr)

    return b1_arr,b2_arr

def PseudorandomSampler5D_func(NoSD_sca,dims_li):
    """
    This function samples in a pseudorandom fashion across a 5D parameter space.

    This function takes:
        NoSD_sca = scalar, number of samples desired across the parameter space
        dim_li = list, a list of dimensions involved (i.e. 5 for 5D problem)
    
        This function returns:
        s1_arr = array, a set of points pseudorandomly sampled from across the strength parameter space in question
        s2_arr = array, a set of points pseudorandomly sampled from across the strength parameter space in question
        s3_arr = array, a set of points pseudorandomly sampled from across the strength parameter space in question
        x1_arr = array, a set of points pseudorandomly sampled from across the balancing parameter space in question
                        x coordinates
        y1_arr = array, a set of points pseudorandomly sampled from across the balancing parameter space in question
                        y coordinates
        b1_arr = array, a set of points pseudorandomly sampled from across the balancing parameter space in question
                        b1 instructions
        b2_arr = array, a set of points pseudorandomly sampled from across the balancing parameter space in question
                        b2 instructions
    """
    import numpy as np
    s1_arr = PseudorandomSampler1D_func(NoSD_sca,[dims_li[0]])
    s2_arr = PseudorandomSampler1D_func(NoSD_sca,[dims_li[1]])
    s3_arr = PseudorandomSampler1D_func(NoSD_sca,[dims_li[2]])

    step_sca = (1 - 0) / 100

    i = 0
    MaxAttempts = 100000
    while i < MaxAttempts:
        if i == 0 or (i % 250) == 0:
            c_low_x1_sca = 0
            c_high_x1_sca = 1
            c_low_y1_sca = 0
            c_high_y1_sca = 1
        abstract_x1_arr = np.random.uniform(low=c_low_x1_sca,high=c_high_x1_sca,size=(NoSD_sca*2))
        abstract_y1_arr = np.random.uniform(low=c_low_y1_sca,high=c_high_y1_sca,size=(NoSD_sca*2))
        x1_AbstractIn_arr,y1_AbstractIn_arr = DetectNoPI2DTri_func(abstract_x1_arr,abstract_y1_arr)
        NoPI_sca = len(x1_AbstractIn_arr)
        if NoSD_sca == NoPI_sca:
            break
        c_low_x1_sca = c_low_x1_sca - step_sca
        c_high_x1_sca = c_low_x1_sca + step_sca
        c_low_y1_sca = c_low_y1_sca - step_sca
        c_high_y1_sca = c_high_y1_sca + step_sca
        i += 1
    
    x1_arr = np.array(x1_AbstractIn_arr)
    y1_arr = np.array(y1_AbstractIn_arr)

    b1_arr,b2_arr = BalancePointFinder2DTri_func(x1_arr,y1_arr)

    return s1_arr,s2_arr,s3_arr,x1_arr,y1_arr,b1_arr,b2_arr

def QuasirandomSampler5D_func(NoSD_sca,dims_li):
    """
    This function samples in a quasirandom fashion across a 5D parameter space.

    This function takes:
        NoSD_sca = scalar, number of samples desired across the parameter space
        dim_li = list, a list of dimensions involved (i.e. 5 for 5D problem)
    
        This function returns:
        s1_arr = array, a set of points pseudorandomly sampled from across the strength parameter space in question
        s2_arr = array, a set of points pseudorandomly sampled from across the strength parameter space in question
        s3_arr = array, a set of points pseudorandomly sampled from across the strength parameter space in question
        x1_arr = array, a set of points pseudorandomly sampled from across the balancing parameter space in question
                        x coordinates
        y1_arr = array, a set of points pseudorandomly sampled from across the balancing parameter space in question
                        y coordinates
        b1_arr = array, a set of points pseudorandomly sampled from across the balancing parameter space in question
                        b1 instructions
        b2_arr = array, a set of points pseudorandomly sampled from across the balancing parameter space in question
                        b2 instructions
    """
    import numpy as np
    from scipy.stats import qmc

    base_sca,NoS_sca = NearestSobolBaseFinder_func(NoSD_sca * 3)

    MaxAttempts = 100000

    # Sweeping the strength parameters in the cube space
    # for the best set of lows and highs

    s1_low_successful_arr = []
    s1_high_successful_arr = []
    s2_low_successful_arr = []
    s2_high_successful_arr = []
    s3_low_successful_arr = []
    s3_high_successful_arr = []

    i = 0
    while i < MaxAttempts:
        if i == 0 or (i % 100) == 0:
            c_low_s1_sca = dims_li[0].lower
            c_high_s1_sca = dims_li[0].upper
            c_low_s2_sca = dims_li[1].lower
            c_high_s2_sca = dims_li[1].upper
            c_low_s3_sca = dims_li[2].lower
            c_high_s3_sca = dims_li[2].upper
        sampler_obj = qmc.Sobol(d=5)
        sample_arr = sampler_obj.random_base2(m=base_sca)
        abstract_s1_arr = sample_arr[:,0]
        abstract_s2_arr = sample_arr[:,1]
        abstract_s3_arr = sample_arr[:,2]
        fixed_s1_arr = SobolFixer(abstract_s1_arr,c_low_s1_sca,c_high_s1_sca)
        fixed_s2_arr = SobolFixer(abstract_s2_arr,c_low_s2_sca,c_high_s2_sca)
        fixed_s3_arr = SobolFixer(abstract_s3_arr,c_low_s3_sca,c_high_s3_sca)
        NoPI3DC_sca,s1PointsIn_arr,s2PointsIn_arr,s3PointsIn_arr = DetectNoPI3DCube_func(fixed_s1_arr,fixed_s2_arr,fixed_s3_arr,dims_li)
        if NoPI3DC_sca == NoSD_sca:
            s1_low_successful_arr.append(c_low_s1_sca)
            s1_high_successful_arr.append(c_high_s1_sca)
            s2_low_successful_arr.append(c_low_s2_sca)
            s2_high_successful_arr.append(c_high_s2_sca)
            s3_low_successful_arr.append(c_low_s3_sca)
            s3_high_successful_arr.append(c_high_s3_sca)
        if len(s1_low_successful_arr) == 25:
            break
        c_low_s1_sca = c_low_s1_sca - (abs(dims_li[0].upper - dims_li[0].lower))/100
        c_high_s1_sca = c_high_s1_sca + (abs(dims_li[0].upper - dims_li[0].lower))/100
        c_low_s2_sca = c_low_s2_sca - (abs(dims_li[1].upper - dims_li[1].lower))/100
        c_high_s2_sca = c_high_s2_sca + (abs(dims_li[1].upper - dims_li[1].lower))/100
        c_low_s3_sca = c_low_s3_sca - (abs(dims_li[2].upper - dims_li[2].lower))/100
        c_high_s3_sca = c_high_s3_sca + (abs(dims_li[2].upper - dims_li[2].lower))/100
        i += 1

    s1_low_avgsuccess_sca = np.average(np.array(s1_low_successful_arr))
    s1_high_avgsuccess_sca = np.average(np.array(s1_high_successful_arr))
    s2_low_avgsuccess_sca = np.average(np.array(s2_low_successful_arr))
    s2_high_avgsuccess_sca = np.average(np.array(s2_high_successful_arr))
    s3_low_avgsuccess_sca = np.average(np.array(s3_low_successful_arr))
    s3_high_avgsuccess_sca = np.average(np.array(s3_high_successful_arr))

    # Sweeping the balance parameters in the triangular space
    # for the best set of lows and highs

    x1_low_successful_arr = []
    x1_high_successful_arr = []
    y1_low_successful_arr = []
    y1_high_successful_arr = []

    i = 0
    while i < MaxAttempts:
        if i == 0 or (i % 100) == 0:
            c_low_x1_sca = dims_li[3].lower
            c_high_x1_sca = dims_li[3].upper
            c_low_y1_sca = dims_li[4].lower
            c_high_y1_sca = dims_li[4].upper
        sampler_obj = qmc.Sobol(d=5)
        sample_arr = sampler_obj.random_base2(m=base_sca)
        abstract_x1_arr = sample_arr[:,3]
        abstract_y1_arr = sample_arr[:,4]
        fixed_x1_arr = SobolFixer(abstract_x1_arr,c_low_x1_sca,c_high_x1_sca)
        fixed_y1_arr = SobolFixer(abstract_y1_arr,c_low_y1_sca,c_high_y1_sca)
        x1PointsIn_arr,y1PointsIn_arr = DetectNoPI2DTri_func(fixed_x1_arr,fixed_y1_arr)
        NoPI2DT_sca = len(x1PointsIn_arr)
        if NoPI2DT_sca == NoSD_sca:
            x1_low_successful_arr.append(c_low_x1_sca)
            x1_high_successful_arr.append(c_high_x1_sca)
            y1_low_successful_arr.append(c_low_y1_sca)
            y1_high_successful_arr.append(c_high_y1_sca)
        if len(x1_low_successful_arr) == 25:
            break
        c_low_x1_sca = c_low_x1_sca - (abs(dims_li[3].upper - dims_li[3].lower))/100
        c_high_x1_sca = c_high_x1_sca + (abs(dims_li[3].upper - dims_li[3].lower))/100
        c_low_y1_sca = c_low_y1_sca - (abs(dims_li[4].upper - dims_li[4].lower))/100
        c_high_y1_sca = c_high_y1_sca + (abs(dims_li[4].upper - dims_li[4].lower))/100
        i += 1

    x1_low_avgsuccess_sca = np.average(np.array(x1_low_successful_arr))
    x1_high_avgsuccess_sca = np.average(np.array(x1_high_successful_arr))
    y1_low_avgsuccess_sca = np.average(np.array(y1_low_successful_arr))
    y1_high_avgsuccess_sca = np.average(np.array(y1_high_successful_arr))

    # Leveraging the best lows and highs for both spaces to obtain a sobol
    # sampled array that suits both the cube and triangular spaces.

    i = 0
    while i < MaxAttempts:
        if i == 0 or (i % 100) == 0:
            c_low_s1_sca = s1_low_avgsuccess_sca
            c_high_s1_sca = s1_high_avgsuccess_sca
            c_low_s2_sca = s2_low_avgsuccess_sca
            c_high_s2_sca = s2_high_avgsuccess_sca
            c_low_s3_sca = s3_low_avgsuccess_sca
            c_high_s3_sca = s3_high_avgsuccess_sca
            c_low_x1_sca = x1_low_avgsuccess_sca
            c_high_x1_sca = x1_high_avgsuccess_sca
            c_low_y1_sca = y1_low_avgsuccess_sca
            c_high_y1_sca = y1_high_avgsuccess_sca
        sampler_obj = qmc.Sobol(d=5)
        sample_arr = sampler_obj.random_base2(m=base_sca)
        abstract_s1_arr = sample_arr[:,0]
        abstract_s2_arr = sample_arr[:,1]
        abstract_s3_arr = sample_arr[:,2]
        abstract_x1_arr = sample_arr[:,3]
        abstract_y1_arr = sample_arr[:,4]
        fixed_s1_arr = SobolFixer(abstract_s1_arr,c_low_s1_sca,c_high_s1_sca)
        fixed_s2_arr = SobolFixer(abstract_s2_arr,c_low_s2_sca,c_high_s2_sca)
        fixed_s3_arr = SobolFixer(abstract_s3_arr,c_low_s3_sca,c_high_s3_sca)
        fixed_x1_arr = SobolFixer(abstract_x1_arr,c_low_x1_sca,c_high_x1_sca)
        fixed_y1_arr = SobolFixer(abstract_y1_arr,c_low_y1_sca,c_high_y1_sca)
        NoPI3DC_sca,s1PointsIn_arr,s2PointsIn_arr,s3PointsIn_arr = DetectNoPI3DCube_func(fixed_s1_arr,fixed_s2_arr,fixed_s3_arr,dims_li)
        x1PointsIn_arr,y1PointsIn_arr = DetectNoPI2DTri_func(fixed_x1_arr,fixed_y1_arr)
        NoPI2DT_sca = len(x1PointsIn_arr)
        if NoPI2DT_sca == NoSD_sca and NoPI3DC_sca == NoSD_sca:
            break

    s1_arr = np.array(s1PointsIn_arr)
    s2_arr = np.array(s2PointsIn_arr)
    s3_arr = np.array(s3PointsIn_arr)
    x1_arr = np.array(x1PointsIn_arr)
    y1_arr = np.array(y1PointsIn_arr)

    b1_arr,b2_arr = BalancePointFinder2DTri_func(x1_arr,y1_arr)

    return s1_arr,s2_arr,s3_arr,x1_arr,y1_arr,b1_arr,b2_arr

def Plot1D_func(s1_arr):
    from matplotlib import pyplot as plt
    plt.close("all")
    plt.rcParams["figure.figsize"] = [12,1]
    plt.hlines(1,0,1)
    plt.eventplot(s1_arr, orientation='horizontal', colors='b')
    plt.xlabel("s1_arr")
    ax = plt.gca()
    ax.get_yaxis().set_visible(False)
    plt.show()

def Plot3D_func(s1_arr,s2_arr,b1_arr):
    from matplotlib import pyplot as plt
    plt.close("all")
    plt.rcParams["figure.figsize"] = [12,4]
    fig, ax = plt.subplots(nrows= 3, ncols=1)
    ax[0].eventplot(s1_arr, orientation='horizontal', colors='b')
    ax[0].hlines(1,0,1)
    ax[0].set_xlabel("s1")
    ax[1].eventplot(s2_arr, orientation='horizontal', colors='b')
    ax[1].hlines(1,0,1)
    ax[1].set_xlabel("s2")
    ax[2].eventplot(b1_arr, orientation='horizontal', colors='b')
    ax[2].hlines(1,0,1)
    ax[2].set_xlabel("b1")
    plt.tight_layout()
    plt.show()
    plt.close("all")

    fig=plt.figure(figsize=[12,8])
    ax=plt.axes(projection='3d')
    ax.scatter(s1_arr,s2_arr,b1_arr)
    ax.set_xlabel("s1")
    ax.set_ylabel("s2")
    ax.set_zlabel("b1")

# Plotting the 2D Equilateral Triangle Space
def SimplexTwoParameterSpacePlot2D_func():
    """
    This function is called when the user would like the sides of the equilateral
    triangle space to be plotted using plt.
    """
    import numpy as np
    from matplotlib import pyplot as plt
    # Laying out a function that describes simple y = mx + c linear functions
    def SI_lin_func(x, m, c):
        y = m * x + c
        return y

    # Slope Intercept Form Variables (m, c)
    L1_SIm_sca = (((1 ** 2) - (0.5 ** 2)) ** (1/2)) / 0.5
    L1_SIc_sca = 0
    L1_SIVars_arr = [L1_SIm_sca,L1_SIc_sca]
    L2_SIm_sca = -1 * ((((1 ** 2) - (0.5 ** 2)) ** (1/2)) / 0.5)
    L2_SIc_sca = (((1 ** 2) - (0.5 ** 2)) ** (1/2)) * 2
    L2_SIVars_arr = [L2_SIm_sca,L2_SIc_sca]
    L3_SIm_sca = 0
    L3_SIc_sca = 0
    L3_SIVars_arr = [L3_SIm_sca,L3_SIc_sca]
    SIVars_arr = [L1_SIVars_arr,L2_SIVars_arr,L3_SIVars_arr]

    # Setting the lines bounding the equilateral triangle space
    L1_x_arr = np.linspace(0,0.5,100)
    L2_x_arr = np.linspace(0.5,1,100)
    L3_x_arr = np.linspace(0,1,100)
    L1_y_arr = SI_lin_func(L1_x_arr,L1_SIVars_arr[0],L1_SIVars_arr[1])
    L2_y_arr = SI_lin_func(L2_x_arr,L2_SIVars_arr[0],L2_SIVars_arr[1])
    L3_y_arr = SI_lin_func(L3_x_arr,L3_SIVars_arr[0],L3_SIVars_arr[1])

    plt.plot(L1_x_arr,L1_y_arr,c="green",label="L1")
    plt.plot(L2_x_arr,L2_y_arr,c="red",label="L2")
    plt.plot(L3_x_arr,L3_y_arr,c="blue",label="L3")

def Plot5D_func(s1_arr,s2_arr,s3_arr,x1_arr,x2_arr,b1_arr,b2_arr):
    """
    This function is called when the user would like to plot an overview of the
    s1,s2,s3,b1,b2 variables in their respective parameter spaces.
    """
    from matplotlib import pyplot as plt
    plt.close("all")
    plt.rcParams["figure.figsize"] = [12,7]
    fig, ax = plt.subplots(nrows= 5, ncols=1)
    ax[0].eventplot(s1_arr, orientation='horizontal', colors='b')
    ax[0].hlines(1,0,1)
    ax[0].set_xlabel("s1")
    ax[1].eventplot(s2_arr, orientation='horizontal', colors='b')
    ax[1].hlines(1,0,1)
    ax[1].set_xlabel("s2")
    ax[2].eventplot(s3_arr, orientation='horizontal', colors='b')
    ax[2].hlines(1,0,1)
    ax[2].set_xlabel("s3")
    ax[3].eventplot(b1_arr, orientation='horizontal', colors='b')
    ax[3].hlines(1,0,1)
    ax[3].set_xlabel("b1")
    ax[4].eventplot(b2_arr, orientation='horizontal', colors='b')
    ax[4].hlines(1,0,1)
    ax[4].set_xlabel("b2")
    plt.tight_layout()
    plt.show()
    plt.close("all")

    fig=plt.figure(figsize=[12,8])
    ax=plt.axes(projection='3d')
    ax.scatter(s1_arr,s2_arr,s3_arr)
    ax.set_xlabel("s1")
    ax.set_ylabel("s2")
    ax.set_zlabel("s3")
    plt.show()

    plt.close("all")
    fig=plt.figure(figsize=[12,8])
    plt.xlim(0,1)
    plt.ylim(0,0.88)
    SimplexTwoParameterSpacePlot2D_func()
    plt.scatter(x1_arr,x2_arr)
    plt.xlabel("x1")
    plt.ylabel("y1")

class SimplexSampler(object):
    """
    Simplex sampler class. Each of the simplexes relate to the number of 
    balancing dimensions involved with the overall Bayesian framework.

    Zeroth Simplex
        Strength Dimensions = 1     s1
        Balancing Dimensions = 0    
            Total Dimensions = 1
    First Simplex
        Strength Dimensions = 2     s1,s2
        Balancing Dimensions = 1    x1
            Total Dimensions = 3
    Second Simplex
        Strength Dimensions = 3     s1,s2,s3
        Balancing Dimensions = 2    x1,y1
            Total Dimensions = 5
    Third Simplex
        Strength Dimensions = 4     s1,s2,s3,s4
        Balancing Dimensions = 3    x1,y1,z1
            Total Dimensions = 7
    Fourth Simplex
        Strength Dimensions = 5     s1,s2,s3,s4,s5
        Balancing Dimensions = 4    x1,y1,z1,w1
            Total Dimensions = 9
    
    Sub classes are therefore split into:
        zero    
        one     
        two     
        three   
        four    

    For each of these parameter spaces, there are multiple initial prior
    generating sampling techniques that can be deployed. Firstly, a grid
    search can be carried out where the entire area is uniformly searched
    in an entirely grid-like manner. Secondly, a gridrandom search can be
    used, where the area is split into the same grid like structure as a 
    full grid search, but random points are removed to fit the number of
    samples being taken. Thirdly, pseudorandom sampling can be carried out
    where a uniform random distribution is queried to develop a random
    set of points across the parameter space of interest. And fourthly, 
    a quasirandom search can be carried out in which a more representative
    random search is generated across the entire parameter space.

    Functions within the sub classes are therefore split into the following:
        zero
            grid            complete
            pseudorandom    complete
            quasirandom     complete
        one
            grid            complete
            pseudorandom    complete
            quasirandom     complete
        two
            pseudorandom    complete
            quasirandom     complete
        three
            pseudorandom
            quasirandom
        four
            pseudorandom
            quasirandom
    """
    def __init__(self):
        self.name = "Simplex Sampler"
        self.zero = self.SimplexZero1D()
        self.one = self.SimplexOne3D()
        self.two = self.SimplexTwo5D()
        self.three = self.SimplexThree7D()
        self.four = self.SimplexFour9D()
    def show(self):
        print("Outer class - simplex samplers")
        print("Name:", self.name)

    class SimplexZero1D():
        def __init__(self):
            self.name = "Zeroth Simplex Sampler"
        def show(self):
            print("Inner class - zeroth simplex sampler")
            print("Name:", self.name)
        def grid(self,arg1,arg2):
            s1_arr = GridSampler1D_func(arg1,arg2)
            return s1_arr
        def pseudorandom(self,arg1,arg2):
            s1_arr = PseudorandomSampler1D_func(arg1,arg2)
            return s1_arr
        def quasirandom(self,arg1,arg2):
            s1_arr = QuasirandomSampler1D_func(arg1,arg2)
            return s1_arr
        def plot(self,arg1):
            Plot1D_func(arg1)
    
    class SimplexOne3D():
        def __init__(self):
            self.name = "First Simplex Sampler"
        def show(self):
            print("Inner class - first simplex sampler")
            print("Name:", self.name)
        def grid(self,arg1,arg2):
            s1_arr,s2_arr,x1_arr,b1_arr = GridSampler3D_func(arg1,arg2)
            return s1_arr,s2_arr,x1_arr,b1_arr
        def pseudorandom(self,arg1,arg2):
            s1_arr,s2_arr,x1_arr,b1_arr = PseudorandomSampler3D_func(arg1,arg2)
            return s1_arr,s2_arr,x1_arr,b1_arr
        def quasirandom(self,arg1,arg2):
            s1_arr,s2_arr,x1_arr,b1_arr = QuasirandomSampler3D_func(arg1,arg2)
            return s1_arr,s2_arr,x1_arr,b1_arr
        def plot(self,arg1,arg2,arg3):
            Plot3D_func(arg1,arg2,arg3)

    class SimplexTwo5D():
        def __init__(self):
            self.name = "Second Simplex Sampler"
        def show(self):
            print("Inner class - second simplex sampler")
            print("Name:", self.name)
        def pseudorandom(self,arg1,arg2):
            s1_arr,s2_arr,s3_arr,x1_arr,y1_arr,b1_arr,b2_arr = PseudorandomSampler5D_func(arg1,arg2)
            return s1_arr,s2_arr,s3_arr,x1_arr,y1_arr,b1_arr,b2_arr
        def quasirandom(self,arg1,arg2):
            s1_arr,s2_arr,s3_arr,x1_arr,y1_arr,b1_arr,b2_arr = QuasirandomSampler5D_func(arg1,arg2)
            return s1_arr,s2_arr,s3_arr,x1_arr,y1_arr,b1_arr,b2_arr
        def plot(self,arg1,arg2,arg3,arg4,arg5,arg6,arg7):
            Plot5D_func(arg1,arg2,arg3,arg4,arg5,arg6,arg7)

    class SimplexThree7D():
        def __init__(self):
            self.name = "Third Simplex Sampler"
        def show(self):
            print("Inner class - third simplex sampler")
            print("Name:", self.name)

    class SimplexFour9D():
        def __init__(self):
            self.name = "Fourth Simplex Sampler"
        def show(self):
            print("Inner class - fourth simplex sampler")
            print("Name:", self.name)