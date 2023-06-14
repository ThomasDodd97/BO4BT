def OneDimensionStoichConv_func(chem_lis,bdims_lis):
    """
    This function takes a 1D problem and returns the stoichiometric values for
    each of the 1 underlying variable chemical and the single non-variable chemical.

    This function takes:
        chem_lis = list, a set of chemicals to be considered, with the chemical 
                    with the set stoichiometry of 1 as the first object passed.
        bdims_lis = list, a set of 'bayesian' dimensions (i.e. set between 0 and 1)
                    that describe the strength and balance parameters of the problem
    
    This function returns:
        a_arr = array, stoichiometry of 1st chemical from the chem_lis
        b_arr = array, stoichiometry of 2nd chemical from the chem_lis
    """
    import numpy as np
    a_stoich_lis = []
    b_stoich_lis = []
    stoich_lis = [a_stoich_lis,b_stoich_lis]
    for s1_sca in bdims_lis[0]:
        for chemical_obj,stoich in zip(chem_lis,stoich_lis):
            delta_sca = chemical_obj.high - chemical_obj.low
            stoich.append((delta_sca * s1_sca) + chemical_obj.low)
    a_arr = np.array(a_stoich_lis)
    b_arr = np.array(b_stoich_lis)
    return a_arr,b_arr

def ThreeDimensionStoichConv_func(chem_lis,bdims_lis):
    """
    This function takes a 3D problem and returns the stoichiometric values for
    each of the 2 underlying variable chemicals and the single non-variable chemical.

    This function takes:
        chem_lis = list, a set of chemicals to be considered, with the chemical 
                    with the set stoichiometry of 1 as the first object passed.
        bdims_lis = list, a set of 'bayesian' dimensions (i.e. set between 0 and 1)
                    that describe the strength and balance parameters of the problem
    
    This function returns:
        a_arr = array, stoichiometry of 1st chemical from the chem_lis
        b_arr = array, stoichiometry of 2nd chemical from the chem_lis
        c_arr = array, stoichiometry of 3rd chemical from the chem_lis
    """
    import numpy as np
    a_stoich_lis = np.ones(len(bdims_lis[0]))
    b_stoich_lis = []
    c_stoich_lis = []
    for s1_sca,b1_sca in zip(bdims_lis[0],bdims_lis[2]):
        delta_sca = chem_lis[1].high - chem_lis[1].low
        b_stoich_lis.append(((delta_sca * s1_sca) + chem_lis[1].low) * (1 - b1_sca))
    for s2_sca,b1_sca in zip(bdims_lis[1],bdims_lis[2]):
        delta_sca = chem_lis[2].high - chem_lis[2].low
        c_stoich_lis.append(((delta_sca * s2_sca) + chem_lis[2].low) * (b1_sca))
    a_arr = a_stoich_lis
    b_arr = np.array(b_stoich_lis)
    c_arr = np.array(c_stoich_lis)
    return a_arr,b_arr,c_arr

def FiveDimensionStoichConv_func(chem_lis,bdims_lis):
    """
    This function takes a 5D problem and returns the stoichiometric values for
    each of the 3 underlying variable chemicals and the single non-variable chemical.

    This function takes:
        chem_lis = list, a set of chemicals to be considered, with the chemical 
                    with the set stoichiometry of 1 as the first object passed.
        bdims_lis = list, a set of 'bayesian' dimensions (i.e. set between 0 and 1)
                    that describe the strength and balance parameters of the problem
    
    This function returns:
        a_arr = array, stoichiometry of 1st chemical from the chem_lis
        b_arr = array, stoichiometry of 2nd chemical from the chem_lis
        c_arr = array, stoichiometry of 3rd chemical from the chem_lis
        d_arr = array, stoichiometry of 4th chemical from the chem_lis
    """
    import numpy as np
    # Generate original x y coordinates from b1 and b2
    c_x_lis = []
    c_y_lis = []
    b2_actual_lis = []
    for b1_sca,b2_sca in zip(bdims_lis[3],bdims_lis[4]):
        c_x_lis.append((np.cos((60/360)*(2*np.pi))) * b1_sca)
        c_y_lis.append((np.sin((60/360)*(2*np.pi))) * b1_sca)
        b2_actual_lis.append(b2_sca * b1_sca)
    f_x_lis = []
    f_y_lis = []
    for c_x_sca,c_y_sca,b2_actual_sca in zip(c_x_lis,c_y_lis,b2_actual_lis):
        f_x_lis.append(c_x_sca + (np.sin((30/360)*(2*np.pi)) * b2_actual_sca))
        f_y_lis.append(c_y_sca - (np.cos((30/360)*(2*np.pi)) * b2_actual_sca))

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

    # Obtaining distances between lines and points
    L1_dist_arr = []
    L2_dist_arr = []
    L3_dist_arr = []
    dist_arr = [L1_dist_arr,L2_dist_arr,L3_dist_arr]
    for x_sca,y_sca in zip(f_x_lis,f_y_lis):
        for L_SVars_arr,L_dist_arr in zip(SVars_arr,dist_arr):
            L_dist_arr.append(abs(dist_func(x_sca,y_sca,L_SVars_arr[0],L_SVars_arr[1],L_SVars_arr[2])))
    
    # Normalising the distances and generating dominance of each variable
    L_totdist_sca = (dist_arr[0])[0] + (dist_arr[1])[0] + (dist_arr[2])[0]
    V1_dom_arr = []
    V2_dom_arr = []
    V3_dom_arr = []
    dom_arr = [V1_dom_arr,V2_dom_arr,V3_dom_arr]
    for V_dom_arr,L_dist_arr in zip(dom_arr,dist_arr):
        for L_dist_sca in L_dist_arr:
            V_dom_arr.append(L_dist_sca / L_totdist_sca)

# Generating basic stoichiometries from strength parameters
    b_BasicStoich_lis = []
    c_BasicStoich_lis = []
    d_BasicStoich_lis = []
    for s1_sca in bdims_lis[0]:
        delta_sca = chem_lis[1].high - chem_lis[1].low
        b_BasicStoich_lis.append((s1_sca * delta_sca) + chem_lis[1].low)
    for s2_sca in bdims_lis[1]:
        delta_sca = chem_lis[2].high - chem_lis[2].low
        c_BasicStoich_lis.append((s2_sca * delta_sca) + chem_lis[2].low)
    for s3_sca in bdims_lis[2]:
        delta_sca = chem_lis[3].high - chem_lis[3].low
        d_BasicStoich_lis.append((s3_sca * delta_sca) + chem_lis[3].low)

# Modifying basic stoichiometries into final stoichiometries using the ultimate dominance parameters
    a_stoich_lis = np.ones(len(bdims_lis[0]))
    b_stoich_lis = []
    c_stoich_lis = []
    d_stoich_lis = []
    for basic_stoich_sca,dominance_sca in zip(b_BasicStoich_lis,V1_dom_arr):
        b_stoich_lis.append(basic_stoich_sca * dominance_sca)
    for basic_stoich_sca,dominance_sca in zip(c_BasicStoich_lis,V2_dom_arr):
        c_stoich_lis.append(basic_stoich_sca * dominance_sca)
    for basic_stoich_sca,dominance_sca in zip(d_BasicStoich_lis,V3_dom_arr):
        d_stoich_lis.append(basic_stoich_sca * dominance_sca)

    # Finally array these lists
    a_arr = a_stoich_lis
    b_arr = np.array(b_stoich_lis)
    c_arr = np.array(c_stoich_lis)
    d_arr = np.array(d_stoich_lis)

    return a_arr,b_arr,c_arr,d_arr

class StoichiometryConverter(object):
    """
    Class for conversion from sample space into target space...
    """
    def __init__(self):
        self.name = "Simplex Sampler"
        self.onedim = self.onedim()
        self.threedim = self.threedim()
        self.fivedim = self.fivedim()
    def show(self):
        print("Outer class - simplex samplers")
        print("Name:", self.name)

    class onedim():
        def __init__(self):
            self.name = "One Dimension Converter"
        def show(self):
            print("Inner class - One Dimension Converter")
            print("Name:", self.name)
        def conv(self,arg1,arg2):
            a_arr,b_arr = OneDimensionStoichConv_func(arg1,arg2)
            return a_arr,b_arr

    class threedim():
        def __init__(self):
            self.name = "Three Dimensional Converter"
        def show(self):
            print("Inner class - Three Dimensional Converter")
            print("Name:", self.name)
        def conv(self,arg1,arg2):
            a_arr,b_arr,c_arr = ThreeDimensionStoichConv_func(arg1,arg2)
            return a_arr,b_arr,c_arr

    class fivedim():
        def __init__(self):
            self.name = "Five Dimensional Converter"
        def show(self):
            print("Inner class - Five Dimensional Converter")
            print("Name:", self.name)
        def conv(self,arg1,arg2):
            a_arr,b_arr,c_arr,d_arr = FiveDimensionStoichConv_func(arg1,arg2)
            return a_arr,b_arr,c_arr,d_arr