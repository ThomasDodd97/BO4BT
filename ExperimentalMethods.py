def PrepolymerMouldFillSequenceGenerator_func(alpha_arr,numeric_arr):
    """
    Function that takes a list of strings and a list of numbers as strings and creates
    a new list with a combination matrix of them both.

    This function takes:
        alpha_lis = list, set of alphabetical values
        numeric_lis = list, set of numeric values
    
    This function returns:
        alphanumeric_lis = list, 
    """
    alphanumeric_arr = []
    for alpha_str in alpha_arr:
        for numeric_str in numeric_arr:
            alphanumeric_arr.append(alpha_str + numeric_str)
    return alphanumeric_arr

def InitCsv_func(chem_lis,dims_li):
    import os
    import glob
    import pandas as pd
    from datetime import date

    # Get the local directory and set the names of data files
    base_path = str(os.getcwd())

    current_date = date.today()
    stykke1_FrontEndPath_str = "raw-data_" + str(current_date)
    stykke2_FrontEndPath_str = "raw-data_" + str(current_date)
    stykke3_FrontEndPath_str = "raw-data_" + str(current_date)
    stykke4_FrontEndPath_str = "raw-data_" + str(current_date)

    stykke1_BackEndPath_str = "_Stykke1.csv"
    stykke2_BackEndPath_str = "_Stykke2.csv"
    stykke3_BackEndPath_str = "_Stykke3.csv"
    stykke4_BackEndPath_str = "_Stykke4.csv"
    backendpath_lis = [stykke1_BackEndPath_str,stykke2_BackEndPath_str,stykke3_BackEndPath_str,stykke4_BackEndPath_str]

    stykke1_EndPath_str = stykke1_FrontEndPath_str + stykke1_BackEndPath_str
    stykke2_EndPath_str = stykke2_FrontEndPath_str + stykke2_BackEndPath_str
    stykke3_EndPath_str = stykke3_FrontEndPath_str + stykke3_BackEndPath_str
    stykke4_EndPath_str = stykke4_FrontEndPath_str + stykke4_BackEndPath_str
    endpath_lis = [stykke1_EndPath_str,stykke2_EndPath_str,stykke3_EndPath_str,stykke4_EndPath_str]

    # Stykke 1
    Stykke1Columns_lis = []
    Stykke1Columns_lis.append("mould_position")
    Stykke1Columns_lis.append("datetime")
    b_count = 1
    for dim_obj in dims_li:
        if dim_obj.name[0] == "s":
            Stykke1Columns_lis.append(dim_obj.name)
        else:
            Stykke1Columns_lis.append("b" + str(b_count))
            b_count += 1
    for chem_obj in chem_lis:
        Stykke1Columns_lis.append(chem_obj.abbrev_name + "_stoichiometry")
    Stykke1Columns_lis.append("MouldMass_g")
    Stykke1Columns_lis.append("MouldPolymerMass_g")
    Stykke1Columns_lis.append("PolymerMass_g")

    # Stykke 2
    Stykke2Columns_lis = []
    Stykke2Columns_lis.append("mould_position")
    Stykke2Columns_lis.append("datetime")
    Stykke2Columns_lis.append("MouldMass_g")
    Stykke2Columns_lis.append("MouldPolymerMass_g")
    Stykke2Columns_lis.append("PolymerMass_g")

    # Stykke 3
    Stykke3Columns_lis = []
    Stykke3Columns_lis.append("mould_position")
    Stykke3Columns_lis.append("point")
    Stykke3Columns_lis.append("datetime")
    Stykke3Columns_lis.append("exp_param_a")
    Stykke3Columns_lis.append("exp_param_b")
    Stykke3Columns_lis.append("exp_param_c")
    Stykke3Columns_lis.append("time_elapsed_hours")
    Stykke3Columns_lis.append("estimated_halflife_hours")
    Stykke3Columns_lis.append("time_elapsed_halflives")
    Stykke3Columns_lis.append("lin_param_m")
    Stykke3Columns_lis.append("lin_param_c")

    # Stykke 4
    Stykke4Columns_lis = []
    Stykke4Columns_lis.append("mould_position")
    b_count = 1
    for dim_obj in dims_li:
        if dim_obj.name[0] == "s":
            Stykke4Columns_lis.append(dim_obj.name)
        else:
            Stykke4Columns_lis.append("b" + str(b_count))
            b_count += 1
    for chem_obj in chem_lis:
        Stykke4Columns_lis.append(chem_obj.abbrev_name + "_stoichiometry")
    Stykke4Columns_lis.append("polymer_start_mass_g")
    Stykke4Columns_lis.append("polymer_end_mass_pct")
    Stykke4Columns_lis.append("delta_polymer_mass_pct")

    StykkeColumns_lis = [Stykke1Columns_lis,Stykke2Columns_lis,Stykke3Columns_lis,Stykke4Columns_lis]

    # Check if datasets have been initialised in the local directory
    for backendpath_str,endpath_str,columns_lis in zip(backendpath_lis,endpath_lis,StykkeColumns_lis):
        full_path = base_path + "/" + endpath_str
        abbreviated_path = "*" + backendpath_str
        if len(glob.glob(abbreviated_path)) > 0:
            pass
        else:
            f = open(full_path,"x")
            df = pd.DataFrame(columns=columns_lis)
            df.to_csv(full_path,index=False)

class ExperimentalMethods(object):
    def __init__(self):
        self.name = "Experimental Methods"
        self.setup = self.setup()
        # self.stykke1 = self.stykke1()
    def show(self):
        print("Outer class - simplex samplers")
        print("Name:", self.name)

    class setup():
        def __init__(self):
            self.name = "Setup"
        def show(self):
            print("Inner class - Setup")
            print("Name:", self.name)
        def moulds(self,arg1,arg2):
            fill_arr = PrepolymerMouldFillSequenceGenerator_func(arg1,arg2)
            return fill_arr
        def initcsv(self,arg1,arg2):
            InitCsv_func(arg1,arg2)
    
    # class stykke1():
    #     def __init__(self):
    #         self.name = "Stykke 1"
    #     def show(self):
    #         print("Inner class - Stykke 1")
    #         print("Name:", self.name)
    #     def run(self):
    #         stykke1_func()