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

def InitCsv_func(chem_lis,dims_lis):
    """
    This function takes the chemical list and the dimensions list
    and checks for data structures in the local directory. If none
    exist it creates them.

    This function takes:
        chem_lis = list, list of chemical objects being considered
        dims_liss = list, list of dimensions being used

    This function returns:
        csv_lis[0] = string, path to the four different databases for the program
        csv_lis[1] = string, path to the four different databases for the program
        csv_lis[2] = string, path to the four different databases for the program
        csv_lis[3] = string, path to the four different databases for the program
    """
    import os
    import glob
    import pandas as pd
    from datetime import date

    # Get the local directory and set the names of data files
    base_path = str(os.getcwd())
    # Make the front end of the paths
    current_date = date.today()
    stykke1_FrontEndPath_str = "raw-data_" + str(current_date)
    stykke2_FrontEndPath_str = "raw-data_" + str(current_date)
    stykke3_FrontEndPath_str = "raw-data_" + str(current_date)
    stykke4_FrontEndPath_str = "raw-data_" + str(current_date)
    # Make the back end of the paths
    stykke1_BackEndPath_str = "_Stykke1.csv"
    stykke2_BackEndPath_str = "_Stykke2.csv"
    stykke3_BackEndPath_str = "_Stykke3.csv"
    stykke4_BackEndPath_str = "_Stykke4.csv"
    backendpath_lis = [stykke1_BackEndPath_str,stykke2_BackEndPath_str,stykke3_BackEndPath_str,stykke4_BackEndPath_str]
    # Make the back desired relative paths
    stykke1_EndPath_str = stykke1_FrontEndPath_str + stykke1_BackEndPath_str
    stykke2_EndPath_str = stykke2_FrontEndPath_str + stykke2_BackEndPath_str
    stykke3_EndPath_str = stykke3_FrontEndPath_str + stykke3_BackEndPath_str
    stykke4_EndPath_str = stykke4_FrontEndPath_str + stykke4_BackEndPath_str
    endpath_lis = [stykke1_EndPath_str,stykke2_EndPath_str,stykke3_EndPath_str,stykke4_EndPath_str]

    # Stykke 1 csv headings generator
    Stykke1Columns_lis = []
    Stykke1Columns_lis.append("mould_position")
    Stykke1Columns_lis.append("datetime")
    b_count = 1
    for dim_obj in dims_lis:
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
    # Stykke 2 csv headings generator
    Stykke2Columns_lis = []
    Stykke2Columns_lis.append("mould_position")
    Stykke2Columns_lis.append("datetime")
    Stykke2Columns_lis.append("MouldMass_g")
    Stykke2Columns_lis.append("MouldPolymerMass_g")
    Stykke2Columns_lis.append("PolymerMass_g")
    # Stykke 3 csv headings generator
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
    # Stykke 4 csv headings generator
    Stykke4Columns_lis = []
    Stykke4Columns_lis.append("mould_position")
    b_count = 1
    for dim_obj in dims_lis:
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
    # Stykke columns appended together into a list
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

    # Get the absolute paths of all the data csvs
    csv_lis = []
    for count,backendpath_str in enumerate(backendpath_lis):
        name_str = f"Csv{count+1}Path_str"
        abbreviated_path = "*" + backendpath_str
        value_str = base_path + "/" + glob.glob(abbreviated_path)[0]
        globals()[name_str] = value_str
        csv_lis.append(globals()[name_str])
    return csv_lis[0],csv_lis[1],csv_lis[2],csv_lis[3]

def StykkeOneRun_func(NoSD_sca,fill_lis,sr_lis,chem_lis):
    """
    This function runs the prepolymer formulation program, it instructs a user
    how to make the different mixes using the stoichiometries it is provided
    and asks the user for information about important variables they obtain
    during the experiment.

    This function takes:
        NoSD_sca = scalar, number of samples to be taken
        fill_lis = list, the names of samples to be generated
        sr_lis = list, the stoichiometric arrays for each of the samples and each of the chemicals
        chem_lis = list, the chemicals objects that are being considered

    This function returns:
        MPs_lis = list, mould positions
        MoSMs_lis = list, mass of silicone moulds at position x
        MoSMPP_lis = list, mass of silicone mould and prepolymer contents at x
        MoPP_lis = list, mass of prepolymer at mould position x
        ToPP_lis = list, time at which prepolymer was added to each mould position x
    """
    # Import modules and classes required
    import os
    from datetime import datetime
    ScriptPath_str = os.getcwd()
    os.chdir("/Users/thomasdodd/Library/CloudStorage/OneDrive-MillfieldEnterprisesLimited/github/BO4BT")
    from Miscellaneous import query
    os.chdir(ScriptPath_str)

    # Instantiate the query class
    q_obj = query()
    # Empty array which will contain the names of the individual silicone moulds
    MPs_lis = []
    # Empty array which will contain the masses of the individual silicone moulds
    MoSMs_lis = []
    # Empty array which will contain the masses of moulds and their prepolymer contents
    MoSMPP_lis = []
    # Empty array to be filled with prepolymer masses for each of the moulds.
    MoPP_lis = []
    # Empty array for times at which prepolymer is added to each of the moulds
    ToPP_lis = []
    # Maximum number of rounds of organic acid addition to feedstock beaker envisioned.
    MaxRounds_sca = NoSD_sca
    # Current round of organic acid addition procedure
    round_sca = 0
    # Starting up the sample genesis program
    while round_sca <= MaxRounds_sca:
        UserInput_str = q_obj.string("\nPlease place command here. (n/b for next/break)", "n", "b")
        if UserInput_str == "b":
            break
        elif UserInput_str == "n":
            if round_sca == MaxRounds_sca:
                print(f"\nSample Preparation Finished")
            else:
                # Asking for the current mass of the main chemical being considered
                MPs_lis.append(fill_lis[round_sca])
                MoG_sca = q_obj.numeric(f"\tWhat is the mass of {chem_lis[0].name} in the beaker? (x.xx g)")
                # Asking for the current mass of the mould
                MoSM_sca = q_obj.numeric("\tWhat is the mass of the silicone mould? (x.xx g)")
                MoSMs_lis.append(MoSM_sca)
                # Calculating the current mixture that needs generating from stoichiometries
                chemX_lis = chem_lis[1:]
                srX_lis = sr_lis[1:]
                MoX_lis = []
                for count,(chemX_obj,srX_arr) in enumerate(zip(chemX_lis,srX_lis)):
                    name_str = f"m{count+1}_sca"
                    value_sca = (((MoG_sca/chem_lis[0].mr)/((sr_lis[0])))*(srX_arr[round_sca]))*chemX_obj.mr
                    globals()[name_str] = value_sca
                    MoX_lis.append(globals()[name_str])
                # Printing the current sample being considered
                print(f"\t\t\t\t\t\tSample {fill_lis[round_sca]}")
                # Printing the current stoichiometry required
                string = "\t\t\t\t\t"
                for count,(sr_arr,chem_obj) in enumerate(zip(sr_lis,chem_lis)):
                    if count == 0:
                        string = string + f"{round(sr_arr[round_sca],2)} {chem_obj.abbrev_name}"
                    else:
                        string = string + f" + {round(sr_arr[round_sca],2)} {chem_obj.abbrev_name}"
                print(string)
                # Printing the current mixture required
                string = "\t\t\t\t\t"
                string = string + f"{round(MoG_sca,2)}g {chem_lis[0].abbrev_name}"
                for count,(chemX_obj,MoX_lis) in enumerate(zip(chemX_lis,MoX_lis)):
                    string = string + f" + {round(MoX_lis[count],2)}g {chemX_obj.abbrev_name}"
                print(string)
                # Asking for the current mass of mould and prepolymer mix
                MoSMPP_sca = q_obj.numeric(f"\n\tWhat is the mass of mould {fill_lis[round_sca]} and its prepolymer contents? (x.xx g)")
                MoSMPP_lis.append(MoSMPP_sca)
                # Mass of prepolymer added calculated
                MoPP_sca = MoSMPP_sca - MoSM_sca
                MoPP_lis.append(MoPP_sca)
                # Obtaining the current unformatted time
                UFTime = datetime.now()
                # Format for the time
                TFormat = "%Y-%m-%d %H:%M:%S"
                # Formatting time
                FTime = UFTime.strftime(TFormat)
                # Placing time into the time array
                ToPP_lis.append(FTime)
        round_sca += 1
    return MPs_lis,MoSMs_lis,MoSMPP_lis,MoPP_lis,ToPP_lis

def StykkeOneShow_func(dims_lis,bdims_lis,sr_lis,chem_lis,MPs_arr,MoSMs_arr,MoSMPP_arr,MoPP_arr,ToPP_arr):
    """
    This function prints the results returned by the Stykke One function.

    This function takes a number of variables as inputs:
        dims_lis = list, a list of dimensionas objects being considered
        bdims_lis = list, a list of 

    This function returns no variables.
    """
    print("\n\nFinal Information-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
    print(f"Silicone Mould Positions List:\n{MPs_arr}")
    for count,(dims_obj,bdims_arr) in enumerate(zip(dims_lis,bdims_lis)):
        if dims_obj.name[0] == "s":
            print(f"{dims_obj.name} Bayes' Dimension List:\n{bdims_arr.tolist()}")
        else:
            print("b" + f"{dims_obj.name[1]} Bayes' Dimension List:\n{bdims_arr.tolist()}")
    for count,(sr_arr,chem_obj) in enumerate(zip(sr_lis,chem_lis)):
        print(f"{chem_obj.abbrev_name} Stoichiometry List:\n{sr_arr.tolist()}")
    print(f"Silicone Mould Masses List:\n{MoSMs_arr}")
    print(f"Silicone Mould & Prepolymer Masses List:\n{MoSMPP_arr}")
    print(f"Prepolymer Masses List:\n{MoPP_arr}")
    print(f"Time of Oven Insertion List:\n{ToPP_arr}")
    print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")

def StykkeOneSave_func(CsvPaths_lis,MPs_lis,ToPP_lis,chem_lis,dims_lis,bdims_lis,sr_lis,MoSMs_lis,MoSMPP_lis,MoPP_lis):
    """
    This function takes the absolute Csv paths that designate where all the
    data is being saved, and saves the latest stykke1 runs data gathered into
    their respective places.

    This function takes:
        CsvPaths_lis
        MPs_lis = list, mould positions
        ToPP_lis = list, time at which prepolymer was added to each mould position x
        chem_lis = list, the chemicals objects that are being considered
        dims_lis = list
        bdims_lis = list
        sr_lis = list
        MoSMs_lis = list, mass of silicone moulds at position x
        MoSMPP_lis = list, mass of silicone mould and prepolymer contents at x
        MoPP_lis = list, mass of prepolymer at mould position x
    """
    import pandas as pd
    CSVPath1_str = CsvPaths_lis[0]
    CSVPath2_str = CsvPaths_lis[1]
    df = pd.DataFrame()
    df["mould_position"] = MPs_lis
    df["datetime"] = ToPP_lis
    for dim_obj,bdim_arr in zip(dims_lis,bdims_lis):
        if dim_obj.name[0] == "s":
            df[dim_obj.name] = bdim_arr
        else:
            df[("b" + dim_obj.name[1])] = bdim_arr
    for chem_obj,sr_arr in zip(chem_lis,sr_lis):
        df[chem_obj.abbrev_name + "_stoichiometry"] = sr_arr
    df["MouldMass_g"] = MoSMs_lis
    df["MouldPolymerMass_g"] = MoSMPP_lis
    df["PolymerMass_g"] = MoPP_lis

    i = df.index.start
    while i < df.index.stop:
        row_lis = df.iloc[i].tolist()
        # Reading in the CSV to a pandas dataframe
        df1 = pd.read_csv(CSVPath1_str)
        # Placing the newfound information into the dataframe we are dealing with
        df1.loc[-1] = row_lis
        # Saving the dataframe as a CSV overwriting the original CSV read in.
        df1.to_csv(CSVPath1_str, index=False)
        i+=1

    for dim_obj in dims_lis:
        if dim_obj.name[0] == "s":
            df.drop([dim_obj.name], axis=1, inplace=True)
        else:
            df.drop([("b" + dim_obj.name[1])], axis=1, inplace=True)
    for chem_obj in chem_lis:
        df.drop([(chem_obj.abbrev_name + "_stoichiometry")], axis=1, inplace=True)

    i = df.index.start
    while i < df.index.stop:
        row_lis = df.iloc[i].tolist()
        # Reading in the CSV to a pandas dataframe
        df2 = pd.read_csv(CSVPath2_str)
        # Placing the newfound information into the dataframe we are dealing with
        df2.loc[-1] = row_lis
        # Saving the dataframe as a CSV overwriting the original CSV read in.
        df2.to_csv(CSVPath2_str, index=False)
        i+=1

class ExperimentalMethods(object):
    def __init__(self):
        self.name = "Experimental Methods"
        self.setup = self.setup()
        self.stykke1 = self.stykke1()
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
            Csv1Path_str,Csv2Path_str,Csv3Path_str,Csv4Path_str = InitCsv_func(arg1,arg2)
            return Csv1Path_str,Csv2Path_str,Csv3Path_str,Csv4Path_str
    class stykke1():
        def __init__(self):
            self.name = "Stykke 1"
        def show(self):
            print("Inner class - Stykke 1")
            print("Name:", self.name)
        def run(self,arg1,arg2,arg3,arg4):
            MPs_lis,MoSMs_lis,MoSMPP_lis,MoPP_lis,ToPP_lis = StykkeOneRun_func(arg1,arg2,arg3,arg4)
            return MPs_lis,MoSMs_lis,MoSMPP_lis,MoPP_lis,ToPP_lis
        def show(self,arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9):
            StykkeOneShow_func(arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9)
        def save(self,arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9,arg10):
            StykkeOneSave_func(arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9,arg10)