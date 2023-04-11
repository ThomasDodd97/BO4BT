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
    Stykke4Columns_lis.append("StartPolymerMass_g")
    Stykke4Columns_lis.append("EndPolymerMass_pct")
    Stykke4Columns_lis.append("DeltaPolymerMass_pct")
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
        CsvPaths_lis = list, all the absolute paths for the data files
        MPs_lis = list, mould positions
        ToPP_lis = list, time at which prepolymer was added to each mould position x
        chem_lis = list, the chemicals objects that are being considered
        dims_lis = list, all the dimensional objects in a list
        bdims_lis = list, all the bayesian dimensional arrays in a list
        sr_lis = list, all the stoichiometric arrays in a list
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

def StykkeTwoRunSave_func(CsvPaths_lis):
    """
    This function is the main script for a simple program which updates the stykke-2 csv dataframe with the latest
    masses of the ice cube tray, so that changes in mass can be plotted against time since start of experiment.

    This function takes two variables:
        CsvPaths_lis = list, a set of CSV strings with absolute paths for all the datasets
    """
    # Importing the modules required
    import numpy as np
    import pandas as pd
    from datetime import datetime
    import os
    ScriptPath_str = os.getcwd()
    os.chdir("/Users/thomasdodd/Library/CloudStorage/OneDrive-MillfieldEnterprisesLimited/github/BO4BT")
    from Miscellaneous import query
    os.chdir(ScriptPath_str)

    q_obj = query()

    df2 = pd.read_csv(CsvPaths_lis[1])
    df4 = pd.read_csv(CsvPaths_lis[3])

    old_moulds = list(set(df4["mould_position"]))
    all_moulds = list(set(df2["mould_position"]))
    current_moulds = [x for x in all_moulds if x not in old_moulds]

    df1 = pd.read_csv(CsvPaths_lis[0])
    for _ in old_moulds:
        df1 = df1[df1["mould_position"].str.contains(_)==False]

    # Looping over each of the moulds to be checked on
    for _,__ in zip(df1["mould_position"],df1["MouldMass_g"]):
        # Obtain permission to gather the next point
        ans = q_obj.string(f"Mass available for silicone mould {_} and its prepolymer contents? (y/n)", "y", "n")
        if ans == "y":
            # Obtaining the user input with respect to mass of silicone ice cube tray and its contents in grams
            val = q_obj.numeric(f"\tWhat is the current mass of silicone mould {_} and its prepolymer contents? (x.xx g)")
            # Read in version so far of the dataframe
            df2 = pd.read_csv(CsvPaths_lis[1])
            # Obtaining the current unformatted time
            UFTime = datetime.now()
            # Format for the time
            TFormat = "%Y-%m-%d %H:%M:%S"
            # Formatting time
            FTime = UFTime.strftime(TFormat)
            # Placing the newfound information into the dataframe we are dealing with
            df2.loc[-1] = [_, FTime, __, val, round((val-__),2)]
            # Saving the dataframe as a CSV overwriting the original CSV read in.
            df2.to_csv(CsvPaths_lis[1], index=False)
        if ans == "n":
            print("Skipping...")

def StykkeThreeRunSave_func(CsvPaths_lis):
    import numpy as np
    import pandas as pd
    from scipy.optimize import curve_fit

    # Paths are set for each of the relevant datasets
    CSVPath1_str = CsvPaths_lis[0]
    CSVPath2_str = CsvPaths_lis[1]
    CSVPath3_str = CsvPaths_lis[2]
    CSVPath4_str = CsvPaths_lis[3]

    # Dataframes 2 and 4 are read in to find moulds that are different between them.
    df2 = pd.read_csv(CSVPath2_str)
    df4 = pd.read_csv(CSVPath4_str)
    # This mould difference tells us what we are currently interested in.
    old_moulds = list(set(df4["mould_position"]))
    all_moulds = list(set(df2["mould_position"]))
    current_moulds = [x for x in all_moulds if x not in old_moulds]
    # We then remove all rows in the dataframes pertaining to the old moulds
    df1 = pd.read_csv(CSVPath1_str)
    for _ in old_moulds:
        df1 = df1[df1["mould_position"].str.contains(_)==False]
    # We then remove all rows in the dataframes pertaining to the old moulds
    df2 = pd.read_csv(CSVPath2_str)
    for _ in old_moulds:
        df2 = df2[df2["mould_position"].str.contains(_)==False]
    # And reset the indexes
    df1 = df1.reset_index(drop=True)
    df2 = df2.reset_index(drop=True)

    # Exponential function defined
    def exp_func(x, a, b, c):
        return a * np.exp(-b * x) + c

    # Linear function defined
    def lin_func(x, m, c):
        return m * x + c

    # Format the datetime correctly.
    df2['datetime'] = pd.to_datetime(df2["datetime"], format='%Y-%m-%d %H:%M:%S')
    df2
    # Get time elapsed since start of measurements in hours
    time_hours_arr = []
    for i in df2['datetime']:
        mould_pos = np.array(df2.loc[df2['datetime'] == i, 'mould_position'])[0]
        earliest_time_idx_4_mould_pos = np.array(df2["mould_position"]).tolist().index(f"{mould_pos}")
        time_hours_arr.append((i - df2.datetime[earliest_time_idx_4_mould_pos]).total_seconds() / 3600)
    df2['time_hours'] = time_hours_arr

    # Get mass in % of original mass
    polymer_mass_pct_arr = []
    for timestamp,mass in zip(df2['datetime'],df2['PolymerMass_g']):
        mould_pos = np.array(df2.loc[df2['datetime'] == timestamp, 'mould_position'])[0]
        earliest_time_idx_4_mould_pos = np.array(df2["mould_position"]).tolist().index(f"{mould_pos}")
        polymer_mass_pct_arr.append((mass / df2.PolymerMass_g[earliest_time_idx_4_mould_pos]) * 100)
    df2['polymer_mass_pct'] = polymer_mass_pct_arr

    # Read in version so far of the stykke3 dataframe
    df3 = pd.read_csv(CSVPath3_str)
    # Drop all data and leave column headings
    df3.drop(df3.index, inplace=True)
    # Saving the dataframe as a CSV overwriting the original CSV read in.
    df3.to_csv(CSVPath3_str, index=False)

    # Iterate through the mould positions
    for current_mould in df1['mould_position']:
        # X and Y arrays for time and mass of polymer in the mould defined
        x_full_arr = np.array(df2.loc[df2['mould_position'] == current_mould, 'time_hours'])
        y_full_arr = np.array(df2.loc[df2['mould_position'] == current_mould, 'polymer_mass_pct'])
        t_full_arr = np.array(df2.loc[df2['mould_position'] == current_mould, 'datetime'])
    
        # Empty arrays defined and to be filled with progressively more of the full datasets above as a part of the below loop.
        x_arr = []
        y_arr = []
    
        # Iterate through the cumulatively growing arrays considering different exponentials and develop df3
        for count,(_,__,___) in enumerate(zip(x_full_arr,y_full_arr,t_full_arr)):
            x_arr.append(_)
            y_arr.append(__)
            if count >= 2:
                # Fit the exponential function to the data, a guess of parameters is made.
                popt_exp, pcov_exp = curve_fit(exp_func, x_arr, y_arr, p0=[y_arr[0],np.log(2)/(x_arr[0]+x_arr[len(x_arr)-1]/2),y_arr[0]-1],maxfev=10000)
                # Create the x's for the last 24 hours
                x_last24exp_arr = np.linspace(x_arr[len(x_arr)-1]-24,(x_arr[len(x_arr)-1]),10)
                # Estimate the last 24 hours of data using the exponential fitted
                y_next24exp_arr = exp_func(x_last24exp_arr,popt_exp[0],popt_exp[1],popt_exp[2])
                # Fit a linear function to the precast, m (% hr^-1) can then be obtained
                popt_lin, pcov_lin = curve_fit(lin_func, x_last24exp_arr, y_next24exp_arr, p0=[((y_next24exp_arr[len(y_next24exp_arr)-1] - y_next24exp_arr[0])/(x_last24exp_arr[len(x_last24exp_arr)-1] - x_last24exp_arr[0])),80],maxfev=100000)
                # Read in version so far of the dataframe
                df3 = pd.read_csv(CSVPath3_str)
                # Placing the newfound information into the dataframe we are dealing with
                df3.loc[-1] = [current_mould,count+1,___,popt_exp[0],popt_exp[1],popt_exp[2],(round(_,2)),round(popt_lin[0],4),round(popt_lin[1],4)]
                # Saving the dataframe as a CSV overwriting the original CSV read in.
                df3.to_csv(CSVPath3_str, index=False)

def PlotDashboard_func(CsvPaths_lis,width_sca,height_sca):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit

    # Paths are set for each of the relevant datasets
    CSVPath1_str = CsvPaths_lis[0]
    CSVPath2_str = CsvPaths_lis[1]
    CSVPath3_str = CsvPaths_lis[2]
    CSVPath4_str = CsvPaths_lis[3]

    # Dataframes 2 and 4 are read in to find moulds that are different between them.
    df2 = pd.read_csv(CSVPath2_str)
    df4 = pd.read_csv(CSVPath4_str)
    # This mould difference tells us what we are currently interested in.
    old_moulds = list(set(df4["mould_position"]))
    all_moulds = list(set(df2["mould_position"]))
    current_moulds = [x for x in all_moulds if x not in old_moulds]
    # We then remove all rows in the dataframes pertaining to the old moulds
    df1 = pd.read_csv(CSVPath1_str)
    for _ in old_moulds:
        df1 = df1[df1["mould_position"].str.contains(_)==False]
    # We then remove all rows in the dataframes pertaining to the old moulds
    df2 = pd.read_csv(CSVPath2_str)
    for _ in old_moulds:
        df2 = df2[df2["mould_position"].str.contains(_)==False]
    # And reset the indexes
    df1 = df1.reset_index(drop=True)
    df2 = df2.reset_index(drop=True)

    # Read in the CSV3 as df3
    df3 = pd.read_csv(CSVPath3_str)

    # Exponential function defined
    def exp_func(x, a, b, c):
        return a * np.exp(-b * x) + c

    # Linear function defined
    def lin_func(x, m, c):
        return m * x + c

    # Format the datetime correctly.
    df2['datetime'] = pd.to_datetime(df2["datetime"], format='%Y-%m-%d %H:%M:%S')
    # Get time elapsed since start of measurements in hours
    time_hours_arr = []
    for i in df2['datetime']:
        mould_pos = np.array(df2.loc[df2['datetime'] == i, 'mould_position'])[0]
        earliest_time_idx_4_mould_pos = np.array(df2["mould_position"]).tolist().index(f"{mould_pos}")
        time_hours_arr.append((i - df2.datetime[earliest_time_idx_4_mould_pos]).total_seconds() / 3600)
    df2['time_hours'] = time_hours_arr

    # Get mass in % of original mass
    polymer_mass_pct_arr = []
    for timestamp,mass in zip(df2['datetime'],df2['PolymerMass_g']):
        mould_pos = np.array(df2.loc[df2['datetime'] == timestamp, 'mould_position'])[0]
        earliest_time_idx_4_mould_pos = np.array(df2["mould_position"]).tolist().index(f"{mould_pos}")
        polymer_mass_pct_arr.append((mass / df2.PolymerMass_g[earliest_time_idx_4_mould_pos]) * 100)
    df2['polymer_mass_pct'] = polymer_mass_pct_arr

    # Number of Moulds
    NoM = len(df1["mould_position"])
    plots_hi = int(np.ceil(NoM/2))
    plots_wi = 2

    # To make sure the plotting system still functions, we must assure a grid formation.
    if plots_hi < 2:
        plots_hi = 2

    # Mould's Plots
    plot_arr = []
    for _ in range(plots_wi):
        for __ in range(plots_hi):
            plot_arr.append((__,_))

    # Plot Setup
    fig, axs = plt.subplots(plots_hi, plots_wi, sharex=True, sharey=True, constrained_layout=True)
    fig.suptitle('Polymer Mass v Time')
    fig.set_size_inches(width_sca, height_sca)
    fig.supxlabel('Time (hours)')
    fig.supylabel('Polymer Mass (%)')

    for ax,_ in zip(plot_arr,df1["mould_position"]):
        x_arr = np.array(df2.loc[df2['mould_position'] == _, 'time_hours'])
        y_arr = np.array(df2.loc[df2['mould_position'] == _, 'polymer_mass_pct'])

        exp_a_arr = np.array(df3.loc[df3['mould_position'] == _, 'exp_param_a'])
        exp_b_arr = np.array(df3.loc[df3['mould_position'] == _, 'exp_param_b'])
        exp_c_arr = np.array(df3.loc[df3['mould_position'] == _, 'exp_param_c'])

        lin_m_arr = np.array(df3.loc[df3['mould_position'] == _, 'lin_param_m'])
        lin_c_arr = np.array(df3.loc[df3['mould_position'] == _, 'lin_param_c'])

        axs[ax].scatter(x_arr,y_arr,label=f"{_}")

        x_exp_linspaced_arr = np.linspace(0,x_arr[len(x_arr)-1],100)
        axs[ax].plot(x_exp_linspaced_arr, exp_func(x_exp_linspaced_arr, * [exp_a_arr[len(exp_a_arr)-1],exp_b_arr[len(exp_b_arr)-1],exp_c_arr[len(exp_c_arr)-1]]),label=f"$T_h$ = {round((np.log(2)/exp_b_arr[len(exp_b_arr)-1]),2)}, $T_h$'s = {round((x_arr[len(x_arr)-1])/(np.log(2)/exp_b_arr[len(exp_b_arr)-1]),2)}")

        x_lin_linspaced_arr = np.linspace(x_arr[len(x_arr)-1]-24,x_arr[len(x_arr)-1],100)
        axs[ax].plot(x_lin_linspaced_arr, lin_func(x_lin_linspaced_arr, * [lin_m_arr[len(lin_m_arr)-1],lin_c_arr[len(lin_c_arr)-1],]),label=f"$m_c$ = {lin_m_arr[len(lin_m_arr)-1]}")

        # All the hours for sample
        hours_arr = np.linspace(int(round(x_arr[0],0)),int(round(x_arr[len(x_arr)-1],0)),int(round(x_arr[len(x_arr)-1],0))+1)

        for hour in hours_arr:
            # Create the x's for the last 24 hours
            x_last24exp_arr = np.linspace(hour-24,hour,100)
            # Estimate the last 24 hours of data using the exponential fitted
            y_last24exp_arr = exp_func(x_last24exp_arr,exp_a_arr[len(exp_a_arr)-1],exp_b_arr[len(exp_b_arr)-1],exp_c_arr[len(exp_c_arr)-1])
            # Fit a linear function to the forecast, m (% hr^-1) can then be obtained
            popt_lin, pcov_lin = curve_fit(lin_func, x_last24exp_arr, y_last24exp_arr, p0=[((y_last24exp_arr[len(y_last24exp_arr)-1] - y_last24exp_arr[0])/(x_last24exp_arr[len(x_last24exp_arr)-1] - x_last24exp_arr[0])),75])
            if popt_lin[0] > -0.028:
                t_final_int = hour
                m_final_int = round(popt_lin[0],3)
                m_pct_final_int = round(exp_func(hour,exp_a_arr[len(exp_a_arr)-1],exp_b_arr[len(exp_b_arr)-1],exp_c_arr[len(exp_c_arr)-1]),2)
                axs[ax].plot(x_last24exp_arr,y_last24exp_arr,label=f"$m_f$ = {m_final_int}")
                break
            else:
                t_final_int = 0
                m_final_int = 0
                m_pct_final_int = 0

        axs[ax].axvline(t_final_int, c="green", label=f"$t_f$ = {t_final_int} hrs, $mass_f$ = {m_pct_final_int}%")

        axs[ax].legend(loc="upper right")

def PlotOverview_func(CsvPaths_lis,width_sca, height_sca):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # Paths are set for each of the relevant datasets
    CSVPath1_str = CsvPaths_lis[0]
    CSVPath2_str = CsvPaths_lis[1]
    CSVPath4_str = CsvPaths_lis[3]

    # Dataframes 2 and 4 are read in to find moulds that are different between them.
    df2 = pd.read_csv(CSVPath2_str)
    df4 = pd.read_csv(CSVPath4_str)
    # This mould difference tells us what we are currently interested in.
    old_moulds = list(set(df4["mould_position"]))
    all_moulds = list(set(df2["mould_position"]))
    current_moulds = [x for x in all_moulds if x not in old_moulds]
    # We then remove all rows in the dataframes pertaining to the old moulds
    df1 = pd.read_csv(CSVPath1_str)
    for _ in old_moulds:
        df1 = df1[df1["mould_position"].str.contains(_)==False]
    # We then remove all rows in the dataframes pertaining to the old moulds
    df2 = pd.read_csv(CSVPath2_str)
    for _ in old_moulds:
        df2 = df2[df2["mould_position"].str.contains(_)==False]
    # And reset the indexes
    df1 = df1.reset_index(drop=True)
    df2 = df2.reset_index(drop=True)

    # Format the datetime correctly.
    df2['datetime'] = pd.to_datetime(df2["datetime"], format='%Y-%m-%d %H:%M:%S')
    # Get time elapsed since start of measurements in hours
    time_hours_arr = []
    for i in df2['datetime']:
        mould_pos = np.array(df2.loc[df2['datetime'] == i, 'mould_position'])[0]
        earliest_time_idx_4_mould_pos = np.array(df2["mould_position"]).tolist().index(f"{mould_pos}")
        time_hours_arr.append((i - df2.datetime[earliest_time_idx_4_mould_pos]).total_seconds() / 3600)
    df2['time_hours'] = time_hours_arr
    # Drop any irrelevant columns
    df2.drop(["datetime", "MouldPolymerMass_g", "MouldMass_g"], axis=1, inplace=True)
    # Generate a column of polymer mass loss in % rather than g
    df2["polymer_mass_%"] = df2["PolymerMass_g"]

    fig = plt.gcf()
    fig.set_size_inches(width_sca, height_sca)

    # Loop through the fresh dataframes to generate a plot of mass loss over time in %
    for _,__ in zip(df1["mould_position"], df1["PolymerMass_g"]):
        alt_df = df2.loc[df2['mould_position'] == _]
        alt_df["polymer_mass_%"] = (alt_df["PolymerMass_g"] / __) * 100
        x_arr = alt_df["time_hours"]
        y_arr = alt_df["polymer_mass_%"]
        plt.plot(x_arr,y_arr,label=f"{_}")
        plt.legend()

    plt.xlabel("Time (hours)")
    plt.ylabel("Polymer Mass (%)")

def StykkeFourRunSave_func(chem_lis,CsvPaths_lis,dims_lis):
    """
    Function for saving the end results of an experiment thus far after stykke 3 has been run
    and it has been manually evaluated that all the moulds in this run have been completed.

    This function takes:
        chem_lis = list, chemical objects being used in this experiment
        CsvPaths_lis = list, csv path strings being used in this experiment
        dims_lis = list, bayesian dimension objects being used in this experiment

    This function does not return any variables but does locate the local
    stykke 4 csv and place the required information into it.
    """
    import pandas as pd
    import numpy as np
    from scipy.optimize import curve_fit

    # Path of the CSV file for Stykke-1
    CSVPath1_str = CsvPaths_lis[0]
    # Path of the CSV file for Stykke-2
    CSVPath2_str = CsvPaths_lis[1]
    # Path of the CSV file for Stykke-3
    CSVPath3_str = CsvPaths_lis[2]
    # Path of the CSV file for Stykke-4
    CSVPath4_str = CsvPaths_lis[3]

    # Dataframes 2 and 4 are read in to find moulds that are different between them.
    df2 = pd.read_csv(CSVPath2_str)
    df4 = pd.read_csv(CSVPath4_str)
    # This mould difference tells us what we are currently interested in.
    OldMoulds_lis = list(set(df4["mould_position"]))
    AllMoulds_lis = list(set(df2["mould_position"]))
    CurrentMoulds_lis = [x for x in AllMoulds_lis if x not in OldMoulds_lis]
    # We then remove all rows in the dataframes pertaining to the old moulds
    df1 = pd.read_csv(CSVPath1_str)
    for _ in OldMoulds_lis:
        df1 = df1[df1["mould_position"].str.contains(_)==False]
    # We then remove all rows in the dataframes pertaining to the old moulds
    df2 = pd.read_csv(CSVPath2_str)
    for _ in OldMoulds_lis:
        df2 = df2[df2["mould_position"].str.contains(_)==False]
    # And reset the indexes
    df1 = df1.reset_index(drop=True)
    df2 = df2.reset_index(drop=True)

    # Read in the CSV3 as df3
    df3 = pd.read_csv(CSVPath3_str)

    # Exponential function defined
    def exp_func(x, a, b, c):
        return a * np.exp(-b * x) + c

    # Linear function defined
    def lin_func(x, m, c):
        return m * x + c

    # Values brought together and placed in final csv
    for mould in df1["mould_position"]:
        a_arr = np.array(df3.loc[df3['mould_position'] == mould, 'exp_param_a'])
        b_arr = np.array(df3.loc[df3['mould_position'] == mould, 'exp_param_b'])
        c_arr = np.array(df3.loc[df3['mould_position'] == mould, 'exp_param_c'])
        t_elapsed_arr = np.array(df3.loc[df3['mould_position'] == mould, 'time_elapsed_hours'])
        a_sca = a_arr[len(a_arr)-1]
        b_sca = b_arr[len(b_arr)-1]
        c_sca = c_arr[len(c_arr)-1]
        t_elapsed_sca = t_elapsed_arr[len(t_elapsed_arr)-1]
        hours_arr = np.linspace(0,int(t_elapsed_sca),int(t_elapsed_sca)+1)

        for hour in hours_arr:
            # Create the x's for the last 24 hours
            x_last24exp_arr = np.linspace(hour-24,hour,100)
            # Estimate the last 24 hours of data using the exponential fitted
            y_last24exp_arr = exp_func(x_last24exp_arr,a_sca,b_sca,c_sca)
            # Fit a linear function to the forecast, m (% hr^-1) can then be obtained
            popt_lin, pcov_lin = curve_fit(lin_func, x_last24exp_arr, y_last24exp_arr, p0=[((y_last24exp_arr[len(y_last24exp_arr)-1] - y_last24exp_arr[0])/(x_last24exp_arr[len(x_last24exp_arr)-1] - x_last24exp_arr[0])),75])
            if popt_lin[0] > -0.028:
                t_final_int = hour
                m_final_int = round(popt_lin[0],3)
                m_pct_final_int = exp_func(hour,a_sca,b_sca,c_sca)
                break
            else:
                t_final_int = 0
                m_final_int = 0
                m_pct_final_int = 0

        # Getting a list of all the bayesian parameters values from df1
        BayesParam_lis = []
        for dim_obj in dims_lis:
            if dim_obj.name[0] == "s":
                header_str = f"{dim_obj.name}"
            else:
                header_str = f"b{dim_obj.name[1]}"
            VarName_str = f"{header_str}_sca"
            VarVal_sca = np.array(df1.loc[df1['mould_position'] == mould, header_str])[0]
            globals()[VarName_str] = VarVal_sca
            BayesParam_lis.append(globals()[VarName_str])

        # Getting a list of all the chemicals' stoichiometries from df1
        ChemStoich_lis = []
        for chem_obj in chem_lis:
            header_str = f"{chem_obj.abbrev_name}_stoichiometry"
            VarName_str = f"{chem_obj.abbrev_name}Stoich_sca"
            VarVal_sca = np.array(df1.loc[df1['mould_position'] == mould, header_str])[0]
            globals()[VarName_str] = VarVal_sca
            ChemStoich_lis.append(globals()[VarName_str])

        # Getting the values for moulds' mass losses etc
        StartMass_g_sca = np.array(df1.loc[df1['mould_position'] == mould, 'PolymerMass_g'])[0]
        EndMass_pct_sca = m_pct_final_int
        DeltaMassLoss_pct_sca = -1 * (100 - m_pct_final_int)

        # Compiling all the 'gotten' information from above into a final array that can be saved to df4's csv
        final_lis = []
        final_lis.append(mould)
        for BayesParam_sca in BayesParam_lis:
            final_lis.append(BayesParam_sca)
        for ChemStoich_sca in ChemStoich_lis:
            final_lis.append(ChemStoich_sca)
        final_lis.append(StartMass_g_sca)
        final_lis.append(EndMass_pct_sca)
        final_lis.append(DeltaMassLoss_pct_sca)

        # Read in version so far of the dataframe
        df4 = pd.read_csv(CSVPath4_str)
        # Placing the newfound information into the dataframe we are dealing with
        df4.loc[-1] = final_lis
        # Saving the dataframe as a CSV overwriting the original CSV read in.
        df4.to_csv(CSVPath4_str, index=False)

class ExperimentalMethods(object):
    def __init__(self):
        self.name = "Experimental Methods"
        self.setup = self.setup()
        self.stykke1 = self.stykke1()
        self.stykke2 = self.stykke2()
        self.stykke3 = self.stykke3()
        self.stykke4 = self.stykke4()
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
        def run(self,arg1,arg2,arg3,arg4):
            MPs_lis,MoSMs_lis,MoSMPP_lis,MoPP_lis,ToPP_lis = StykkeOneRun_func(arg1,arg2,arg3,arg4)
            return MPs_lis,MoSMs_lis,MoSMPP_lis,MoPP_lis,ToPP_lis
        def show(self,arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9):
            StykkeOneShow_func(arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9)
        def save(self,arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9,arg10):
            StykkeOneSave_func(arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9,arg10)

    class stykke2():
        def __init__(self):
            self.name = "Stykke 2"
        def runsave(self,arg1):
            StykkeTwoRunSave_func(arg1)
    
    class stykke3():
        def __init__(self):
            self.name = "Stykke 3"
        def runsave(self,arg1):
            StykkeThreeRunSave_func(arg1)
        def plotdash(self,arg1,arg2,arg3):
            PlotDashboard_func(arg1,arg2,arg3)
        def plotoverview(self,arg1,arg2,arg3):
            PlotOverview_func(arg1,arg2,arg3)

    class stykke4():
        def __init__(self):
            self.name = "Stykke 4"
        def runsave(self,arg1,arg2,arg3):
            StykkeFourRunSave_func(arg1,arg2,arg3)