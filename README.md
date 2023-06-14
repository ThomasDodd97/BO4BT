# BO4BT
This repository complements the Bayesian Optimisation for Biodegradable Thermosets (BO4BT) project developed at The Materials Science and Metallurgy Department of the University of Cambridge. This repository is the culmination of a year's work on optimising extent of polymerisation in biodegradable polyester thermosets. The thesis developed alongside this code will be published here once complete.

├── BO4BT-Packages
│   ├── BO4BT-Package_Ax
│   │   ├── ExperimentalMethods.py
│   │   ├── Miscellaneous.py
│   │   ├── SimplexSampler.py
│   │   ├── StoichiometryConverter.py
│   │   ├── __init__.py
│   │   └── __pycache__
│   │       ├── ExperimentalMethods.cpython-310.pyc
│   │       ├── ExperimentalMethods.cpython-311.pyc
│   │       ├── Miscellaneous.cpython-310.pyc
│   │       ├── Miscellaneous.cpython-311.pyc
│   │       ├── SimplexSampler.cpython-310.pyc
│   │       ├── SimplexSampler.cpython-311.pyc
│   │       ├── StoichiometryConverter.cpython-310.pyc
│   │       ├── StoichiometryConverter.cpython-311.pyc
│   │       └── __init__.cpython-310.pyc
│   └── BO4BT-Package_scikit-optimize
│       ├── ExperimentalMethods.py
│       ├── Miscellaneous.py
│       ├── SimplexSampler.py
│       ├── StoichiometryConverter.py
│       ├── __init__.py
│       └── __pycache__
│           ├── ExperimentalMethods.cpython-310.pyc
│           ├── ExperimentalMethods.cpython-311.pyc
│           ├── Miscellaneous.cpython-310.pyc
│           ├── Miscellaneous.cpython-311.pyc
│           ├── SimplexSampler.cpython-310.pyc
│           ├── SimplexSampler.cpython-311.pyc
│           ├── StoichiometryConverter.cpython-310.pyc
│           ├── StoichiometryConverter.cpython-311.pyc
│           └── __init__.cpython-310.pyc
├── CondaEnvironments
│   ├── env_Ax.yml
│   └── env_scikit-optimize.yml
├── OptimisationExamples
│   ├── SequentialExamples
│   │   ├── OptimisationExample_Ax
│   │   │   └── PGCiIt-BOpt-9,27,1-S2B1T1
│   │   │       ├── experiment.json
│   │   │       ├── program.ipynb
│   │   │       ├── raw-data_Stykke1.csv
│   │   │       ├── raw-data_Stykke2.csv
│   │   │       ├── raw-data_Stykke3.csv
│   │   │       └── raw-data_Stykke4.csv
│   │   └── OptimisationExample_scikit-optimize
│   │       └── PGCiIt-BOpt-9,27,1-S2B1T1
│   │           ├── program.ipynb
│   │           ├── raw-data_Stykke1.csv
│   │           ├── raw-data_Stykke2.csv
│   │           ├── raw-data_Stykke3.csv
│   │           └── raw-data_Stykke4.csv
│   └── SpaceFillingExamples
│       └── QuasirandomExample
│           └── PGCiIt-QR-27-S2B1T1
│               ├── program.ipynb
│               ├── raw-data_Stykke1.csv
│               ├── raw-data_Stykke2.csv
│               ├── raw-data_Stykke3.csv
│               └── raw-data_Stykke4.csv
└── README.md

## "BO4BT-Packages" Directory
This directory contains two python packages:
* "BO4BT-Package_Ax", developed to be run alongside Ax.
* "BO4BT-Package_scikit-optimize", developed to be run alongside scikit-learn.

The BO4BT package is split in two, so that it can be used alongside two different packages which can run Bayesian optimisation. Both are opensource on GitHub- one is named [scikit-optimize](https://scikit-optimize.github.io/stable/#), and the other is called [Ax](https://github.com/facebook/Ax).

 The BO4BT package brings together four python module files which were developed for a part-computational part-laboratory based optimisation of properties in thermoset biodegradable plastics:
* "ExperimentalMethods.py", a script containing functions useful for automating practical laboratory components of the workflow, these are unified by a class called ExperimentalMethods.
* "Miscellaneous.py", a script with extra functions and classes necessary for the workflow developed.
* "SimplexSampler.py", a script containing functions for the representative sampling of parameter spaces in multiple dimensions, as well as balancing between verious parameters. All the techniques are unified by the class called SimplexSampler.
* "StoichiometryConverter.py", a script which provides interconversion functionality when predictor variables need to be converted to stoichiometries for experimental work.

## "CondaEnvironments" Directory
This directory contains two different conda environments described in a .yml format. These environments can be imported to a local version of conda, so that the example scripts being carried out in the jupyter notebooks within this repository can be run.
* "env_Ax.yml", conda environment for running Bayesian optimisation experiments using the Ax package.
* "env.scikit-optimize.yml", conda environment for running Bayesian optimisation experiments using the scikit-optimize package.

## "OptimisationExamples" Directory
This directory contains two different example case directories:
* "SequentialExamples", this directory deals with how sequential optimisation techniques (i.e. Bayesian optimisation) were applied.
* "SpaceFillingExamples", this directory deals with how space-filling techniqes (e.g. Sobol sampling) were applied.

The split between space-filling and sequential is important, as both are necessary for successful sequential optimisation, but are implemented in seperate scripts.

Within the "SequentialExamples" subdirectory are two more directories which relate to examples relevantly applied to either the Ax or scikit-optimize versions of the BO4BT package:
* "OptimisationExample_Ax", example of Ax version of BO4BT package being applied to a Bayesian optimisation procedure.
* "OptimisationExample_scikit-optimize", example of scikit-optimize version of BO4BT package being applied to a Bayesian optimisation procedure.

In these respective subdirectories are directories describing the form of experiment engaged with:
* "PGCiIt-BOpt-9,27,1-S2B1T1", An optimisation experiment for poly(glycerol citrate itaconate) (P(GCiIt)) with a Bayesian optimisation format (BOpt), where the sequential sampling procedure begins at sample 9 and ends at sample 27 with 1 sample used per iteration (9,27,1), there are two strength predictor variables and one balancing predictor variable being optimised in regard to a single target variable (S2B1T1).

When one of these subdirectories are opened files governing the sequential optimisations are apparent:
* "experiment.json", a json file containing the "raw-data_Stykke4.csv" dataset and all information required by the Ax optimisation package to carry out the next round of sequential optimisation. Naturally only used  in the Ax version of this directory.
* "program.ipynb", a jupyter notebook that presents how a BO4BT package is used to carry out a round of optimisaton. It provides a means by which the next sample can be obtained from the Bayesian optimisation procedure, an automated mechanism for the calculation of sample stoichiometries from predictor variables, an automated sample monitroing system to keep track of reaction progress in samples, and a method of logging the characterised extents of polymerisation at the end of a round of optimisation.
* "raw-data_Stykke1.csv", A csv keeping track of current sampled being prepared.
* "raw-data_Stykke2.csv", A csv keeping track of current samples progress in terms of their polycondensation.
* "raw-data_Stykke3.csv", A csv for the calculation and visualisation of reaction progress.
* "raw-data_Stykke4.csv", A csv for the organising of all predictor and target variable values obtained from initial space-filling and following sequential optimisation.

Alternative to the directories and files associated with the "SequentialExamples" directory, there also exists an example of space-filling sampling using the BO4BT package, this example has the subdirectory "QuasirandomExample" since it shows off a quasirandom space-filling sampling technique called Sobol sampling, and it's subdirectory is called:
* "PGCiIt-QR-27-S2B1T1", An optimisation experiment for poly(glycerol citrate itaconate) (P(GCiIt)) with a Space-filling Quasirandom (QR) Sampling format based on the Sobol sampling technique, 27 samples are generated across the parameter space (27), and there are two strength predictor variables and one balancing predictor variable being optimised in regard to a single target variable (S2B1T1).
