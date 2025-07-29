# Reproducible Package for _"A Study on Training Set Size and Model Performance in Smartphone- and Smartwatch-Based Human Activity Recognition"_

[![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/GeoTecINIT/har-performance-study/)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/GeoTecINIT/har-performance-study/HEAD)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16567319.svg)](https://doi.org/10.5281/zenodo.16567319)


This repository is the reproducibility package for the _“A Study on Training Set Size and Model Performance in Smartphone- and Smartwatch-Based Human Activity Recognition"_ journal paper, authored by 
Miguel Matey-Sanz <a href="https://orcid.org/0000-0002-1189-5079" target="_blank"><img alt="ORCID logo" src="https://info.orcid.org/wp-content/uploads/2019/11/orcid_16x16.png" width="16" height="16"/></a>, 
Joaquín Torres-Sospedra <a href="https://orcid.org/0000-0003-4338-4334" target="_blank"><img alt="ORCID logo" src="https://info.orcid.org/wp-content/uploads/2019/11/orcid_16x16.png" width="16" height="16"/></a>,
Sven Casteleyn <a href="https://orcid.org/0000-0003-0572-5716" target="_blank"><img alt="ORCID logo" src="https://info.orcid.org/wp-content/uploads/2019/11/orcid_16x16.png" width="16" height="16"/></a>
and Carlos Granell <a href="https://orcid.org/0000-0003-1004-9695" target="_blank"><img alt="ORCID logo" src="https://info.orcid.org/wp-content/uploads/2019/11/orcid_16x16.png" width="16" height="16"/></a>.

> Matey-Sanz, M., Torres-Sospedra, J., Casteleyn, S. & Granell, C. "A Study on Training Set Size and Model Performance in Smartphone- and Smartwatch-Based Human Activity Recognition". 

## Contents
The repository includes all the data, code and other resources employed throughout the develoment of the paper:

- `01_DATA`: contains the source (dataset) and intermediate (raw results of scripts) data used for obtaining the results presented in the paper.
- `02_RESULTS`: contains the final results presented in the paper, generated from analysing the raw results obtained from executing the experiments.
- `lib`: Python library contanining all the code employed to execute the experiments (`lib/pipeline/`) and analyses (`lib/analysis/`) presented in the paper.
- `*.ipynb` files: Jupyter notebooks containing the analyses whose results are presented in the paper.
- `requirements.txt`: Python libraries employed to execute experiments and analyses. All these experiments and analyses have been executed using Python 3.9.
- `Dockerfile`: file to build a Docker image with a computational environment to reproduce the experiments and analyses.


## Reproducibility
This repository contains all the required data (except the dataset, which can be downloaded from its source), code and scripts to reproduce the experiments and results presented in the paper.

### Reproducibility setup
Several options to setup a computational environment to reproduce the analyses are offered: online and locally.

#### Reproduce online with Binder
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/GeoTecINIT/har-performance-study/HEAD)

[Binder](https://mybinder.readthedocs.io/en/latest/) allows to create custom computing environments in the cloud so it can be shared to many remote users.
To open the Binder computing environment, click on the "Binder" badge above. 

> [!NOTE]
> Building the computing enviroment in Binder can be slow.


#### Reproduce locally
Install Python 3.9, download or clone the repository, open a command line in the root of the directory and install the required software executing the following command:

```bash
pip install -r requirements.txt
```

> [!TIP]
> The usage of a virtual enviroment such as the ones provided by [Conda](https://conda.io/projects/conda/en/latest/index.html) or [venv](https://docs.python.org/3/library/venv.html) are recommended.


#### Reproduce locally with Docker
Install [Docker](https://www.docker.com) for building an image based on the provided `.docker/Dockerfile` with a Jupyter environment and running a container based on the image.

Download the repository, open a command line in the root of the directory and:

1. Build the image:

```bash
docker build . --tag har-performance-study 
```

2. Run the image:

```bash
docker run -it -p 8888:8888 har-performance-study
```

3. Click on the login link (or copy and paste in the browser) shown in the console to access to a Jupyter environment.


### Reproduce the analyses
The Python scripts employed to execute the experiments described in the paper are located in `lib/pipeline/[n]_*.py`, where `n` determines the order in which the scripts must be executed. The reproduction of these scripts is not needed since their outputs are already stored in the `01_DATA/02_GRID-SEARCH/` and `01_DATA/03_MODEL-REPORTS/` directories.

> [!NOTE]
> When executing a script with a component of randomness (i.e., ML models), the obtained results might change compared with the reported ones.

> [!CAUTION]
> It is not recommended to execute these scripts, since they can run for hours, days or weeks depending on the computer's hardware.

To reproduce the outcomes presented in the paper, open the desired Jupyter Notebook (`*.ipynb`) file and execute its cells to generate reported results from the data generated in the experiments (`lib/pipeline/[n]_*.py` scripts). More concretely, the Jupyter Nobebooks are the following:

- [`0_grid-search.ipynb`](./0_grid-search.ipynb): contains the results of the Grid Search hyperparameters optimization process, i.e., results generated by executing `lib/pipeline/02_hyperparameter-optimization.py`. These results are reported in the paper's Table II (Section III-C).
- [`1_training-data.ipynb`](./1_training-data.ipynb): shows the accuracy evolution over the addition of training data in the selected models. It analyses the data generated by the `lib/pipeline/03_incremental-loso.py` script. These results are reported in paper's Figure 2 and 3 (Section IV-A).
- [`2_data-sources.ipynb`](./2_data-sources.ipynb): shows the difference in performance regarding employed datasource for each selected model and amount of training data, i.e., _which data source provides better results_. It analyses the data generated by the `lib/pipeline/03_incremental-loso.py` script. These results are reported in paper's Figure 4 (Section IV-B).
- [`3_models.ipynb`](./3_models.ipynb): shows the difference in performance regarding employed model type for each data source and amount of training data, i.e., _which model architecture provides better results_. It analyses the data generated by the `lib/pipeline/03_incremental-loso.py` script. These results are reported in paper's Figure 5 (Section IV-C).


## License

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![License: CC BY-SA 4.0](https://img.shields.io/badge/Documents%20License-CC_BY--SA_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)

All the code contained in the `.ipynb` notebooks and the `lib` folder is licensed under the [Apache License 2.0](./LICENSE).

The remaining documents included in this repository are licensed under the [Creative Commons Attribution-ShareAlike](https://creativecommons.org/licenses/by-sa/4.0/) (CC BY-SA 4.0).

## Acknowledgements

This work has been funded by the Spanish Ministry of Universities (grant FPU19/05352), by the Spanish Ministry of Science and Innovation (MCIN/AEI/10.13039/501100011033) and ``ERDF/EU'' (grants PID2020-120250RB-I00, PID2022-1404475OB-C21 and PID2022-140475OB-C22), and partially funded by the Department of Innovation, Universities, Science, and Digital Society of the Valencian Government, Spain (grant number CIAICO/2022/111.