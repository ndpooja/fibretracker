FibreTracker
============

A python library to track fibre in a volume

Project Organization
--------------------

```
├── LICENSE 			<- MIT license
├── Makefile 			<- Makefile with commands like `make environment` or `make requirements`
├── README.md		<- Readme file for using the repository
├── data
│   └── examples		<- Small dataset for fibre tracking [Soon Available!!]
├── docs				<- for documentation [soon available!!]
├── fibretracker
│   ├── __init__.py			<- Makes fibretracker a Python module (call all modules)
│   ├── io				<- Scripts to load data
│   │   └── read_file.py
│   ├── models			<- Scripts for fibre detection and tracking
│   │   ├── detector.py
│   │   └── tracker.py
│   └── viz				<- Scripts to visualize the slice, detected slice, tracked fibres
│       ├── plotting.py
│       └── visualize.py
├── notebooks
│   └── Fibre_tracking.ipynb <- Notebook to do fibretracking
├── requirements.txt		<- The requirements file for reproducing the analysis environment
└── setup.py			<- makes project pip installable (pip install -e .) so fibretracker library can be imported
```

## 💻 Getting Started

Start by downloading or cloning repository (make sure git is installed)

```
git clone https://github.com/ndpooja/fibretracker
```

```
cd fibretracker
```

Create a new environment (highly recommended)

`make create_env` or `conda create -n fibretracker`

```
conda activate fibretracker
```

Install the necessary libraries

```
make requirements
```

Go to `notebooks/fibre_tracking.ipynb` and run the notebook with `fibretracker` enviroment

### Data 

Following are the dataset on which fibre tracking is tested

* Mock and UD [[link](https://zenodo.org/records/5483719)] - `UD-01_FoV_2_B2_recon.txm` and `Mock-01_FoV_2_B2_recon.txm` (tested on 150 slices)
* GFRP [[link](https://zenodo.org/records/4771123)] - `GFRP_Initial.zip` (tested on 150 slices)
* SRCT [[link](https://zenodo.org/records/1195879)] - `SRCT.zip` (61 slices available)

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
