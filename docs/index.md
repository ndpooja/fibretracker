# FibreTracker
<img style="float: center;" src="figures/logo.gif" width="256"> 

A python library to track fibre in a volume

## 💻 Getting Started

Create a new environment (highly recommended)

??? info "Miniconda installation and setup"

    [Miniconda](https://docs.anaconda.com/miniconda/) is a free lightweight installer for conda. 

    Here are commands to quickly setup the conda. For reference, you can also use [installation link](https://docs.anaconda.com/miniconda/miniconda-install/)

    === "Windows"
        Following commands will install the latest 64-bit version and delete the installer. To install different version change the `.exe` version to desired version in the `curl` command line.

        ```bash
        curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe -o miniconda.exe
        start /wait "" miniconda.exe /S
        del miniconda.exe
        ```
        After successful installation, search and open "Ananconda prompt (miniconda3)".
    
    === "macOS"
        Following commands will install the latest 64-bit version and delete the installer. To install different version change the `.sh` version to desired version in the `curl` command line.

        ```bash
        mkdir -p ~/miniconda3
        curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o ~/miniconda3/miniconda.sh
        bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
        rm -rf ~/miniconda3/miniconda.sh
        ```

        After successful installation, initialize your miniconda (in general, it is intialized; just close the current terminal and open a new terminal). If not, following commands initialize for bash and zsh shells :

        ```bash
        ~/miniconda3/bin/conda init bash
        ~/miniconda3/bin/conda init zsh
        ```

    === "Linux"
        Following commands will install the latest 64-bit version and delete the installer. To install different version change the `.sh` version to desired version in the `wget` command line.

        ```bash
        mkdir -p ~/miniconda3
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
        bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
        rm -rf ~/miniconda3/miniconda.sh
        ```

        After installing, initialize your newly-installed Miniconda (in general, it is initialized; just close the current terminal and open a new terminal). If not, following commands initialize for bash and zsh shells:

        ```bash
        ~/miniconda3/bin/conda init bash
        ~/miniconda3/bin/conda init zsh
        ```
Once installed, create the environment

```
conda create -n fibretracker python=3.11
```

Activate the environment by running

```
conda activate fibretracker
```

To read .txm file, install `dxchange` using `conda` [install before fibretracker module to avoid version conflicts and related error]

```
conda install -c conda-forge dxchange
```

Install the FibreTracker tool using `pip`

```
pip install fibretracker
```

Open jupyter notebook and create a new notebook

```
jupyter notebook
```

Go to Example and run the notebook with `fibretracker` enviroment

## Data

Following are the dataset for which fibre tracking is tested on 250 slices

* Mock and UD [[link](https://zenodo.org/records/5483719)] -    `UD-01_FoV_2_B2_recon.txm`
                                                                `Mock-01_FoV_2_B2_recon.txm`
* GFRP [[link](https://zenodo.org/records/4771123)] - `GFRP_Initial.zip`
* XCT Low-Res [[link](https://zenodo.org/records/1195879)] - `XCT_L.zip`

## License

`fibretracker` was created by Kumari Pooja. It is licensed under the terms
of the MIT license.

## Credits

This work is supported by the [RELIANCE](https://www.chalmers.se/en/projects/reliance/) doctoral network via the Marie Skłodowska-Curie Actions HORIZON-MSCA-2021-DN- 01. Project no: 101073040 

<img style="float: center;" src="figures/reliance_logo.png" width="128"> 

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
