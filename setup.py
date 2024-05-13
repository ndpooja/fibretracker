from setuptools import find_packages, setup

setup(
    name="fibretracker",
    version="0.1.0",
    author="Kumari Pooja",
    author_email="pooja@dtu.dk",
    packages=find_packages(),
    description='A library to track fibre in a volume',
    license='MIT',


    python_requires=">=3.10",
    install_requires=[
        "h5py>=3.9.0",
        "matplotlib>=3.8.0",
        "pydicom>=2.4.4",
        "numpy>=1.26.0",
        "outputformat>=0.1.3",
        "Pillow>=10.0.1",
        "plotly>=5.14.1",
        "scipy>=1.11.2",
        "seaborn>=0.12.2",
        "setuptools>=68.0.0",
        "tifffile>=2023.4.12",
        "tqdm>=4.65.0",
        "nibabel>=5.2.0",
        "ipywidgets>=8.1.2",
        "olefile>=0.46",
        "ipympl>=0.7.0",
    ],
)
