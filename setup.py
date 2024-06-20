from setuptools import find_packages, setup

# Read the contents of your README file
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


setup(
    name="fibretracker",
    version="0.1.0",
    author="Kumari Pooja",
    author_email="pooja@dtu.dk",
    packages=find_packages(),
    include_package_data=True,
    url="https://ndpooja.github.io/fibretracker",
    description='A python library to track fibre in a volume',
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='MIT',

    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Natural Language :: English",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.10",
    install_requires=[
        "h5py>=3.9.0",
        "matplotlib>=3.8.0",
        "pydicom>=2.4.4",
        "numpy>=1.26.0",
        "outputformat>=0.1.3",
        "plotly>=5.14.1",
        "seaborn>=0.12.2",
        "Pillow>=10.0.1",
        "scipy>=1.11.2",
        "setuptools>=68.0.0",
        "tifffile>=2023.4.12",
        "tqdm>=4.65.0",
        "nibabel>=5.2.0",
        "ipywidgets>=8.1.2",
        "olefile>=0.46",
        "ipympl>=0.9.4",
        "scikit-image>=0.19.0",
        "notebook>=6.4.0",
        "ipykernel>=6.29.4",
    ],
)
