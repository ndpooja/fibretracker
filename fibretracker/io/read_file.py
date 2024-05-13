"""
Provides functionality for loading data from various file formats.

# Module taken from qim3d library io loader.py


Example:
    ```
    import fibretracker
    data = fibretracker.io.load("image.tif")
    vol = fibretracker.io.load(data_path2, contains='.tif') # Load a stack of TIF files
    ```

"""

import difflib
import os
import re
import struct
from pathlib import Path

# import dask.array as da
import h5py
import nibabel as nib
import numpy as np
# import olefile
import pydicom
import tifffile
from PIL import Image, UnidentifiedImageError


class DataLoader:
    """Utility class for loading data from different file formats.

    Attributes:
        dataset_name (str): Specifies the name of the dataset to be loaded
            (only relevant for HDF5 files)
        return_metadata (bool): Specifies if metadata is returned or not
            (only relevant for HDF5, TXRM/TXM/XRM and NIfTI files)
        contains (str): Specifies a part of the name that is common for the
            TIFF file stack to be loaded (only relevant for TIFF stacks)

    Methods:
        load_tiff(path): Load a TIFF file from the specified path.
        load_h5(path): Load an HDF5 file from the specified path.
        load_tiff_stack(path): Load a stack of TIFF files from the specified path.
        load_txrm(path): Load a TXRM/TXM/XRM file from the specified path
        load(path): Load a file or directory based on the given path

    Example:
        loader = fibretracker.io.DataLoader(virtual_stack=True)
        data = loader.load_tiff("image.tif")
    """

    def __init__(self, **kwargs):
        """Initializes a new instance of the DataLoader class.

        Args:
            dataset_name (str, optional): Specifies the name of the dataset to be loaded
                in case multiple dataset exist within the same file. Default is None (only for HDF5 files)
            return_metadata (bool, optional): Specifies whether to return metadata or not. Default is False (only for HDF5, TXRM/TXM/XRM and NIfTI files)
            contains (str, optional): Specifies a part of the name that is common for the TIFF file stack to be loaded (only for TIFF stacks).
                Default is None.
            force_load (bool, optional): If false and user tries to load file that exceeds available memory, throws a MemoryError. If true, this error is
                changed to warning and dataloader tries to load the file. Default is False.
            dim_order (tuple, optional): The order of the dimensions in the volume. Default is (2,1,0) which corresponds to (z,y,x)
        """
        # self.virtual_stack = kwargs.get("virtual_stack", False)
        self.dataset_name = kwargs.get("dataset_name", None)
        self.return_metadata = kwargs.get("return_metadata", False)
        self.contains = kwargs.get("contains", None)
        self.force_load = kwargs.get("force_load", False)
        self.dim_order = kwargs.get("dim_order", (2, 1, 0))

    def load_tiff(self, path):
        """Load a TIFF file from the specified path.

        Args:
            path (str): The path to the TIFF file.

        Returns:
            numpy.ndarray : The loaded volume.

        """
        # Get the number of TIFF series (some BigTIFF have multiple series)
        with tifffile.TiffFile(path) as tif:
            series = len(tif.series)

        vol = tifffile.imread(path, key=range(series) if series > 1 else None)

        # log.info("Loaded shape: %s", vol.shape)

        return vol

    def load_h5(self, path):
        """Load an HDF5 file from the specified path.

        Args:
            path (str): The path to the HDF5 file.

        Returns:
            numpy.ndarray, h5py._hl.dataset.Dataset or tuple: The loaded volume.
                If 'self.return_metadata' is True, returns a tuple (volume, metadata).

        Raises:
            ValueError: If the specified dataset_name is not found or is invalid.
            ValueError: If the dataset_name is not specified in case of multiple datasets in the HDF5 file
            ValueError: If no datasets are found in the file.
        """

        # Read file
        f = h5py.File(path, "r")
        data_keys = _get_h5_dataset_keys(f)
        datasets = []
        metadata = {}
        for key in data_keys:
            if (
                f[key].ndim > 1
            ):  # Data is assumed to be a dataset if it is two dimensions or more
                datasets.append(key)
            if f[key].attrs.keys():
                metadata[key] = {
                    "value": f[key][()],
                    **{attr_key: val for attr_key, val in f[key].attrs.items()},
                }

        # Only one dataset was found
        if len(datasets) == 1:
            # if self.dataset_name:
            #     log.info(
            #         "'dataset_name' argument is unused since there is only one dataset in the file"
            #     )
            name = datasets[0]
            vol = f[name]

        # Multiple datasets were found
        elif len(datasets) > 1:
            if self.dataset_name in datasets:  # Provided dataset name is valid
                name = self.dataset_name
                vol = f[name]
            else:
                if self.dataset_name:  # Dataset name is provided
                    similar_names = difflib.get_close_matches(
                        self.dataset_name, datasets
                    )  # Find closest matching name if any
                    if similar_names:
                        suggestion = similar_names[0]  # Get the closest match
                        raise ValueError(
                            f"Invalid dataset name. Did you mean '{suggestion}'?"
                        )
                    else:
                        raise ValueError(
                            f"Invalid dataset name. Please choose between the following datasets: {datasets}"
                        )
                else:
                    raise ValueError(
                        f"Found multiple datasets: {datasets}. Please specify which of them that you want to load with the argument 'dataset_name'"
                    )

        # No datasets were found
        else:
            raise ValueError(f"Did not find any data in the file: {path}")

        # if not self.virtual_stack:
        vol = vol[()]  # Load dataset into memory
        f.close()

        # log.info("Loaded the following dataset: %s", name)
        # log.info("Loaded shape: %s", vol.shape)

        if self.return_metadata:
            return vol, metadata
        else:
            return vol

    def load_tiff_stack(self, path):
        """Load a stack of TIFF files from the specified path.

        Args:
            path (str): The path to the stack of TIFF files.

        Returns:
            numpy.ndarray or numpy.memmap: The loaded volume.

        Raises:
            ValueError: If the 'contains' argument is not specified.
            ValueError: If the 'contains' argument matches multiple TIFF stacks in the directory
        """
        if not self.contains:
            raise ValueError(
                "Please specify a part of the name that is common for the TIFF file stack with the argument 'contains'"
            )

        tiff_stack = [
            file
            for file in os.listdir(path)
            if (file.endswith(".tif") or file.endswith(".tiff"))
            and self.contains in file
        ]
        tiff_stack.sort()  # Ensure proper ordering

        # Check that only one TIFF stack in the directory contains the provided string in its name
        tiff_stack_only_letters = []
        for filename in tiff_stack:
            name = os.path.splitext(filename)[0]  # Remove file extension
            tiff_stack_only_letters.append(
                "".join(filter(str.isalpha, name))
            )  # Remove everything else than letters from the name

        # Get unique elements from tiff_stack_only_letters
        unique_names = list(set(tiff_stack_only_letters))
        if len(unique_names) > 1:
            raise ValueError(
                f"The provided part of the filename for the TIFF stack matches multiple TIFF stacks: {unique_names}.\nPlease provide a string that is unique for the TIFF stack that is intended to be loaded"
            )

        vol = tifffile.imread(
            [os.path.join(path, file) for file in tiff_stack], out="memmap"
        )

        # if not self.virtual_stack:
        vol = np.copy(vol)  # Copy to memory

        # log.info("Found %s file(s)", len(tiff_stack))
        # log.info("Loaded shape: %s", vol.shape)

        return vol

    def load_txrm(self, path):
        """Load a TXRM/XRM/TXM file from the specified path.

        Args:
            path (str): The path to the TXRM/TXM file.

        Returns:
            numpy.ndarray, dask.array.core.Array or tuple: The loaded volume.
                If 'virtual_stack' is True, returns a dask.array.core.Array object.
                If 'return_metadata' is True, returns a tuple (volume, metadata).

        Raises:
            ValueError: If the dxchange library is not installed
        """

        try:
            import dxchange
        except ImportError:
            raise ValueError(
                "The library dxchange is required to load TXRM files. Please find installation instructions at https://dxchange.readthedocs.io/en/latest/source/install.html"
            )

        # if self.virtual_stack:
        #     if not path.endswith(".txm"):
        #         print(
        #             "Virtual stack is only thoroughly tested for reconstructed volumes in TXM format and is thus not guaranteed to load TXRM and XRM files correctly"
        #         )

        #     # Get metadata
        #     ole = olefile.OleFileIO(path)
        #     metadata = dxchange.reader.read_ole_metadata(ole)

        #     # Compute data offsets in bytes for each slice
        #     offsets = _get_ole_offsets(ole)

        #     if len(offsets) != metadata["number_of_images"]:
        #         raise ValueError(
        #             f'Metadata is erroneous: number of images {metadata["number_of_images"]} is different from number of data offsets {len(offsets)}'
        #         )

        #     slices = []
        #     for _, offset in offsets.items():
        #         slices.append(
        #             np.memmap(
        #                 path,
        #                 dtype=dxchange.reader._get_ole_data_type(metadata).newbyteorder(
        #                     "<"
        #                 ),
        #                 mode="r",
        #                 offset=offset,
        #                 shape=(1, metadata["image_height"], metadata["image_width"]),
        #             )
        #         )

        #     vol = da.concatenate(slices, axis=0)
        #     print(
        #         "Virtual stack volume will be returned as a dask array. To load certain slices into memory, use normal indexing followed by the compute() method, e.g. vol[:,0,:].compute()"
        #     )

        # else:
        vol, metadata = dxchange.read_txrm(path)
        vol = (
            vol.squeeze()
        )  # In case of an XRM file, the third redundant dimension is removed

        if self.return_metadata:
            return vol, metadata
        else:
            return vol

    def load_nifti(self, path):
        """Load a NIfTI file from the specified path.

        Args:
            path (str): The path to the NIfTI file.

        Returns:
            numpy.ndarray, nibabel.arrayproxy.ArrayProxy or tuple: The loaded volume.
                If 'self.virtual_stack' is True, returns a nibabel.arrayproxy.ArrayProxy object
                If 'self.return_metadata' is True, returns a tuple (volume, metadata).
        """

        data = nib.load(path)

        # Get image array proxy
        vol = data.dataobj

        # if not self.virtual_stack:
        vol = np.asarray(vol, dtype=data.get_data_dtype())

        if self.return_metadata:
            metadata = {}
            for key in data.header:
                metadata[key] = data.header[key]

            return vol, metadata
        else:
            return vol

    def load_pil(self, path):
        """Load a PIL image from the specified path

        Args:
            path (str): The path to the image supported by PIL.

        Returns:
            numpy.ndarray: The loaded image/volume.
        """
        return np.array(Image.open(path))

    # def _load_vgi_metadata(self, path):
    #     """Helper functions that loads metadata from a VGI file

    #     Args:
    #         path (str): The path to the VGI file.

    #     returns:
    #         dict: The loaded metadata.
    #     """
    #     meta_data = {}
    #     current_section = meta_data
    #     section_stack = [meta_data]

    #     should_indent = True

    #     with open(path, "r") as f:
    #         for line in f:
    #             line = line.strip()
    #             # {NAME} is start of a new object, so should indent
    #             if line.startswith("{") and line.endswith("}"):
    #                 section_name = line[1:-1]
    #                 current_section[section_name] = {}
    #                 section_stack.append(current_section)
    #                 current_section = current_section[section_name]

    #                 should_indent = True
    #             # [NAME] is start of a section, so should not indent
    #             elif line.startswith("[") and line.endswith("]"):
    #                 section_name = line[1:-1]

    #                 if not should_indent:
    #                     if len(section_stack) > 1:
    #                         current_section = section_stack.pop()

    #                 current_section[section_name] = {}
    #                 section_stack.append(current_section)
    #                 current_section = current_section[section_name]

    #                 should_indent = False
    #             # = is a key value pair
    #             elif "=" in line:
    #                 key, value = line.split("=", 1)
    #                 current_section[key.strip()] = value.strip()
    #             elif line == "":
    #                 if len(section_stack) > 1:
    #                     current_section = section_stack.pop()

    #     return meta_data

    def load_dicom(self, path):
        """Load a DICOM file

        Args:
            path (str): Path to file
        """
        dcm_data = pydicom.dcmread(path)

        if self.return_metadata:
            return dcm_data.pixel_array, dcm_data
        else:
            return dcm_data.pixel_array

    def load_dicom_dir(self, path):
        """Load a directory of DICOM files into a numpy 3d array

        Args:
            path (str): Directory path
        """
        if not self.contains:
            raise ValueError(
                "Please specify a part of the name that is common for the DICOM file stack with the argument 'contains'"
            )

        dicom_stack = [file for file in os.listdir(path) if self.contains in file]
        dicom_stack.sort()  # Ensure proper ordering

        # Check that only one DICOM stack in the directory contains the provided string in its name
        dicom_stack_only_letters = []
        for filename in dicom_stack:
            name = os.path.splitext(filename)[0]  # Remove file extension
            dicom_stack_only_letters.append(
                "".join(filter(str.isalpha, name))
            )  # Remove everything else than letters from the name

        # Get unique elements from tiff_stack_only_letters
        unique_names = list(set(dicom_stack_only_letters))
        if len(unique_names) > 1:
            raise ValueError(
                f"The provided part of the filename for the DICOM stack matches multiple DICOM stacks: {unique_names}.\nPlease provide a string that is unique for the DICOM stack that is intended to be loaded"
            )

        # dicom_list contains the dicom objects with metadata
        dicom_list = [pydicom.dcmread(os.path.join(path, f)) for f in dicom_stack]
        # vol contains the pixel data
        vol = np.stack([dicom.pixel_array for dicom in dicom_list], axis=0)

        if self.return_metadata:
            return vol, dicom_list
        else:
            return vol


    def load(self, path):
        """
        Load a file or directory based on the given path.

        Args:
            path (str or os.PathLike): The path to the file or directory.

        Returns:
            vol (numpy.ndarray, numpy.memmap, h5py._hl.dataset.Dataset, nibabel.arrayproxy.ArrayProxy or tuple): The loaded volume

                If `virtual_stack=True`, returns `numpy.memmap`, `h5py._hl.dataset.Dataset` or `nibabel.arrayproxy.ArrayProxy` depending on file format
                If `return_metadata=True` and file format is either HDF5, NIfTI or TXRM/TXM/XRM, returns `tuple` (volume, metadata).

        Raises:
            ValueError: If the format is not supported
            ValueError: If the file or directory does not exist.
            MemoryError: If file size exceeds available memory and force_load is not set to True. In check_size function.

        Example:
            loader = qim3d.io.DataLoader()
            data = loader.load("image.tif")
        """

        # Stringify path in case it is not already a string
        path = str(path)

        # Load a file
        if os.path.isfile(path):
            # Choose the loader based on the file extension
            # self.check_file_size(path)
            if path.endswith(".tif") or path.endswith(".tiff"):
                return self.load_tiff(path)
            elif path.endswith(".h5"):
                return self.load_h5(path)
            elif path.endswith((".txrm", ".txm", ".xrm")):
                return self.load_txrm(path)
            elif path.endswith((".nii", ".nii.gz")):
                return self.load_nifti(path)
            elif path.endswith((".dcm", ".DCM")):
                return self.load_dicom(path)
            else:
                try:
                    return self.load_pil(path)
                except UnidentifiedImageError:
                    raise ValueError("Unsupported file format")

        # Load a directory
        elif os.path.isdir(path):
            # load tiff stack if folder contains tiff files else load dicom directory
            if any(
                [f.endswith(".tif") or f.endswith(".tiff") for f in os.listdir(path)]
            ):
                return self.load_tiff_stack(path)
            else:
                return self.load_dicom_dir(path)

        # Fails
        else:
            # Find the closest matching path to warn the user
            parent_dir = os.path.dirname(path) or "."
            parent_files = os.listdir(parent_dir) if os.path.isdir(parent_dir) else ""
            valid_paths = [os.path.join(parent_dir, file) for file in parent_files]
            similar_paths = difflib.get_close_matches(path, valid_paths)
            if similar_paths:
                suggestion = similar_paths[0]  # Get the closest match
                message = f"Invalid path. Did you mean '{suggestion}'?"
                raise ValueError(repr(message))
            else:
                raise ValueError("Invalid path")


def _get_h5_dataset_keys(f):
    keys = []
    f.visit(lambda key: keys.append(key) if isinstance(f[key], h5py.Dataset) else None)
    return keys


def _get_ole_offsets(ole):
    slice_offset = {}
    for stream in ole.listdir():
        if stream[0].startswith("ImageData"):
            sid = ole._find(stream)
            direntry = ole.direntries[sid]
            sect_start = direntry.isectStart
            offset = ole.sectorsize * (sect_start + 1)
            slice_offset[f"{stream[0]}/{stream[1]}"] = offset

    # sort dictionary after natural sorting (https://blog.codinghorror.com/sorting-for-humans-natural-sort-order/)
    sorted_keys = sorted(
        slice_offset.keys(),
        key=lambda string_: [
            int(s) if s.isdigit() else s for s in re.split(r"(\d+)", string_)
        ],
    )
    slice_offset_sorted = {key: slice_offset[key] for key in sorted_keys}

    return slice_offset_sorted


def load(
    path,
    # virtual_stack=False,
    dataset_name=None,
    return_metadata=False,
    contains=None,
    force_load: bool = False,
    dim_order=(2, 1, 0),
    **kwargs,
):
    """
    Load data from the specified file or directory.

    Args:
        path (str or os.PathLike): The path to the file or directory.
        virtual_stack (bool, optional): Specifies whether to use virtual
            stack when loading files. Default is False.
        dataset_name (str, optional): Specifies the name of the dataset to be loaded
            in case multiple dataset exist within the same file. Default is None (only for HDF5 files)
        return_metadata (bool, optional): Specifies whether to return metadata or not. Default is False (only for HDF5 and TXRM/TXM/XRM files)
        contains (str, optional): Specifies a part of the name that is common for the TIFF file stack to be loaded (only for TIFF stacks).
            Default is None.
        force_load (bool, optional): If the file size exceeds available memory, a MemoryError is raised.
            If force_load is True, the error is changed to warning and the loader tries to load it anyway. Default is False.
        dim_order (tuple, optional): The order of the dimensions in the volume for .vol files. Default is (2,1,0) which corresponds to (z,y,x)
        **kwargs: Additional keyword arguments to be passed
        to the DataLoader constructor.

    Returns:
        vol (numpy.ndarray, numpy.memmap, h5py._hl.dataset.Dataset, nibabel.arrayproxy.ArrayProxy or tuple): The loaded volume

            If `virtual_stack=True`, returns `numpy.memmap`, `h5py._hl.dataset.Dataset` or `nibabel.arrayproxy.ArrayProxy` depending on file format
            If `return_metadata=True` and file format is either HDF5, NIfTI or TXRM/TXM/XRM, returns `tuple` (volume, metadata).

    Raises:
        MemoryError: if the given file size exceeds available memory

    Example:
        ```python
        import qim3d

        vol = fibretracker.io.load("path/to/image.tif")
        ```
    """

    loader = DataLoader(
        # virtual_stack=virtual_stack,
        dataset_name=dataset_name,
        return_metadata=return_metadata,
        contains=contains,
        force_load=force_load,
        dim_order=dim_order,
        **kwargs,
    )

    data = loader.load(path)

    # def log_memory_info(data):
    #     mem = Memory()
    #     log.info(
    #         "Volume using %s of memory\n",
    #         sizeof(data[0].nbytes if isinstance(data, tuple) else data.nbytes),
    #     )
    #     mem.report()

    # if return_metadata and not isinstance(data, tuple):
    #     log.warning("The file format does not contain metadata")

    # if not virtual_stack:
    #     log_memory_info(data)
    # else:
    #     # Only log if file type is not a np.ndarray, i.e., it is some kind of memmap object
    #     if not isinstance(
    #         type(data[0]) if isinstance(data, tuple) else type(data), np.ndarray
    #     ):
    #         log.info("Using virtual stack")
    #     else:
    #         log.warning("Virtual stack is not supported for this file format")
    #         log_memory_info(data)

    return data


# class ImgExamples:
#     """Image examples

#     Attributes:
#         blobs_256x256 (numpy.ndarray): A 2D image of blobs.
#         blobs_256x256x256 (numpy.ndarray): A 3D volume of blobs.
#         bone_128x128x128 (numpy.ndarray): A 3D volume of bone.
#         cement_128x128x128 (numpy.ndarray): A 3D volume of cement.
#         fly_150x256x256 (numpy.ndarray): A 3D volume of a fly.
#         NT_10x200x100 (numpy.ndarray): A 3D volume of a neuron.
#         NT_128x128x128 (numpy.ndarray): A 3D volume of a neuron.
#         shell_225x128x128 (numpy.ndarray): A 3D volume of a shell.

#     Tip:
#         Call `qim3d.examples.<name>` to access the image examples easily as this class is instantiated when importing `qim3d`

#     Example:
#         ```python
#         import qim3d

#         data = qim3d.examples.blobs_256x256
#         ```


#     """

#     def __init__(self):
#         img_examples_path = Path(qim3d.__file__).parents[0] / "img_examples"
#         img_paths = list(img_examples_path.glob("*.tif"))


#         update_dict = {path.stem : load(path) for path in img_paths}
#         self.__dict__.update(update_dict)

def normalize_volume(vol):
    """Normalize the volume to the range [0, 1].

    Args:
        vol (numpy.ndarray): The volume to normalize.

    Returns:
        numpy.ndarray: The normalized volume.
    """
    return (vol - np.min(vol)) / (np.max(vol) - np.min(vol))