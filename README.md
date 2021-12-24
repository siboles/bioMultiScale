Description
===========
Automates the multiscale finite element modelling process for tissue to cellular scales. 
The original implementation was aimed at usage in articular cartilage, but may be extended 
to other applications.

Installation
============
The Python package utilizes the conda package manager for dependency resolution. We recommend 
installing the miniforge3 implementation of conda from <https://conda-forge.org/miniforge/>

After conda installation first add the siboles channel by executing:

```
conda config --add channels siboles
```

If you did not install miniforge3, you may also need to add the conda-forge channel with:

```
conda config --add channels conda-forge
```

One can then create an isolated conda environment with all necessary dependencies with the command:

```
conda create -n NAME_OF_ENVIRONMENT febio pygalmesh vtk pyyaml pandas scipy seaborn wquantiles
```

where, NAME_OF_ENVIRONMENT, is the user's choice. To use the bioMultiScale Python package
the conda environment must be activated with:

```
conda activate NAME_OF_ENVIRONMENT
```

The raster_geometry package must be installed into the environment from PyPi with:

```
python -m pip install raster_geometry
```

---
**Note**
The conda environment must be active when the pip command above is executed!
---

Finally, bioMultiScale can be installed for system-wide execution with:

```
cd src
python setup.py install
```

Usage
=====
The multiscale modelling procedure requires the generation and solution of a tissue scale model a priori. Cellular
scale models will then use the results from this model in their definition. The solutions of the tissue and cellular
scale models can then be post-processed with the read_tissue_plot, read_cell_plot, and process_vtk modules.

Tissue Scale Model Generation
-----------------------------
Tissue scale models can be configured with YAML files, such as models/tissue_scale/tissue_5percent.yaml.
The model can then be generated with:

```
python -m bioMultiScale.TissueScaleModel PATH_TO_YAML_FILE
```

This will generate a model defintion file with the ".feb" extension and a VTK rectilinear grid with the ".vtr"
extension. The names of these are specified in the YAML file.

To solve the model, FEBio must be called with:

```
febio -i PATH_TO_MODEL_FILE 
```

Here, the febio command may vary by system.

Cellular Scale Model Generation
-------------------------------
Cellular scale models are, likewise, configured with YAML files such as models/cell_scale/cell_scale_all.yaml.
As the cellular model definition relies on results and geometry from the tissue scale model, these models and solutions
must exist and be specified appropriately. Model generation is then achieved with:

```
python -m bioMultiScale.CellScaleModel PATH_TO_YAML_FILE
```

Material Fitting
================
The script material_fitting/HolmesMow_opt.py is not part of the bioMultiScale package, but may be useful. This optimizes
the parameters $E$ and $\beta$ of the Holmes-Mow elastic potential as defined in the 
[FEBio User Manual](https://help.febio.org/FebioUser/FEBio_um_3-4-4.1.3.9.html). The material testing data is indicated
in material_fitting/material_test.yaml. This assumes the steady-state stress from a uniaxial strain 
(confined compression) test.