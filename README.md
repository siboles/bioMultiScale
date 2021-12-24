Description
===========
Automates the multiscale finite element modelling process for tissue to cellular scales. 
The original implementation was aimed at usage in articular cartilage, but may be extended 
to other applications.

Installation
============
```buildoutcfg
conda install febio pygalmesh vtk pyyaml pandas scipy seaborn wquantiles
```

```buildoutcfg
python -m pip install raster_geometry
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

```buildoutcfg
python -m bioMultiScale.TissueScaleModel PATH_TO_YAML_FILE
```

This will generate a model defintion file with the ".feb" extension and a VTK rectilinear grid with the ".vtr"
extension. The names of these are specified in the YAML file.

To solve the model, FEBio must be called with:

```buildoutcfg
febio -i PATH_TO_MODEL_FILE 
```

Here, the febio command may vary by system.

Cellular Scale Model Generation
-------------------------------
Cellular scale models are, likewise, configured with YAML files such as models/cell_scale/cell_scale_all.yaml.
As the cellular model definition relies on results and geometry from the tissue scale model, these models and solutions
must exist and be specified appropriately. Model generation is then achieved with:

```buildoutcfg
python -m bioMultiScale.CellScaleModel PATH_TO_YAML_FILE
```

Material Fitting
================
The script material_fitting/HolmesMow_opt.py is not part of the bioMultiScale package, but may be useful. This optimizes
the parameters $E$ and $\beta$ of the Holmes-Mow elastic potential as defined in the 
[FEBio User Manual](https://help.febio.org/FebioUser/FEBio_um_3-4-4.1.3.9.html). The material testing data is indicated
in material_fitting/material_test.yaml. This assumes the steady-state stress from a uniaxial strain 
(confined compression) test.