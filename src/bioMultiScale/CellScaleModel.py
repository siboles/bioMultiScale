import sys
from collections import MutableMapping

import vtk
import numpy as np
from vtk.util import numpy_support
import raster_geometry
import pygalmesh
import febio
import yaml


class FixedDict(MutableMapping):
    def __init__(self, data):
        self.__data = data

    def __len__(self):
        return len(self.__data)

    def __iter__(self):
        return iter(self.__data)

    def __setitem__(self, k, v):
        if k not in self.__data:
            raise KeyError("{:s} is not an acceptable key.".format(k))

        self.__data[k] = v

    def __delitem__(self, k):
        raise NotImplementedError

    def __getitem__(self, k):
        if k not in self.__data:
            raise KeyError("{:s} is not an acceptable key.".format(k))
        return self.__data[k]

    def __contains__(self, k):
        return k in self.__data

    def __repr__(self):
        return repr(self.__data)


class Model():
    def __init__(self, **kwargs):
        self.options = FixedDict({
            "Micro Model Base Name": "cell_scale",
            "Macro Model Geometry File": "tissue.vtr",
            "Macro Model Plot File": "tissue.xplt",
            "Shape Parameters": FixedDict({
                "PCM Radius": [6.976, 3.9763, 2.9175],
                "Cell Radius": [6.0, 3.0, 2.0],
                "PCM Squareness": [2.1, 2.1, 2.1],
                "Cell Squareness": [2.1, 2.1, 2.1]}),
            "Mesh Divisions": FixedDict({
                "Cell": 4.0,
                "PCM": 6.0,
                "ECM": 5.0}),
            "Symmetry": "full",
            "Cell Shifts": {"neutral": 0.0},
            "Model Centroids": [[50, 50, 490]],
            "Material Properties": FixedDict({
                "PCM E ratio": 0.2,
                "PCM v ratio": 1.0,
                "PCM beta ratio": 1.0,
                "PCM phi": 0.2,
                "PCM perm ratio": 1.0,
                "PCM m": 4.638,
                "PCM alpha": 0.0848,
                "Cell Modulus": 0.001,
                "Cell Poisson": 0.01,
                "Cell Permeability": 1e-3,
                "Cell phi": 0.1,
                "Membrane Modulus": 0.001,
                "Membrane Poisson": 0.0,
                "Membrane Permeability": 7e-10,
                "Membrane phi": 0.1,
                "Membrane Thickness": 1e-5
            })})

        self.config = None

        for key, value in kwargs.items():
            if key == "options":
                for k, v in value.items():
                    for k2, v2 in v.items():
                        self.options[k][k2] = v2
            else:
                setattr(self, key, value)

        if self.config is not None:
            self.parseConfig()

        self.ecm_materials = {}
        self.pcm_materials = {}

        print("... Extracting Macro Scale Results and Geometry from \n\t {} and {}".format(
            self.options["Macro Model Plot File"],
            self.options["Macro Model Geometry File"]))
        self.assembleMacroData()
        print("... Extraction Complete")
        for case, shift in self.options["Cell Shifts"].items():
            print("... Creating Mesh for {} case".format(case))
            self.makeMesh(shift)
            print("... Mesh Completed")
            for translation in self.options["Model Centroids"]:
                print("... ... Creating FEBio model for {} case at location: {:6.3f}, {:6.3f}, {:6.3f}".format(
                    case,
                    *translation))
                self.make_febio_model(case, translation)
                print("... ... Model written to {}_{}_{:d}.feb".format(
                    self.options["Micro Model Base Name"],
                    case,
                    int(translation[2])))

    def parseConfig(self):
        """
        Parse configuration file in YAML format.
        """
        with open(self.config) as user:
            user_settings = yaml.safe_load(user)

        for k, v in list(user_settings.items()):
            if isinstance(v, dict):
                for k2, v2 in list(v.items()):
                    self.options[str(k)][str(k2)] = v2
            else:
                self.options[str(k)] = v

    def assembleMacroData(self):
        # read macro_model from vtk grid
        reader = vtk.vtkXMLRectilinearGridReader()
        reader.SetFileName(self.options["Macro Model Geometry File"])
        reader.Update()
        self.macro_data = reader.GetOutput()

        # parse FEBio xplt
        self.macro_results = febio.FebPlt(self.options["Macro Model Plot File"])

    def makeMesh(self, cell_shift):
        resolution = 500  # Image resolution for image-based mesh
        cell_diameter = [r * 2 for r in self.options["Shape Parameters"]["Cell Radius"]]
        pcm_diameter = [r * 2 for r in self.options["Shape Parameters"]["PCM Radius"]]
        ecm_diameter = [d * 10.0**(1. / 3.) for d in pcm_diameter]

        cell_exponent = list(self.options["Shape Parameters"]["Cell Squareness"])
        pcm_exponent = list(self.options["Shape Parameters"]["PCM Squareness"])
        ecm_exponent = pcm_exponent

        symmetry = self.options["Symmetry"]
        if symmetry == 'full':
            position = [0.5, 0.5, 0.5]
            cell_position = [0.5,
                             0.5,
                             0.5 + cell_shift / ecm_diameter[2]]
        elif symmetry == 'half':
            position = [0.0, 0.5, 0.5]
            cell_position = [0.0,
                             0.5,
                             0.5 + cell_shift / ecm_diameter[2]]
        elif symmetry == 'quarter':
            position = [0.0, 0.0, 0.5]
            cell_position = [0.0,
                             0.0,
                             0.5 + cell_shift / ecm_diameter[2]]
        else:
            raise KeyError("Symmetry must be set to 1 of the following: full, half, quarter")

        dmax = np.max(ecm_diameter)
        resolution = [int(d3 / dmax * resolution) for d3 in ecm_diameter]

        semisizes1 = [0.5 * d1 / d3 for d1, d3 in zip(cell_diameter, ecm_diameter)]
        semisizes2 = [0.5 * d2 / d3 for d2, d3 in zip(pcm_diameter, ecm_diameter)]

        s1 = raster_geometry.nd_superellipsoid(resolution,
                                               semisizes=semisizes1,
                                               indexes=cell_exponent,
                                               position=cell_position,
                                               smoothing=True)
        s2 = raster_geometry.nd_superellipsoid(resolution,
                                               semisizes=semisizes2,
                                               indexes=pcm_exponent,
                                               position=position,
                                               smoothing=True)
        s3 = raster_geometry.nd_superellipsoid(resolution,
                                               semisizes=[0.5, 0.5, 0.5],
                                               indexes=ecm_exponent,
                                               position=position,
                                               smoothing=True)

        s1 += 0.2 
        s2 += 0.2
        s3 += 0.2

        v = s1.astype(np.uint8) + s2.astype(np.uint8) + s3.astype(np.uint8)

        h = [r / res for res, r in zip(resolution, ecm_diameter)]

        divisions = self.options["Mesh Divisions"]
        cell_sizes_map = {1: np.min(np.array(ecm_diameter) - np.array(pcm_diameter)) / divisions["ECM"] / 2.0,
                          2: np.min(np.array(pcm_diameter) - np.array(cell_diameter)) / divisions["PCM"] / 2.0,
                          3: np.min(np.array(cell_diameter)) / divisions["Cell"] / 2.0}

        max_facet_distance = np.min(h)
        if symmetry != 'full':
            max_facet_distance /= 2.0

        mesh = pygalmesh.generate_from_array(v,
                                             h,
                                             max_facet_distance=max_facet_distance,
                                             max_radius_surface_delaunay_ball=cell_sizes_map[1],
                                             max_circumradius_edge_ratio=1.3,
                                             max_cell_circumradius=cell_sizes_map,
                                             max_edge_size_at_feature_edges=cell_sizes_map[2],
                                             lloyd=True,
                                             odt=True,
                                             verbose=False)
        mesh.points = (mesh.points - np.array(ecm_diameter) / 2.0) / 1000.

        self.nodes = mesh.points
        # Get face connectivity and flatten - keep only unique node ids
        surface = np.sort(np.unique(mesh.get_cells_type('triangle').ravel()))
        # Get Domain IDs of all nodes
        node_type = mesh.point_data['medit:ref']
        # Keep domain ID of only face nodes
        s_node_types = node_type[surface]
        # Define node set for outermost faces - domain ID = 1
        self.surface_nodes = surface[s_node_types == 1]

        self.elements = mesh.get_cells_type('tetra')
        # get the domain ID of the solid elements
        element_data = mesh.get_cell_data('medit:ref', 'tetra')
        # Make sets from domain IDs
        self.cell_set = np.argwhere(element_data == 3)
        self.pcm_set = np.argwhere(element_data == 2)
        self.ecm_set = np.argwhere(element_data == 1)

        # Get Face connectivity list
        triangles = mesh.get_cells_type('triangle')
        # Get Domain ID of face elements
        surface_data = mesh.get_cell_data('medit:ref', 'triangle')
        # Define membrane elements - Domain ID = 3
        self.membrane = triangles[surface_data == 3]
        # Define element set for membrane
        self.membrane_set = np.argwhere(surface_data == 3)

    def make_vtk(self, translation):
        vtk_mesh = vtk.vtkUnstructuredGrid()
        translated_nodes = self.nodes + np.array(translation) / 1000.
        nodearray = numpy_support.numpy_to_vtk(
            translated_nodes.ravel(), deep=True, array_type=vtk.VTK_DOUBLE)
        nodearray.SetNumberOfComponents(3)
        nodes = vtk.vtkPoints()
        nodes.SetData(nodearray)
        vtk_mesh.SetPoints(nodes)

        element_array = np.hstack([np.ones((self.elements.shape[0], 1), dtype=int) * 4,
                                   self.elements])
        size = element_array.size
        element_array = numpy_support.numpy_to_vtk(
            element_array.ravel(), deep=True, array_type=vtk.VTK_ID_TYPE)
        elements = vtk.vtkCellArray()
        elements.SetCells(size // 5, element_array)
        vtk_mesh.SetCells(10, elements)

        element_set = self._interpMaterials(vtk_mesh)
        element_set[self.cell_set] = 1
        element_set = numpy_support.numpy_to_vtk(element_set.ravel(), deep=True, array_type=vtk.VTK_CHAR)
        element_set.SetNumberOfComponents(1)
        element_set.SetName('Material ID')
        vtk_mesh.GetCellData().AddArray(element_set)

        return vtk_mesh

    def _interpMaterials(self, polydata):
        self.ecm_materials = {}
        self.pcm_materials = {}
        probe = vtk.vtkProbeFilter()
        probe.SetSourceData(self.macro_data)
        probe.SetInputData(polydata)
        probe.PassPointArraysOff()
        probe.PassCellArraysOn()
        probe.Update()

        to_cell = vtk.vtkPointDataToCellData()
        to_cell.SetInputData(probe.GetOutput())
        to_cell.Update()
        mapped_data = to_cell.GetOutput()
        materialIDs = numpy_support.vtk_to_numpy(mapped_data.GetCellData().GetArray('ElementSetID')).astype(int)
        ksi = numpy_support.vtk_to_numpy(mapped_data.GetCellData().GetArray('ksi'))
        E = numpy_support.vtk_to_numpy(mapped_data.GetCellData().GetArray('E'))
        v = numpy_support.vtk_to_numpy(mapped_data.GetCellData().GetArray('v'))
        beta = numpy_support.vtk_to_numpy(mapped_data.GetCellData().GetArray('beta'))
        phi = numpy_support.vtk_to_numpy(mapped_data.GetCellData().GetArray('phi'))
        permeability = numpy_support.vtk_to_numpy(mapped_data.GetCellData().GetArray('permeability'))
        m = numpy_support.vtk_to_numpy(mapped_data.GetCellData().GetArray('m'))
        alpha = numpy_support.vtk_to_numpy(mapped_data.GetCellData().GetArray('alpha'))
        unique_materialIDs = np.sort(np.unique(materialIDs))

        pcm_counter = 0
        pcm_offset = unique_materialIDs.size
        for i, matid in enumerate(unique_materialIDs):
            idx = np.argwhere(materialIDs == matid)
            ecm_ids = np.intersect1d(idx.ravel(), self.ecm_set)
            pcm_ids = np.intersect1d(idx.ravel(), self.pcm_set)
            E_ratio = self.options['Material Properties']['PCM E ratio']
            v_ratio = self.options['Material Properties']['PCM v ratio']
            beta_ratio = self.options['Material Properties']['PCM beta ratio']
            perm_ratio = self.options['Material Properties']['PCM perm ratio']
            self.ecm_materials[i + 3] = {'E': E[idx[0]][0],
                                         'v': v[idx[0]][0],
                                         'beta': beta[idx[0]][0],
                                         'phi': phi[idx[0]][0],
                                         'permeability': permeability[idx[0]][0],
                                         'm': m[idx[0]][0],
                                         'alpha': alpha[idx[0]][0],
                                         'ksi': ksi[idx[0], :][0],
                                         'element_ids': ecm_ids}
            materialIDs[ecm_ids] = i + 3
            if pcm_ids.any():
                self.pcm_materials[pcm_counter + 3 + pcm_offset] = {
                    'E': E[idx[0]][0] * E_ratio,
                    'v': v[idx[0]][0] * v_ratio,
                    'beta': beta[idx[0]][0] * beta_ratio,
                    'phi': phi[idx[0]][0],
                    'permeability': permeability[idx[0]][0] * perm_ratio,
                    'm': m[idx[0]][0],
                    'alpha': alpha[idx[0]][0],
                    'element_ids': pcm_ids}
                materialIDs[pcm_ids] = pcm_counter + 3 + pcm_offset
                pcm_counter += 1
        return materialIDs

    def make_febio_model(self, case, translation):
        model_name = "{}_{}_{:d}".format(self.options["Micro Model Base Name"],
                                         case,
                                         int(translation[2]))
        fe_mesh = febio.MeshDef()
        for i in range(self.nodes.shape[0]):
            fe_mesh.nodes.append([i + 1] + list(self.nodes[i, :] + np.array(translation) / 1000.))

        for i in range(self.elements.shape[0]):
            fe_mesh.elements.append(['tet4', i + 1] + list(self.elements[i, :] + 1))

        membrane_set = np.zeros(self.membrane.shape[0], dtype=int)
        for i in range(self.membrane.shape[0]):
            fe_mesh.elements.append(
                ['tri3', i + 1 + self.elements.shape[0]] + list(self.membrane[i] + 1))
            membrane_set[i] = i + 1 + self.elements.shape[0]

        vtkmesh = self.make_vtk(translation)
        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName(model_name+'.vtu')
        writer.SetInputData(vtkmesh)
        writer.Write()
        # Cell Element Set
        fe_mesh.addElementSet(setname='Cell', eids=self.cell_set + 1)
        # Membrane Element Set
        fe_mesh.addElementSet(setname='Membrane', eids=membrane_set)
        # PCM Element Set
        for k, v in self.pcm_materials.items():
            fe_mesh.addElementSet(setname='PCM {:d}'.format(k), eids=v['element_ids'] + 1)
        # ECM Element Set
        for k, v in self.ecm_materials.items():
            fe_mesh.addElementSet(setname='ECM {:d}'.format(k), eids=v['element_ids'] + 1)

        fe_model = febio.Model(modelfile='.'.join([model_name, 'feb']),
                               steps=[{'Displace': 'biphasic'}])

        mat_props = self.options["Material Properties"]
        # Cell Material
        cell_mat = febio.MatDef(matid=1, mname='Cell', mtype='biphasic',
                                elsets='Cell',
                                attributes={'phi0': '{:.6E}'.format(
                                    mat_props["Cell phi"])})
        cell_mat.addBlock(branch=1, btype='solid', mtype='neo-Hookean',
                          attributes={"E": "{:.6E}".format(mat_props["Cell Modulus"]),
                                      "v": "{:.6E}".format(mat_props["Cell Poisson"])})
        cell_mat.addBlock(branch=1, btype='permeability', mtype='perm-const-iso',
                          attributes={"perm": "{:.6E}".format(mat_props["Cell Permeability"])})
        fe_model.addMaterial(cell_mat)

        # Membrane Material
        mem_mat = febio.MatDef(matid=2, mname='Membrane', mtype='biphasic',
                               elsets='Membrane', 
                               attributes={"phi0": "{:.6E}".format(
                                   mat_props["Membrane phi"])})
        mem_mat.addBlock(branch=1, btype='solid', mtype='neo-Hookean',
                         attributes={"E": "{:.6E}".format(mat_props["Membrane Modulus"]),
                                     "v": "{:.6E}".format(mat_props["Membrane Poisson"])})
        mem_mat.addBlock(branch=1, btype='permeability', mtype='perm-const-iso',
                         attributes={"perm": "{:.6E}".format(mat_props["Membrane Permeability"])})
        fe_mesh.addElementData(elset='Membrane',
                               attributes={"thickness": "{:.6E}".format(
                                   mat_props["Membrane Thickness"])})
        fe_model.addMaterial(mem_mat)
        # ECM Materials
        ecm_mats = []
        for k, v in self.ecm_materials.items():
            mname = 'ECM {:d}'.format(k)
            ecm_mats.append(febio.MatDef(matid=k, mname=mname, mtype='biphasic',
                                         elsets=mname, attributes={'phi0': '{:6E}'.format(v['phi'])}))
            ecm_mats[-1].addBlock(branch=1, btype='solid', mtype='solid mixture',
                                  attributes={'mat_axis': ['vector', "1,0,0", "0,1,0"]})
            ecm_mats[-1].addBlock(branch=2, btype='solid', mtype='Holmes-Mow',
                                  attributes={"E": "{:.6E}".format(v["E"]),
                                              "v": "{:.6E}".format(v["v"]),
                                              "beta": "{:.6E}".format(v["beta"])})
            ecm_mats[-1].addBlock(branch=2, btype='solid', mtype='ellipsoidal fiber distribution',
                                  attributes={"ksi": "{:.6E}, {:.6E}, {:.6E}".format(v["ksi"][0],
                                                                                     v["ksi"][1],
                                                                                     v["ksi"][2]),
                                              "beta": "{:.6E}, {:.6E}, {:.6E}".format(2.0,
                                                                                      2.0,
                                                                                      2.0)})
            ecm_mats[-1].addBlock(branch=1, btype='permeability', mtype='perm-Holmes-Mow',
                                  attributes={"perm": "{:.6E}".format(v["permeability"]),
                                              "M": "{:.6E}".format(v["m"]),
                                              "alpha": "{:.6E}".format(v["alpha"])})

            fe_model.addMaterial(ecm_mats[-1])

        # PCM Material
        pcm_mats = []
        for k, v in self.pcm_materials.items():
            mname = 'PCM {:d}'.format(k)
            pcm_mats.append(febio.MatDef(matid=k, mname=mname, mtype='biphasic',
                                         elsets=mname, attributes={'phi0': '{:6E}'.format(v['phi'])}))
            pcm_mats[-1].addBlock(branch=1, btype='solid', mtype='Holmes-Mow',
                                  attributes={"E": "{:.6E}".format(v["E"]),
                                              "v": "{:.6E}".format(v["v"]),
                                              "beta": "{:.6E}".format(v["beta"])})
            pcm_mats[-1].addBlock(branch=1, btype='permeability', mtype='perm-Holmes-Mow',
                                  attributes={"perm": "{:.6E}".format(v["permeability"]),
                                              "M": "{:.6E}".format(v["m"]),
                                              "alpha": "{:.6E}".format(v["alpha"])})

            fe_model.addMaterial(pcm_mats[-1])


        fe_model.addGeometry(mesh=fe_mesh, mats=[cell_mat, mem_mat] + ecm_mats + pcm_mats)

        # Boundary Conditions
        fe_model.addLoadCurve(lc='1', lctype='linear', points=np.ravel(
            list(zip(self.macro_results.TIME, self.macro_results.TIME))))
        boundary = febio.Boundary()
        nids = self.surface_nodes
        dx, dy, dz, p = self._interpBCs(vtkmesh)
        cnt = 2
        for i in range(dx.shape[0]):
            fe_model.addLoadCurve(
                lc='{:d}'.format(cnt),
                lctype='smooth',
                points=np.ravel(list(zip(self.macro_results.TIME, dx[i, :]))))
            boundary.addPrescribed(nodeid=nids[i] + 1, scale=1.0, dof='x', lc='{:d}'.format(cnt))
            cnt += 1
            fe_model.addLoadCurve(
                lc='{:d}'.format(cnt),
                lctype='smooth',
                points=np.ravel(list(zip(self.macro_results.TIME, dy[i, :]))))
            boundary.addPrescribed(nodeid=nids[i] + 1, scale=1.0, dof='y', lc='{:d}'.format(cnt))
            cnt += 1
            fe_model.addLoadCurve(
                lc='{:d}'.format(cnt),
                lctype='smooth',
                points=np.ravel(list(zip(self.macro_results.TIME, dz[i, :]))))
            boundary.addPrescribed(nodeid=nids[i] + 1, scale=1.0, dof='z', lc='{:d}'.format(cnt))
            cnt += 1
            fe_model.addLoadCurve(
                lc='{:d}'.format(cnt),
                lctype='smooth',
                points=np.ravel(list(zip(self.macro_results.TIME, p[i, :]))))
            boundary.addPrescribed(nodeid=nids[i] + 1, scale=1.0, dof='p', lc='{:d}'.format(cnt))
            cnt += 1
        fe_model.addBoundary(boundary)

        ctrl = febio.Control()
        ctrl.setAttributes({'title': model_name,
                            'time_steps': '{:d}'.format(np.ceil(
                                self.macro_results.TIME[-1] / self.macro_results.TIME[1]).astype(int)),
                            'step_size': '{:12.6f}'.format(self.macro_results.TIME[1]),
                            'time_stepper': {'aggressiveness': '0', 'dtmin': '0.01',
                                             'dtmax': 'lc=1', 'max_retries': '10', 'opt_iter': '10'},
                            'max_ups': '0', 'max_refs': '10', 'dtol': '0.001', 'ptol': '0.01', 'lstol': '0.9',
                            'plot_level': 'PLOT_MUST_POINTS'})
        fe_model.addControl(ctrl, step=0)
        fe_model.addOutput(output=["shell strain",
                                   "Lagrange strain",
                                   "effective shell fluid pressure",
                                   "relative volume"])

        fe_model.writeModel()

    def _interpBCs(self, micro_mesh):
        nids = self.surface_nodes
        macro_results = self.macro_results
        macro_data = self.macro_data
        disp_bcs_x = np.zeros((nids.size, len(macro_results.TIME)))
        disp_bcs_y = np.zeros((nids.size, len(macro_results.TIME)))
        disp_bcs_z = np.zeros((nids.size, len(macro_results.TIME)))
        pressure_bcs = np.zeros((nids.size, len(macro_results.TIME)))
        for i, dummy in enumerate(macro_results.TIME):
            displacement = np.zeros((len(macro_results.NodeData.keys()), 3), float)
            pressure = np.zeros((len(macro_results.NodeData.keys()), 1), float)
            for j, k in enumerate(macro_results.NodeData.keys()):
                displacement[j, :] = macro_results.NodeData[k]['displacement'][i, :]
                pressure[j, 0] = macro_results.NodeData[k]['effective fluid pressure'][i, 0]
            arr1 = numpy_support.numpy_to_vtk(displacement, deep=True, array_type=vtk.VTK_FLOAT)
            arr1.SetName('Displacement')
            arr1.SetNumberOfComponents(3)
            macro_data.GetPointData().AddArray(arr1)

            arr2 = numpy_support.numpy_to_vtk(pressure, deep=True, array_type=vtk.VTK_FLOAT)
            arr2.SetName('Pressure')
            arr2.SetNumberOfComponents(1)
            macro_data.GetPointData().AddArray(arr2)

            probe = vtk.vtkProbeFilter()
            probe.SetSourceData(macro_data)
            probe.SetInputData(micro_mesh)
            probe.PassPointArraysOn()
            probe.Update()
            disp_bcs = numpy_support.vtk_to_numpy(
                probe.GetOutput().GetPointData().GetArray('Displacement'))[nids, :]
            disp_bcs -= np.mean(disp_bcs, axis=0)
            pressure_bcs[:, i] = numpy_support.vtk_to_numpy(probe.GetOutput().GetPointData().GetArray('Pressure'))[nids]
            disp_bcs_x[:, i] = disp_bcs[:, 0]
            disp_bcs_y[:, i] = disp_bcs[:, 1]
            disp_bcs_z[:, i] = disp_bcs[:, 2]
        return disp_bcs_x, disp_bcs_y, disp_bcs_z, pressure_bcs


if __name__ == "__main__":
    if len(sys.argv) == 1:
        Model()
    else:
        Model(config=sys.argv[-1])
