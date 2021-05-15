import sys

from collections import MutableMapping, OrderedDict

import vtk
import febio
from vtk.util import numpy_support
import numpy as np
from scipy.interpolate import UnivariateSpline
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
        # if config is None:
        #    raise ValueError("A configuration file must be provided. Exitting...")

        self.options = FixedDict({
            "Model Name": "tissue",
            "Geometry": FixedDict({
                "origin": [0.0, 0.0, 0.0],
                "dimensions": [1.0, 1.0, 0.5],
                "element_divisions": [20, 20, 10]}),
            "Material Properties": FixedDict({
                "spline_breakpoints": (0.0, 0.4, 0.5, 0.8, 0.9, 1.0),
                "E": (1.1914, 0.6584, 0.4371, 0.1939, 0.0986, 0.0986),
                "v": (0.3972, 0.25652, 0.22135, 0.11584, 0.0455, 0.0455),
                "beta": (0, 0, 0, 0, 0.004470494574063, 0.282884197412423,
                         0.661042516343724, 0.868609534473908,0.649949025480955),
                "phi": (0.2, 0.2, 0.2, 0.2, 0.2, 0.2),
                "permeability": (1.3e-3, 2.5e-3, 3.8e-3, 3.8e-3, 2.7e-3, 2.7e-3),
                "m": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                "alpha": (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
                "ksi1": (10.0, 10.0, 33.3, 33.3, 100.0, 100.0),
                "ksi2": (10.0, 10.0, 33.3, 33.3, 10.0, 10.0),
                "ksi3": (100.0, 100.0, 33.3, 33.3, 10.0, 10.0)}),
            "Load Curves": FixedDict({
                "1": ["step", 0.0, 0.5, 1.0, 0.5,
                      5.0, 2.5, 10.0, 5.0, 20.0, 10.0, 40.0, 20.0,
                      80.0, 40.0, 160.0, 80.0, 200.0, 20.0],
                "2": ["linear", 0.0, 0.0, 1.0, 1.0, 200.0, 1.0]}),
            "Boundary Conditions": FixedDict({
                "Fixed": FixedDict({
                    "Bottom": ["xyz"],
                    "Top": [],
                    "Left": [],
                    "Right": [],
                    "Back": [],
                    "Front": []
                }),
                "Prescribed": FixedDict({
                    "Bottom": [],
                    "Top": ["2", -0.05],
                    "Left": [],
                    "Right": [],
                    "Back": [],
                    "Front": []
                }),
                "Force Loads": FixedDict({
                    "Bottom": [],
                    "Top": [],
                    "Left": [],
                    "Right": [],
                    "Back": [],
                    "Front": []
                }),
                "Pressure Loads": FixedDict({
                    "Bottom": [],
                    "Top": [],
                    "Left": [],
                    "Right": [],
                    "Back": [],
                    "Front": []
                })})})

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

        self._makeMaterialSplines()
        self._makeVTK()
        self._makeFEBio()

    def parseConfig(self):
        """
        Parse configuration file in YAML format.
        """
        with open(self.config) as user:
            user_settings = yaml.load(user, Loader=yaml.FullLoader)

        for k, v in list(user_settings.items()):
            if isinstance(v, dict):
                for k2, v2 in list(v.items()):
                    self.options[str(k)][str(k2)] = v2
            else:
                self.options[str(k)] = v

    def _makeMaterialSplines(self):
        breakpoints = self.options["Material Properties"]["spline_breakpoints"]
        self.material_splines = {}
        for p in ("E", "v", "beta", "phi", "permeability",
                  "m", "alpha", "ksi1", "ksi2", "ksi3"):
            self.material_splines[p] = UnivariateSpline(np.array(breakpoints),
                                                        np.array(self.options["Material Properties"][p]),
                                                        k=1, s=0)

    def _assignNodeSetId(self, irange, jrange, krange, name, idx):
        for i in irange:
            for j in jrange:
                for k in krange:
                    pid = self.vtkmesh.ComputePointId([i, j, k])
                    self.vtkmesh.GetPointData().GetArray(name).SetTuple1(pid, idx)

    def _makeVTK(self):
        # number of elements in each dimension
        nelm = self.options["Geometry"]["element_divisions"]
        # box dimensions
        bdim = self.options["Geometry"]["dimensions"]
        # box origin
        origin = self.options["Geometry"]["origin"]
        true_z_edge = bdim[2] / nelm[2]
        depths = np.linspace(origin[2] + true_z_edge / 2.0,
                             origin[2] + bdim[2], nelm[2], endpoint=False)
        depths /= np.max(depths)
        self.vtkmesh = vtk.vtkRectilinearGrid()
        self.vtkmesh.SetExtent(0, nelm[0],
                               0, nelm[1],
                               0, nelm[2])
        self.vtkmesh.SetXCoordinates(numpy_support.numpy_to_vtk(
            np.linspace(origin[0], origin[0] + bdim[0],
                        nelm[0] + 1, endpoint=True).ravel(), deep=True,
            array_type=vtk.VTK_DOUBLE))
        self.vtkmesh.SetYCoordinates(numpy_support.numpy_to_vtk(
            np.linspace(origin[1], origin[1] + bdim[1],
                        nelm[1] + 1, endpoint=True).ravel(), deep=True,
            array_type=vtk.VTK_DOUBLE))
        self.vtkmesh.SetZCoordinates(numpy_support.numpy_to_vtk(
            np.linspace(origin[2], origin[2] + bdim[2],
                        nelm[2] + 1, endpoint=True).ravel(), deep=True,
            array_type=vtk.VTK_DOUBLE))
        for i in ['Left', 'Right', 'Front', 'Back', 'Bottom', 'Top']:
            nodeSetIDs = vtk.vtkUnsignedCharArray()
            nodeSetIDs.SetNumberOfComponents(1)
            nodeSetIDs.SetNumberOfTuples(self.vtkmesh.GetNumberOfPoints())
            nodeSetIDs.Fill(0)
            nodeSetIDs.SetName(i)
            self.vtkmesh.GetPointData().AddArray(nodeSetIDs)
            if i == 'Left':
                self._assignNodeSetId([0], range(nelm[1] + 1), range(nelm[2] + 1), i, 1)
            if i == 'Right':
                self._assignNodeSetId([nelm[0]], range(nelm[1] + 1), range(nelm[2] + 1), i, 1)
            if i == 'Front':
                self._assignNodeSetId(range(nelm[0] + 1), [0], range(nelm[2] + 1), i, 1)
            if i == 'Back':
                self._assignNodeSetId(range(nelm[0] + 1), [nelm[1]], range(nelm[2] + 1), i, 1)
            if i == 'Bottom':
                self._assignNodeSetId(range(nelm[0] + 1), range(nelm[1] + 1), [0], i, 1)
            if i == 'Top':
                self._assignNodeSetId(range(nelm[0] + 1), range(nelm[1] + 1), [nelm[2]], i, 1)

        elementSetIDs = vtk.vtkUnsignedCharArray()
        elementSetIDs.SetNumberOfComponents(1)
        elementSetIDs.SetNumberOfTuples(self.vtkmesh.GetNumberOfCells())
        elementSetIDs.SetName("ElementSetID")
        self.vtkmesh.GetCellData().AddArray(elementSetIDs)

        modulusArray = vtk.vtkFloatArray()
        modulusArray.SetNumberOfComponents(1)
        modulusArray.SetNumberOfTuples(self.vtkmesh.GetNumberOfCells())
        modulusArray.SetName("E")
        self.vtkmesh.GetCellData().AddArray(modulusArray)

        poissonArray = vtk.vtkFloatArray()
        poissonArray.SetNumberOfComponents(1)
        poissonArray.SetNumberOfTuples(self.vtkmesh.GetNumberOfCells())
        poissonArray.SetName("v")
        self.vtkmesh.GetCellData().AddArray(poissonArray)

        betaArray = vtk.vtkFloatArray()
        betaArray.SetNumberOfComponents(1)
        betaArray.SetNumberOfTuples(self.vtkmesh.GetNumberOfCells())
        betaArray.SetName("beta")
        self.vtkmesh.GetCellData().AddArray(betaArray)

        permArray = vtk.vtkFloatArray()
        permArray.SetNumberOfComponents(1)
        permArray.SetNumberOfTuples(self.vtkmesh.GetNumberOfCells())
        permArray.SetName("permeability")
        self.vtkmesh.GetCellData().AddArray(permArray)

        mArray = vtk.vtkFloatArray()
        mArray.SetNumberOfComponents(1)
        mArray.SetNumberOfTuples(self.vtkmesh.GetNumberOfCells())
        mArray.SetName("m")
        self.vtkmesh.GetCellData().AddArray(mArray)

        alphaArray = vtk.vtkFloatArray()
        alphaArray.SetNumberOfComponents(1)
        alphaArray.SetNumberOfTuples(self.vtkmesh.GetNumberOfCells())
        alphaArray.SetName("alpha")
        self.vtkmesh.GetCellData().AddArray(alphaArray)

        phiArray = vtk.vtkFloatArray()
        phiArray.SetNumberOfComponents(1)
        phiArray.SetNumberOfTuples(self.vtkmesh.GetNumberOfCells())
        phiArray.SetName("phi")
        self.vtkmesh.GetCellData().AddArray(phiArray)

        ksiArray = vtk.vtkFloatArray()
        ksiArray.SetNumberOfComponents(3)
        ksiArray.SetNumberOfTuples(self.vtkmesh.GetNumberOfCells())
        ksiArray.SetName("ksi")
        self.vtkmesh.GetCellData().AddArray(ksiArray)

        for i in range(nelm[0]):
            for j in range(nelm[1]):
                for k in range(nelm[2]):
                    cellID = self.vtkmesh.ComputeCellId([i, j, k])
                    self.vtkmesh.GetCellData().GetArray("ElementSetID").SetTuple1(cellID, k + 1)
                    self.vtkmesh.GetCellData().GetArray("E").SetTuple1(cellID, self.material_splines["E"](depths[k]))
                    self.vtkmesh.GetCellData().GetArray("v").SetTuple1(cellID, self.material_splines["v"](depths[k]))
                    self.vtkmesh.GetCellData().GetArray("beta").SetTuple1(cellID, self.material_splines["beta"](depths[k]))
                    self.vtkmesh.GetCellData().GetArray("ksi").SetTuple3(
                        cellID, self.material_splines["ksi1"](depths[k]),
                        self.material_splines["ksi2"](depths[k]),
                        self.material_splines["ksi3"](depths[k]))
                    self.vtkmesh.GetCellData().GetArray("permeability").SetTuple1(
                        cellID, self.material_splines["permeability"](depths[k]))
                    self.vtkmesh.GetCellData().GetArray("m").SetTuple1(
                        cellID, self.material_splines["m"](depths[k]))
                    self.vtkmesh.GetCellData().GetArray("alpha").SetTuple1(
                        cellID, self.material_splines["alpha"](depths[k]))
                    self.vtkmesh.GetCellData().GetArray("phi").SetTuple1(
                        cellID, self.material_splines["phi"](depths[k]))
        writer = vtk.vtkXMLRectilinearGridWriter()
        writer.SetFileName(".".join([self.options["Model Name"], "vtr"]))
        writer.SetInputData(self.vtkmesh)
        writer.Write()

    def _makeFEBio(self):
        fe_mesh = febio.MeshDef()
        fe_mesh.nodes = []
        box = self.vtkmesh
        for i in range(box.GetNumberOfPoints()):
            point = [0.0, 0.0, 0.0]
            box.GetPoint(i, point)
            fe_mesh.nodes.append([i + 1] + point)

        fe_mesh.elements = []
        for i in range(box.GetNumberOfCells()):
            pids = vtk.vtkIdList()
            pids.SetNumberOfIds(8)
            box.GetCellPoints(i, pids)
            element = ['hex8', i + 1]
            for j in [0, 1, 3, 2, 4, 5, 7, 6]:
                element.append(pids.GetId(j) + 1)
            fe_mesh.elements.append(element)
        fe_model = febio.Model(modelfile=self.options["Model Name"] + '.feb',
                               steps=[{'Displace': 'biphasic'}])
        element_set_ids = numpy_support.vtk_to_numpy(box.GetCellData().GetArray("ElementSetID"))
        element_sets = np.unique(element_set_ids)
        mats = []
        for i, s in enumerate(element_sets):
            eids = list(np.argwhere(element_set_ids == s).ravel() + 1)
            E = box.GetCellData().GetArray("E").GetTuple(eids[0] - 1)[0]
            v = box.GetCellData().GetArray("v").GetTuple(eids[0] - 1)[0]
            beta = box.GetCellData().GetArray("beta").GetTuple(eids[0] - 1)[0]
            perm = box.GetCellData().GetArray("permeability").GetTuple(eids[0] - 1)[0]
            m = box.GetCellData().GetArray("m").GetTuple(eids[0] - 1)[0]
            alpha = box.GetCellData().GetArray("alpha").GetTuple(eids[0] - 1)[0]
            phi = box.GetCellData().GetArray("phi").GetTuple(eids[0] - 1)[0]
            ksi = box.GetCellData().GetArray("ksi").GetTuple(eids[0] - 1)

            fe_mesh.addElementSet(setname='{}'.format(s), eids=eids)
            mats.append(
                febio.MatDef(matid=i + 1,
                             mname='{}'.format(s), mtype='biphasic',
                             elsets=['{}'.format(s)], attributes={'phi0': '{:.6E}'.format(phi)}))
            mats[-1].addBlock(branch=1, btype='solid', mtype='solid mixture',
                              attributes={'mat_axis': ['vector', '1.0,0.0,0.0', '0.0,1.0,0.0']})
            mats[-1].addBlock(branch=2, btype='solid', mtype='Holmes-Mow',
                              attributes={'E': '{:.6E}'.format(E),
                                          'v': '{:.6E}'.format(v),
                                          'beta': '{:6E}'.format(beta)})
            mats[-1].addBlock(branch=2, btype='solid', mtype='ellipsoidal fiber distribution',
                              attributes={"ksi": "{:.6E}, {:.6E}, {:.6E}".format(ksi[0], ksi[1], ksi[2]),
                                          "beta": "2.0, 2.0, 2.0"})
            mats[-1].addBlock(branch=1, btype='permeability', mtype='perm-Holmes-Mow',
                              attributes={'perm': '{:.6E}'.format(perm),
                                          'M': '{:.6E}'.format(m),
                                          'alpha': '{:.6E}'.format(alpha)})
            fe_model.addMaterial(mat=mats[-1])

        fe_model.addGeometry(mesh=fe_mesh, mats=mats)
        node_sets = ['Left', 'Right', 'Front', 'Back', 'Bottom', 'Top']
        for ns in node_sets:
            set_flags = numpy_support.vtk_to_numpy(box.GetPointData().GetArray(ns))
            fe_mesh.nsets[ns] = list(np.argwhere(set_flags==1).ravel() + 1)

        boundary = febio.Boundary()
        for k,v in self.options['Boundary Conditions']['Fixed'].items():
            if v:
                boundary.addFixed(nset=fe_mesh.nsets[k], dof=v[0])
        for k,v in self.options['Boundary Conditions']['Prescribed'].items():
            if v:
                boundary.addPrescribed(nset=fe_mesh.nsets[k], dof=v[0], lc=v[1], scale=str(v[2]))

        fe_model.addBoundary(boundary=boundary)

        for k, v in self.options["Load Curves"].items():
            fe_model.addLoadCurve(lc=k, lctype=v[0], points=v[1:])

        ctrl = febio.Control()
        ctrl.setAttributes({'title': self.options["Model Name"],
                            'time_steps': '{:d}'.format(
                                np.ceil(self.options["Load Curves"]['1'][-2] /
                                        self.options["Load Curves"]['1'][3]).astype(int)),
                            'step_size': '{:12.6f}'.format(
                                self.options["Load Curves"]['1'][3]),
                            'time_stepper': {'aggressiveness': '0', 'dtmin': '0.01', 'dtmax': 'lc=1', 'max_retries': '10', 'opt_iter': '10'},
                            'max_ups':'0','max_refs':'20','dtol':'0.001','ptol':'0.01','lstol':'0.9'})
        fe_model.addControl(ctrl, step=0)
        fe_model.addOutput(output=["Lagrange strain",
                                   "relative volume"])
        fe_model.writeModel()

    def writeToVTK(self):
        writer = vtk.vtkXMLRectilinearGridWriter()
        writer.SetFileName(self.options["Model Name"] + '.vtr')
        writer.SetInputData(self.vtkmesh)
        writer.Write()

if __name__ == "__main__":
    if len(sys.argv) == 1:
        Model()
    else:
        Model(config=sys.argv[-1])
