import sys

import vtk
import numpy as np
from vtk.util import numpy_support
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from weighted import quantile

sns.set('paper')


def writeToVTK(name='dummy', data=None):
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName('{}.vtu'.format(name))
    writer.SetInputData(data)
    writer.Write()


class Data():
    def __init__(self, filepaths=None, shortnames=None, elset_mapping={1: "Cell", 3: "PCM", 4: "ECM"}):
        if filepaths is None:
            raise TypeError("filepath must be set to path of vtk polydata file during object creation.")
        if not isinstance(filepaths, list):
            filepaths = [filepaths]
        self.elset_mapping = elset_mapping
        self.filepaths = filepaths
        self.shortnames = shortnames
        self.element_sets = {}
        for v in self.elset_mapping.values():
            self.element_sets[v] = {}
        self.polydatas = {}
        self._readPolydata()
        self._getElementMetrics()
        self._defineElementSetsByMaterialID()
        self._defineElementSetsByLocation()

    def _readPolydata(self):
        for i, f in enumerate(self.filepaths):
            reader = vtk.vtkXMLUnstructuredGridReader()
            reader.SetFileName(f)
            reader.Update()
            if self.shortnames is not None:
                key = self.shortnames[i]
            else:
                key = f
            self.polydatas[key] = reader.GetOutput()

    def _getElementMetrics(self):
        for k, p in self.polydatas.items():
            # create a deformed polydata object for deformed metrics
            warp = vtk.vtkWarpVector()
            warp.SetInputData(p)
            warp.SetScaleFactor(1)
            p.GetPointData().SetActiveVectors('displacement')
            warp.Update()
            warped = warp.GetOutput()

            # Use the MeshQuality filter with the metric set to volume to quickly get cell volumes
            # for reference and deformed cases.
            mesh_quality = vtk.vtkMeshQuality()
            mesh_quality.SetTetQualityMeasureToVolume()
            mesh_quality.SetInputData(p)
            mesh_quality.Update()
            r_volumes = mesh_quality.GetOutput().GetCellData().GetArray("Quality")
            r_volumes.SetName('Reference Volume')
            p.GetCellData().AddArray(r_volumes)
            mesh_quality2 = vtk.vtkMeshQuality()
            mesh_quality2.SetTetQualityMeasureToVolume()
            mesh_quality2.SetInputData(warped)
            mesh_quality2.Update()
            d_volumes = mesh_quality2.GetOutput().GetCellData().GetArray("Quality")
            d_volumes.SetName('Deformed Volume')
            p.GetCellData().AddArray(d_volumes)

            # CellCenters filter generates new polydata with points at centroids. Grab these points
            # data array and add it to polydata for reference and deformed.
            cell_centers = vtk.vtkCellCenters()
            cell_centers.VertexCellsOff()
            cell_centers.SetInputData(p)
            cell_centers.Update()
            r_centroids = cell_centers.GetOutput().GetPoints().GetData()
            r_centroids.SetName('Reference Centroid')
            p.GetCellData().AddArray(r_centroids)
            cell_centers2 = vtk.vtkCellCenters()
            cell_centers2.VertexCellsOff()
            cell_centers2.SetInputData(warped)
            cell_centers2.Update()
            d_centroids = cell_centers2.GetOutput().GetPoints().GetData()
            d_centroids.SetName('Deformed Centroid')
            p.GetCellData().AddArray(d_centroids)

    def _defineElementSetsByMaterialID(self):
        for k, v in self.elset_mapping.items():
            for k2, v2 in self.polydatas.items():
                threshold = vtk.vtkThreshold()
                threshold.SetInputData(v2)
                threshold.ThresholdBetween(int(k) - 0.5, int(k) + 0.5)
                threshold.SetInputArrayToProcess(0, 0, 0,
                                                 vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS,
                                                 "Material ID")
                threshold.Update()
                self.element_sets[v][k2] = threshold.GetOutput()

    def _defineElementSetsByLocation(self):
        for k in list(self.element_sets.keys()):
            for k2, v2 in self.element_sets[k].items():
                zrange = v2.GetBounds()[4:]
                for case in ['Top', 'Bottom']:
                    if case == 'Top':
                        bounds = (zrange[1] - 0.25 * (zrange[1] - zrange[0]), zrange[1])
                    elif case == 'Bottom':
                        bounds = (zrange[0], zrange[0] + 0.25 * (zrange[1] - zrange[0]))
                    threshold = vtk.vtkThreshold()
                    threshold.SetInputData(v2)
                    threshold.ThresholdBetween(bounds[0], bounds[1])
                    threshold.SetComponentModeToUseSelected()
                    threshold.SetSelectedComponent(2)
                    threshold.SetInputArrayToProcess(0, 0, 0,
                                                     vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS,
                                                     "Reference Centroid")
                    threshold.Update()
                    key = ' '.join([k, case])
                    if key in self.element_sets.keys():
                        self.element_sets[key][k2] = threshold.GetOutput()
                    else:
                        self.element_sets[key] = {}
                        self.element_sets[key][k2] = threshold.GetOutput()

    def generateBoxPlot(self, setnames=None, variable=None, filename='output'):
        if setnames is None:
            print("Warning: no setnames provided. Assuming all elements and nodes in model.")
        elif not isinstance(setnames, list):
            setnames = [setnames]
        if variable is None or not isinstance(variable, str):
            raise TypeError(variable)
        for s in setnames:
            fig = plt.Figure()
            ax = fig.add_subplot(111)
            quartiles = []
            medians = []
            df1 = pd.DataFrame()
            for k, v in self.element_sets[s].items():
                weights = numpy_support.vtk_to_numpy(v.GetCellData().GetArray('Reference Volume'))
                data = numpy_support.vtk_to_numpy(v.GetCellData().GetArray(variable))
                # idx = np.argwhere(data > 0.1).ravel()
                # weights = weights[idx]
                weights /= np.max(weights)
                # data = data[idx]
                medians.append(quantile(data, weights, 0.5))
                quartiles.append([quantile(data, weights, x) for x in [0.0, 0.25, 0.75, 1.0]])
                df2 = pd.DataFrame({k: np.array(data, dtype="object")})
                df1 = pd.concat([df1, df2], axis=1)
            df1.boxplot(ax=ax, vert=False, grid=False)
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position('top')
            fig.savefig('{}.svg'.format(filename))

    def getElementSetCentroid(self, setnames=None, filename='output'):
        if setnames is None:
            print("Warning: no setnames provided. Assuming all elements and nodes in model.")
        elif not isinstance(setnames, list):
            setnames = [setnames]
        for s in setnames:
            df1 = pd.DataFrame()
            for k, v in self.element_sets[s].items():
                rv = numpy_support.vtk_to_numpy(v.GetCellData().GetArray('Reference Volume'))
                rcentroids = numpy_support.vtk_to_numpy(v.GetCellData().GetArray('Reference Centroid')) * 1000
                rcentroid = np.average(rcentroids, axis=0, weights=rv)
                dv = numpy_support.vtk_to_numpy(v.GetCellData().GetArray('Deformed Volume'))
                dcentroids = numpy_support.vtk_to_numpy(v.GetCellData().GetArray('Deformed Centroid')) * 1000
                dcentroid = np.average(dcentroids, axis=0, weights=dv)
                df2 = pd.DataFrame({' '.join([k, 'Reference Centroid']): rcentroid,
                                    ' '.join([k, 'Deformed Centroid']): dcentroid,
                                    ' '.join([k, 'Difference']): dcentroid - rcentroid})
                df1 = pd.concat([df1, df2], axis=1)
            df1.to_excel('{}.xlsx'.format(filename), sheet_name=s)
            z_displacements = df1.filter(like='Difference', axis=1).iloc[2]
            z_displacements.plot.barh(grid=False)
            plt.show()


if __name__ == '__main__':
    data = Data(filepaths=sys.argv[1:])
