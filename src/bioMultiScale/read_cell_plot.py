import sys
import os
import xml.etree.ElementTree as ET

import febio
import vtk
from vtk.util import numpy_support
import numpy as np

def __indent(elem,level):
    i = '\n' + level*'  '
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + '  '
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for child in elem:
                __indent(child, level+1)
            if not child.tail or not child.tail.strip():
                child.tail = i
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

def main(plotname, time):
    plt = febio.FebPlt(plotname)
    pathname = os.path.dirname(plotname)
    basename = os.path.basename(plotname).replace('.xplt', '_vtk_files')
    outputdir = os.path.join(pathname, basename)
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)
    if time[0] == 'all':
        time = plt.TIME[1:]

    root1 = ET.Element("VTKFile", type="Collection", version="0.1")
    root2 = ET.Element("VTKFile", type="Collection", version="0.1")
    collection1 = ET.SubElement(root1, "Collection")
    collection2 = ET.SubElement(root2, "Collection")
    for t in time:
        state = int(np.argwhere(np.abs(np.array(plt.TIME) - float(t)) < 1e-7))
        time_string = "{:7.5f}".format(plt.TIME[state])
        name = os.path.join(outputdir, plotname.replace('.xplt', '_{:03d}'.format(state)))
        writeToVTK(plt, ['stress', 'displacement', 'effective fluid pressure', 'shell strain', 'Lagrange strain', 'relative volume', 'fluid flux'], state, name)
        ET.SubElement(collection1, "DataSet", timestep=time_string, file=os.path.basename(name) + ".vtu")
        ET.SubElement(collection2, "DataSet", timestep=time_string, file=os.path.basename(name) + ".vtp")
    # Assemble XML tree
    tree1 = ET.ElementTree(root1)
    tree2 = ET.ElementTree(root2)
    
    level = 0
    elem = tree1.getroot()
    __indent(elem,level)

    level = 0
    elem = tree2.getroot()
    __indent(elem,level)

    #Write XML file
    tree1.write(os.path.join(outputdir, "solid_data.pvd"))
    tree2.write(os.path.join(outputdir, "shell_data.pvd"))
    

def getPrincipals(v):
    pvs = [np.zeros(v.shape[0], dtype=float),
            np.zeros(v.shape[0], dtype=float),
            np.zeros(v.shape[0], dtype=float)]
    pv_dirs = [np.zeros((v.shape[0], 3), dtype=float),
                np.zeros((v.shape[0], 3), dtype=float),
                np.zeros((v.shape[0], 3), dtype=float)]
    for i in range(v.shape[0]):
        l, w = np.linalg.eigh(np.array([[v[i, 0], v[i, 3], v[i, 5]],
                                        [v[i, 3], v[i, 1], v[i, 4]],
                                        [v[i, 5], v[i, 4], v[i, 2]]]))
        idx = np.argsort(l)
        l = l[idx]
        w = w[:, idx]
        for j in range(3):
            pvs[j][i] = l[j]
            pv_dirs[j][i,:] = w[:,j]
    return pvs, pv_dirs

def writeToVTK(plt, variables, state, name):
    element_variables = ['stress',
                         'relative volume',
                         'Lagrange strain',
                         'fluid flux']
    node_variables = ['displacement',
                      'effective fluid pressure',]
    shell_element_variables = ['shell strain']
    variable_lengths = {'displacement': 3,
                        'effective fluid pressure': 1,
                        'fluid flux': 3, 
                        'stress': 6,
                        'shell strain': 6,
                        'relative volume': 1,
                        'Lagrange strain': 6}

    mesh = vtk.vtkUnstructuredGrid()
    # define nodes as VTK point array
    nodearray = numpy_support.numpy_to_vtk(plt.NODE_SECTION.ravel(), deep=True, array_type=vtk.VTK_DOUBLE)
    nodearray.SetNumberOfComponents(3)
    nodes = vtk.vtkPoints()
    nodes.SetData(nodearray)
    # Add to nodes to VTK unstructured grid
    mesh.SetPoints(nodes)
    t = vtk.vtkDoubleArray()
    t.SetName("TIME")
    t.SetNumberOfTuples(1)
    t.SetTuple1(0, plt.TIME[state])
    mesh.GetFieldData().AddArray(t)

    shell_mesh = vtk.vtkPolyData()
    shell_mesh.SetPoints(nodes)
    shell_mesh.GetFieldData().AddArray(t)

    num_solid_elements = 0
    num_shell_elements = 0
    hex_elements = False
    tet_elements = False
    for d in plt.DOMAIN_SECTION:
        etype = d['DOMAIN_HEADER']['ELEM_TYPE']
        if etype == 'TET4':
            num_solid_elements += d['DOMAIN_HEADER']['ELEMENTS']
            tet_elements = True
        elif etype == 'HEX8':
            num_solid_elements += d['DOMAIN_HEADER']['ELEMENTS']
            hex_elements = True
        elif etype == 'TRI3':
            num_shell_elements += d['DOMAIN_HEADER']['ELEMENTS']
    if hex_elements and tet_elements:
        raise SystemError('Mixed meshes e.g. hexahedron and tetrahedron are not supported at this time.')

    solid_element_set_ids = np.zeros(num_solid_elements, dtype=np.uint8)
    if tet_elements:
        solid_element_array = np.zeros((solid_element_set_ids.size, 5), dtype=int)
    elif hex_elements:
        solid_element_array = np.zeros((solid_element_set_ids.size, 9), dtype=int)
    shell_element_set_ids = np.zeros(num_shell_elements, dtype=np.uint8)
    shell_element_array = np.zeros((shell_element_set_ids.size, 4), dtype=int)
    # loop through domain sections extracting elements and material ids
    for d in plt.DOMAIN_SECTION:
        mat_id = d['DOMAIN_HEADER']['MAT_ID']
        if d['DOMAIN_HEADER']['ELEM_TYPE'] == 'TET4':
            for e in d['ELEMENT_LIST']:
                solid_element_set_ids[e[0] - 1] = mat_id
                solid_element_array[e[0] - 1, :] = [4, e[1], e[2], e[3], e[4]]
        elif d['DOMAIN_HEADER']['ELEM_TYPE'] == 'HEX8':
            for e in d['ELEMENT_LIST']:
                solid_element_set_ids[e[0] - 1] = mat_id
                solid_element_array[e[0] - 1, :] = [8, e[1], e[2], e[3], e[4],
                                                    e[5], e[6], e[7], e[8]]
        elif d['DOMAIN_HEADER']['ELEM_TYPE'] == 'TRI3':
            for e in d['ELEMENT_LIST']:
                shell_element_set_ids[e[0] - 1 - num_solid_elements] = mat_id
                shell_element_array[e[0] - 1 - num_solid_elements, :] = [3, e[1], e[2], e[3]]
    arr = numpy_support.numpy_to_vtk(solid_element_array.ravel(), deep=True, array_type=vtk.VTK_ID_TYPE)
    elements = vtk.vtkCellArray()
    if tet_elements:
        elements.SetCells(solid_element_array.size // 5, arr)
        mesh.SetCells(10, elements)
    elif hex_elements:
        elements.SetCells(solid_element_array.size // 9, arr)
        mesh.SetCells(12, elements)


    element_set = numpy_support.numpy_to_vtk(solid_element_set_ids.ravel(), deep=True, array_type=vtk.VTK_CHAR)
    element_set.SetNumberOfComponents(1)
    element_set.SetName('Material ID')
    mesh.GetCellData().AddArray(element_set)

    arr = numpy_support.numpy_to_vtk(shell_element_array.ravel(), deep=True, array_type=vtk.VTK_ID_TYPE)
    shell_elements = vtk.vtkCellArray()
    shell_elements.SetCells(shell_element_array.size // 4, arr)
    shell_mesh.SetPolys(shell_elements)

    for v in variables:
        if v in element_variables:
            data_array = np.zeros((num_solid_elements, variable_lengths[v]), dtype=float)
            for i in range(num_solid_elements):
                data_array[i, :] = plt.ElementData[i + 1][v][state]
            arr = numpy_support.numpy_to_vtk(data_array.ravel(), deep=True, array_type=vtk.VTK_DOUBLE)
            arr.SetNumberOfComponents(variable_lengths[v])
            arr.SetName(v)
            mesh.GetCellData().AddArray(arr)
            if v == 'stress' or v == 'Lagrange strain':
                pvs, pv_dirs = getPrincipals(data_array)
                for i in range(3):
                    arr = numpy_support.numpy_to_vtk(pvs[i].ravel(), deep=True, array_type=vtk.VTK_DOUBLE)
                    arr.SetNumberOfComponents(1)
                    arr.SetName("Principal {} {:d}".format(v, 3 - i))
                    mesh.GetCellData().AddArray(arr)

                    arr = numpy_support.numpy_to_vtk(pv_dirs[i].ravel(), deep=True, array_type=vtk.VTK_DOUBLE)
                    arr.SetNumberOfComponents(3)
                    arr.SetName("Principal {} Direction {:d}".format(v, 3 - i))
                    mesh.GetCellData().AddArray(arr)

        elif v in node_variables:
            data_array = np.zeros((plt.NODES, variable_lengths[v]), dtype=float)
            for i in range(plt.NODES):
                data_array[i, :] = plt.NodeData[i + 1][v][state]
            arr = numpy_support.numpy_to_vtk(data_array.ravel(), deep=True, array_type=vtk.VTK_DOUBLE)
            arr.SetNumberOfComponents(variable_lengths[v])
            arr.SetName(v)
            mesh.GetPointData().AddArray(arr)
            shell_mesh.GetPointData().AddArray(arr)
        elif v in shell_element_variables:
            data_array = np.zeros((num_shell_elements, variable_lengths[v]), dtype=float)
            for i in range(num_shell_elements):
                data_array[i, :] = plt.ElementData[i + 1 + num_solid_elements][v][state]
            arr = numpy_support.numpy_to_vtk(data_array.ravel(), deep=True, array_type=vtk.VTK_DOUBLE)
            arr.SetNumberOfComponents(variable_lengths[v])
            arr.SetName(v)
            shell_mesh.GetCellData().AddArray(arr)
            if v == 'shell strain':
                pvs, pv_dirs = getPrincipals(data_array)
                for i in range(3):
                    arr = numpy_support.numpy_to_vtk(pvs[i].ravel(), deep=True, array_type=vtk.VTK_DOUBLE)
                    arr.SetNumberOfComponents(1)
                    arr.SetName("Principal {} {:d}".format(v, 3 - i))
                    shell_mesh.GetCellData().AddArray(arr)

                    arr = numpy_support.numpy_to_vtk(pv_dirs[i].ravel(), deep=True, array_type=vtk.VTK_DOUBLE)
                    arr.SetNumberOfComponents(3)
                    arr.SetName("Principal {} Direction {:d}".format(v, 3 - i))
                    shell_mesh.GetCellData().AddArray(arr)

    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetInputData(mesh)
    writer.SetFileName(name + '.vtu')
    writer.Write()

    clean_vertices = vtk.vtkCleanPolyData()
    clean_vertices.SetInputData(shell_mesh)
    clean_vertices.Update()
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetInputData(clean_vertices.GetOutput())
    writer.SetFileName(name + '.vtp')
    writer.Write()


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])
