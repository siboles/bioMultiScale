from paraview.simple import *

layout = GetLayout()

layout.SplitViewHorizontal(view = GetActiveView(), fraction = 2.0 / 3.0)
layout.SplitViewHorizontal(view = GetActiveView(), fraction = 1.0 / 2.0)

