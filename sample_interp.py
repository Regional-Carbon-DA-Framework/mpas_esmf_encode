import ESMF
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt

import ESMF.util.helpers as helpers
import ESMF.api.constants as constants

from datetime import datetime, timedelta

if __name__=='__main__':

    fName='sfcf000.nc'
    with nc.Dataset(fName, 'r') as infile:
        lonGrid=infile['lon'][()]
        latGrid=infile['lat'][()]
        varGrid=infile['tmpsfc'][0]

    gausGrid = ESMF.Grid(np.array([lonGrid.shape[0], lonGrid.shape[1]]), 
                        coord_sys=ESMF.CoordSys.SPH_DEG,
                        staggerloc=ESMF.StaggerLoc.CENTER,
                        num_peri_dims=1, periodic_dim=0, pole_dim=1)
    gausGridCoordLat = gausGrid.get_coords(1)
    gausGridCoordLon = gausGrid.get_coords(0)

    gausGridCoordLon[:] = lonGrid
    gausGridCoordLat[:] = latGrid

    gausField = ESMF.Field(gausGrid, name='gausField')
    gausField.data[()]=varGrid
        
    meshFile='mpas_esmf.nc'
    cellGrid=ESMF.Mesh(filename=meshFile, filetype=ESMF.FileFormat.ESMFMESH)    
    cellField=ESMF.Field(cellGrid, name='cellLocs', meshloc=1)
    cellField.data[...]=-999.

    gaus2cell=ESMF.Regrid(gausField, cellField,
                          regrid_method=ESMF.RegridMethod.BILINEAR,
                          unmapped_action=ESMF.UnmappedAction.IGNORE)
    cellField=gaus2cell(gausField, cellField)
    varCell=cellField.data.copy()

    latCell=cellGrid.get_coords(1, meshloc=1).copy()
    lonCell=cellGrid.get_coords(0, meshloc=1).copy()
    lonCell[lonCell<0]+=360

    from scipy.interpolate import griddata

    x, y, z=lonCell, latCell, varCell
    varGrid=griddata(
        np.array([x, y]).T, z,
        np.array([lonGrid.flatten(), latGrid.flatten()]).T,
        method='linear').reshape(*lonGrid.shape)

    clevs=np.arange(220, 340+1.E-6, 20)
    plt.contourf(lonGrid, latGrid, varGrid, cmap='jet', extend='both', levels=clevs)
    plt.colorbar()
    plt.savefig('test')
    
