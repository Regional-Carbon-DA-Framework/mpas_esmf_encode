import numpy as np
import netCDF4 as nc

import xarray as xr
import dask.array as da
import os
from datetime import datetime

def center2corner(latCT, lonCT):
	assert len(latCT.shape)==2 and len(lonCT.shape)==2, '2D lat/lon arrays exepcted in center2corner'
	assert latCT.shape[0]>latCT.shape[1], 'nlon=latCT.shape[0] and nlat=latCT.shape[1]'

	center_lat2d = da.from_array(latCT)
	center_lon2d = da.from_array(lonCT)

	center_lat2d_ext = da.from_array(np.pad(center_lat2d.compute(), (1,1),  mode='reflect', reflect_type='odd'))
	ur = (center_lat2d_ext[1:-1,1:-1]+
          center_lat2d_ext[0:-2,1:-1]+
          center_lat2d_ext[1:-1,2:]+
          center_lat2d_ext[0:-2,2:])/4.0
	ul = (center_lat2d_ext[1:-1,1:-1]+
          center_lat2d_ext[0:-2,1:-1]+
          center_lat2d_ext[1:-1,0:-2]+
          center_lat2d_ext[0:-2,0:-2])/4.0
	ll = (center_lat2d_ext[1:-1,1:-1]+
          center_lat2d_ext[1:-1,0:-2]+
          center_lat2d_ext[2:,1:-1]+
          center_lat2d_ext[2:,0:-2])/4.0
	lr = (center_lat2d_ext[1:-1,1:-1]+
          center_lat2d_ext[1:-1,2:]+
          center_lat2d_ext[2:,1:-1]+
          center_lat2d_ext[2:,2:])/4.0
	corner_lat = da.stack([ul.T.reshape((-1,)).T, ll.T.reshape((-1,)).T, lr.T.reshape((-1,)).T, ur.T.reshape((-1,)).T], axis=1)

	center_lon2d_ext = da.from_array(np.pad(center_lon2d.compute(), (1,1),  mode='reflect', reflect_type='odd'))
	ur = (center_lon2d_ext[1:-1,1:-1]+
          center_lon2d_ext[0:-2,1:-1]+
          center_lon2d_ext[1:-1,2:]+
          center_lon2d_ext[0:-2,2:])/4.0
	ul = (center_lon2d_ext[1:-1,1:-1]+
          center_lon2d_ext[0:-2,1:-1]+
          center_lon2d_ext[1:-1,0:-2]+
          center_lon2d_ext[0:-2,0:-2])/4.0
	ll = (center_lon2d_ext[1:-1,1:-1]+
          center_lon2d_ext[1:-1,0:-2]+
          center_lon2d_ext[2:,1:-1]+
          center_lon2d_ext[2:,0:-2])/4.0
	lr = (center_lon2d_ext[1:-1,1:-1]+
          center_lon2d_ext[1:-1,2:]+
          center_lon2d_ext[2:,1:-1]+
          center_lon2d_ext[2:,2:])/4.0
	corner_lon = da.stack([ul.T.reshape((-1,)).T, ll.T.reshape((-1,)).T, lr.T.reshape((-1,)).T, ur.T.reshape((-1,)).T], axis=1)

	return corner_lat.compute(), corner_lon.compute()

def write_to_scrip(filename, center_lat, center_lon, corner_lat, corner_lon, mask):
    """
    Writes SCRIP grid definition to file
    dask array doesn't support order='F' for Fortran-contiguous (row-major) order
    the workaround is to arr.T.reshape.T
    """
    # create new dataset for output 
    out = xr.Dataset()

    out['grid_dims'] = xr.DataArray(np.array(center_lat.shape, dtype=np.int32), 
                                    dims=('grid_rank',)) 
    out.grid_dims.encoding = {'dtype': np.int32}

    out['grid_center_lat'] = xr.DataArray(center_lat.T.reshape((-1,)).T, 
                                          dims=('grid_size'),
                                          attrs={'units': 'degrees'})

    out['grid_center_lon'] = xr.DataArray(center_lon.T.reshape((-1,)).T, 
                                          dims=('grid_size'),
                                          attrs={'units': 'degrees'})

    out['grid_corner_lat'] = xr.DataArray(corner_lat.T.reshape((4, -1)).T,
                                          dims=('grid_size','grid_corners'),
                                          attrs={'units': 'degrees'})

    out['grid_corner_lon'] = xr.DataArray(corner_lon.T.reshape((4, -1)).T,
                                          dims=('grid_size','grid_corners'),
                                          attrs={'units': 'degrees'})

    out['grid_imask'] = xr.DataArray(mask.T.reshape((-1,)).T, 
                                     dims=('grid_size'),
                                     attrs={'units': 'unitless'})
    out.grid_imask.encoding = {'dtype': np.int32}

    # force no '_FillValue' if not specified
    for v in out.variables:
        if '_FillValue' not in out[v].encoding:
            out[v].encoding['_FillValue'] = None

    # add global attributes
    out.attrs = {'title': 'Rectangular grid with {} dimension'.format('x'.join(list(map(str,center_lat.shape)))),
                 'created_by': os.path.basename(__file__),
                 'date_created': '{}'.format(datetime.now()),
                 'conventions': 'SCRIP',
                }

    # write output file
    if filename is not None:
        print('Writing {} ...'.format(filename))
        out.to_netcdf(filename)    

if __name__=='__main__':

	with nc.Dataset('sfcf000.nc', 'r') as infile:
		latCT=infile['lat'][()].T  # convert to shape [nlon, nlat]
		lonCT=infile['lon'][()].T

	latCorner, lonCorner=center2corner(latCT, lonCT)
	mask=latCT*0.+1
	write_to_scrip('scrip_gaussian.nc', latCT, lonCT, latCorner, lonCorner, mask)

