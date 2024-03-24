# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 11:36:22 2024

@author: Administrator
"""

import plotly.graph_objects as go
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import matplotlib.image as mpimg
# Function to convert spherical coordinates (lat, lon) to Cartesian coordinates (x, y, z)


def spherical_to_cartesian(r, lat_deg, lon_deg):
    lat_rad = np.radians(lat_deg)
    lon_rad = np.radians(lon_deg)
    x = r * np.cos(lat_rad) * np.cos(lon_rad)
    y = r * np.cos(lat_rad) * np.sin(lon_rad)
    z = r * np.sin(lat_rad)
    return x, y, z

def spherical_to_cartesian_wind(r, lat_deg, lon_deg,height,HGT):
    lat_rad = np.radians(lat_deg)
    lon_rad = np.radians(lon_deg)
    x = (r + height + HGT) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (r + height + HGT) * np.cos(lat_rad) * np.sin(lon_rad)
    z = (r + height + HGT) * np.sin(lat_rad) 
    return x, y, z

def wind_transform_cye(lat, lon,uu,vv,ww):
    u = uu*np.sin(lon)+vv*np.sin(lat)*np.sin(lon)+ww*np.sin(lat)*np.cos(lon)
    v = uu*np.cos(lon)-vv*np.sin(lat)*np.sin(lon)+ww*np.sin(lat)*np.sin(lon)
    w = vv*np.cos(lat)+ww*np.cos(lat)
    return u,v,w

def wind_transform2(lats, lons,u,v,w):
    r_earth=6371

    lat_rad = np.radians(lats)
    lon_rad = np.radians(lons)
    # Calculate Cartesian coordinates for the wind data points assuming they're at the surface of the Earth
    # Here we don't add the altitude yet since the vectors are assumed to originate from the surface.
    x_surface = r_earth * np.cos(lat_rad) * np.cos(lon_rad)
    y_surface = r_earth * np.cos(lat_rad) * np.sin(lon_rad)
    z_surface = r_earth * np.sin(lat_rad)
    
    # Normalise the latitude and longitude vectors (to unit vectors)
    norm_lat = np.sqrt(x_surface**2 + y_surface**2)
    norm_lon = norm_lat
    
    # Calculate the Cartesian vector components
    # Note: These equations depend on u being eastward, v being northward, and w being upward.
    # The transformation accounts for the reference frame change from local NEU to global ECEF.
    x_vector = (-u * y_surface / norm_lon) + (v * x_surface * z_surface / (norm_lat * r_earth)) + (w * x_surface / r_earth)
    y_vector = (u * x_surface / norm_lon) + (v * y_surface * z_surface / (norm_lat * r_earth)) + (w * y_surface / r_earth)
    z_vector = (-v * norm_lat / r_earth) + (w * z_surface / r_earth)
    return x_vector,y_vector,z_vector
##################################################################################################################################
earth_radius = 63.71
spacing = 10
level = 30

# Create a sphere to represent Earth
f1 = r"E:\adaptor.mars.internal-1708945841.711128-26502-11-bd9f757c-a5e8-41ec-99ed-fcb815717c64.nc"
ds = xr.open_dataset(f1)

lons = ds.longitude.data[::spacing]-180.
lats = ds.latitude.data[::spacing]
lon_grid, lat_grid = np.meshgrid(lons, lats)

f2 = "D:\BaiduSyncdisk\honours\earth_relief_30m_g.grd"
ds2 = xr.open_dataset(f2)
HGT = ds2.interp(lon=lons,lat=lats).z.data
HGT[HGT<0]=0 #wind is above sea surface
# Assume 'data' is your 2D numpy array
data = HGT  # Replace this with your actual data
# Normalize the data to 0-1 if it isn't already
data_normalized = (data - np.min(data)) / (np.max(data) - np.min(data))
image_data = (data_normalized * 255).astype(np.uint8)
# Use plt.imshow to plot the data
# Remove axes for a cleaner image
plt.imshow(image_data, cmap='terrain')  # Choose an appropriate color map.
plt.axis('off')  # Turn off the axis.
plt.savefig('D:\BaiduSyncdisk\honours\earth_surface.png', bbox_inches='tight', pad_inches=0)
plt.close()
img_data = mpimg.imread('D:\BaiduSyncdisk\honours\earth_surface.png')

# Calculate the required zoom factors for the latitude and longitude dimensions
zoom_factor_lat = lats.shape[0] / img_data.shape[0]
zoom_factor_lon = lons.shape[0] / img_data.shape[1]
# Perform the resampling
resampled_img_data = zoom(img_data, (zoom_factor_lat, zoom_factor_lon, 1))
##################################################################################################################################
# Radius of the Earth - use a smaller value so the wind vectors are visible outside of the Earth
x_sphere, y_sphere, z_sphere = spherical_to_cartesian(earth_radius, lat_grid, lon_grid)
# Simulated wind data at different latitudes and longitudes

uu = ds.u[:,0,-level:,::spacing,::spacing].mean(axis=0).data
vv = ds.v[:,0,-level:,::spacing,::spacing].mean(axis=0).data
ww = ds.w[:,0,-level:,::spacing,::spacing].mean(axis=0).data*(-1)
u,v,w = wind_transform2(lat_grid, lon_grid,uu,vv,ww)

# Earth surface plot
earth_surface = go.Surface(
    x=x_sphere,
    y=y_sphere,
    z=z_sphere,
    colorscale=[(0, 'blue'), (1, 'blue')],  # Single color 'green'
    # surfacecolor = resampled_img_data,
    opacity=1,
    showscale=False,  # Hide the colorbar
    # cmin=0,
    # cmax=255,
)
cmap = plt.get_cmap('terrain')
# Number of discrete colors you want from the colormap
num_colors = 256
# Create the colorscale by sampling the colormap
plotly_scale = [
    [i / (num_colors - 1), 'rgb({},{},{})'.format(*[int(x*255) for x in cmap(i / (num_colors - 1))[:3]])] 
    for i in range(num_colors)
]
# Create the scatter plot to simulate "texturing" with your data
scatter = go.Scatter3d(
    x=x_sphere.flatten(),
    y=y_sphere.flatten(),
    z=z_sphere.flatten(),
    mode='markers',
    marker=dict(
        symbol='diamond',
        size=8,  # Adjust the size of the markers to your liking
        color=image_data.flatten(),  # Apply the 2D data as colors
        colorscale=plotly_scale,  # Use an appropriate color scale for your data
        opacity=1.0,
        cmin=image_data.min(), 
        cmax=image_data.max(),
        showscale=False,
    )
)

# Convert wind vector locations to Cartesian coordinates
# In this simple example, we're placing the wind vectors just above the Earth's surface
wind_x, wind_y, wind_z = spherical_to_cartesian_wind(
    earth_radius ,  # Slightly above the Earth's surface to place the cones
    lat_grid,
    lon_grid,
  (ds.z[0,0,-level:,::spacing,::spacing].data/(9.8*1000)),
  HGT/1000.
)

# Cone plot for the wind vectors
wind_cones = go.Cone(
    x=wind_x.flatten(),
    y=wind_y.flatten(),
    z=wind_z.flatten() ,
    u=u.flatten(),
    v=v.flatten(),
    w=w.flatten(),
    # colorscale='Portland',
    # cmin=0,
    # cmax=15,
    colorbar=dict(title='Wind Speed'),
    sizemode='absolute',
    sizeref=1e7,  # Adjust the reference size for better visualization
)

# # Create the figure by combining the Earth surface and wind cones
fig = go.Figure(data=[earth_surface,scatter, wind_cones])
# # Add grid lines for longitude
# for lon in lons:
#     lon_rad = np.radians(lon)
#     x_line = earth_radius*np.cos(lon_rad) * np.cos(np.radians(lats))
#     y_line = earth_radius*np.sin(lon_rad) * np.cos(np.radians(lats))
#     z_line = earth_radius*np.sin(np.radians(lats))
    
#     fig.add_trace(go.Scatter3d(
#         x=x_line, y=y_line, z=z_line, mode='lines',
#         line=dict(color='black', width=2),
#         # name=f'Lon {lon}'
#     ))

# # Add grid lines for latitude
# for lat in lats:
#     lat_rad = np.radians(lat)
#     z_constant = earth_radius*np.sin(lat_rad)
#     x_circle = earth_radius*np.cos(np.radians(lons)) * np.cos(lat_rad)
#     y_circle = earth_radius*np.sin(np.radians(lons)) * np.cos(lat_rad)
    
#     fig.add_trace(go.Scatter3d(
#         x=x_circle, y=y_circle, z=z_constant * np.ones(lats.shape[0]),
#         mode='lines',
#         line=dict(color='black', width=2),
#         # name=f'Lat {lat}'
#     ))

fig.show()
fig.write_html('cone_plot2.html', auto_open=True)
