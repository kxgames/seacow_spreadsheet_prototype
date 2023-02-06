#!/usr/bin/env python3
# 

"""\
Usage:
    make_map.py [-r <seed>] [-w <width>] [-h <height>] [-n <n_cells>] [-D <method>] [-a <avg_density>] [-U]
    

Options:
    -r --random-seed <int or 'time'>      [default: time]
        Use the given random seed when generating the interaction graph.

    -w --map-width <int>      [default: 100]
        Width of the map.

    -h --map-height <int>      [default: 100]
        Height of the map.

    -n --n-cells <int>      [default: 36]
        Number of Voronoi cells to create using randomly positioned points. 
        More of a target number than a hard number as some cell distribution 
        methods might need change this number slightly.

    -D --cell-distribution-method <str>       [default: grid-rectangular]
        Distribution method for picking cell locations. Current options are 
        'uniform' and 'grid-rectangular'. The grid methods may need to use 
        more or less cells than directed with the --n-cells argument in order 
        to keep a regular grid structure.

    -a --avg-resource-density <float>       [default: 1.5]
        The map-averaged resource density target when assigning resources to 
        tiles. The actual map-average may be slightly different.
        e.g. Given an average of 1.5, there will be ceiling(1.5 * n_cells) 
        resources total for the whole map that are assigned to tiles randomly.

    -U --no-upload
        Don't upload the resulting market interactions to Google drive.
"""

import docopt
import seacow
import time
#import random
import numpy as np
import scipy.spatial
import pandas as pd
import matplotlib.pyplot as plt

from math import *

args = docopt.docopt(__doc__)

seed = args['--random-seed']
if seed == 'time':
    seed = time.time_ns()
else:
    seed = int(seed)
print(f"Using random seed {seed}")
#random.seed(seed)
np_rng = np.random.default_rng(seed)

map_width = int(args['--map-width'])
map_height = int(args['--map-height'])
n_cells = int(args['--n-cells'])
assert map_width > 0
assert map_height > 0
assert n_cells > 0

# Set up Seacow docs and load available resources
doc = seacow.load_doc()
sheet_markets = seacow.load_markets(doc)
resources = sheet_markets.index.values
#print("Temporarily hardcoding resources")
#resources = np.array(['A', 'B', 'C', 'D'])

## Generate a set of random points
# For now using a uniform distribution for simplicity.
# Will need to fix later because points can be too close
distribution_method = args['--cell-distribution-method']

if distribution_method == 'uniform':
    points = np_rng.uniform((0,0), (map_width, map_height), (n_cells, 2))

elif distribution_method == 'grid-rectangular':
    # Figure out the dimensions of the cell grid by calculating the exact 
    # dimensions that maintain the map aspect ratio, then rounding the 
    # dimensions so that the cell grid has integer dimensions.
    #
    # n_cells = nx * ny
    # nx / ny = map_width / map_height
    #
    # ny = nx * map_height / map_width
    # n_cells = nx**2 * map_height / map_width
    nx = round(sqrt(n_cells * map_width / map_height))
    #
    # nx = ny * map_width / map_height
    # n_cells = ny**2 * map_width / map_height
    ny = round(sqrt(n_cells * map_height / map_width))
    
    #print(f"Exact: nx = {nx}  ny = {ny}  nx * ny = {nx*ny}")
    #print(f"Approx: nx ~ {round(nx)}  ny ~ {round(ny)}  nx * ny = {round(nx)*round(ny)}")
    #print()
    if nx*ny != n_cells:
        print(f"Creating {nx*ny} cells instead of {n_cells} cells to create a {nx}x{ny} grid of cells")
        n_cells = nx*ny
    # Evenly space the points between the map extent such that cells would all 
    # be the same size
    radius_x = 1 / (2 * nx) * map_width # distance from cell center to cell edge
    radius_y = 1 / (2 * ny) * map_height # distance from cell center to cell edge
    coor_x = np.linspace(radius_x, map_width - radius_x, nx)
    coor_y = np.linspace(radius_y, map_height - radius_y, ny)
    mesh_coordinates = np.meshgrid(coor_x, coor_y)
    grid_points = np.dstack(mesh_coordinates).reshape(n_cells, 2)

    # Add randomness to each point
    # Uses uniform distribution to determine direction and magnitude of offsets
    max_unit_offset = 9/10 # e.g. offset stays within a half radius
    polar_offsets = np_rng.uniform((0,0), (2*np.pi, max_unit_offset), (n_cells, 2))
    offsets = np.column_stack((
        np.cos(polar_offsets[:,0]) * polar_offsets[:,1] * radius_x,
        np.sin(polar_offsets[:,0]) * polar_offsets[:,1] * radius_y,
        ))

    points = grid_points + offsets

elif distribution_method == 'grid-hex':
    # Generate a hex grid then add some randomness
    raise NotImplementedError

elif distribution_method == 'min-distance':
    # Similar to a uniform distribution but enforce a minimum distance between 
    # any two points.
    # Use a K-D Tree to efficiently organize the points for lookup?
    raise NotImplementedError

else:
    print(f"Distribution type '{distribution_method}' is not recognized")
    assert False
tile_labels = np.arange(n_cells)

## Generate the voronoi diagram
voronoi = scipy.spatial.Voronoi(points)

## Generate dataframes for the Voronoi map
sheet_tile_locations = pd.DataFrame(
        zip(tile_labels, points[:,0], points[:,1]),
        columns=['Tile', 'X', 'Y'],
        )

sheet_tile_neighbors = pd.DataFrame(
        voronoi.ridge_points,
        columns=['Tile 1', 'Tile 2'],
        )

## Distribute resources
map_resource_density = float(args['--avg-resource-density'])
total_resource_count = ceil(map_resource_density * n_cells)
resource_tiles = np_rng.choice(tile_labels, total_resource_count)
resource_names = np_rng.choice(resources, total_resource_count)
sheet_tile_resources = pd.DataFrame(
        zip(resource_tiles, resource_names),
        columns=['Tile', 'Resource'],
        )
sheet_tile_resources.sort_values(['Tile', 'Resource'], inplace=True, ignore_index=True)
tile_resources = sheet_tile_resources.groupby(by='Tile')['Resource'].apply(list)

## Upload information to Google Docs
if not args['--no-upload']:
    print("Uploading to Google Drive")
    seacow.record_map_tiles(doc, sheet_tile_locations, sheet_tile_neighbors)
    seacow.record_map_resources(doc, sheet_tile_resources)

else:
    # Print information about the cells
    print("Tile locations sheet:")
    print(sheet_tile_locations)
    print("Tile connections sheet:")
    print(sheet_tile_neighbors)
    print("Resources by tile:")
    #print(sheet_tile_resources)
    print(tile_resources)

    # Plot voronoi diagram
    vor_fig = scipy.spatial.voronoi_plot_2d(voronoi)
    vor_ax = vor_fig.get_axes()[0]
    vor_ax.set_aspect('equal')
    vor_ax.set_xlim(0,map_width)
    vor_ax.set_ylim(0,map_height)

    label_offset = (map_width/100, map_height/100) # 1% of map extent
    for label, p in zip(tile_labels, points):
        if label in tile_resources:
            msg = f"{label}-{''.join(tile_resources.loc[label])}"
        else:
            msg = f"{label}--"
        vor_ax.text(p[0] + label_offset[0], p[1] + label_offset[1], msg)

    plt.show()

