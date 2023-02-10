#!/usr/bin/env python3
# 

"""\
Usage:
    make_map.py [-r <seed>] [-w <width>] [-h <height>] [-n <n_cells>] [-D <method>] [-O <offset_radius_ratio>] [-a <avg_resource_density>] [-U]
    

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

    -D --cell-distribution-method <str>       [default: grid-hex]
        Distribution method for picking cell locations. Current options are 
        'uniform', 'grid-rectangular', and 'grid-hex'. The grid methods may need to use 
        more or less cells than directed with the --n-cells argument in order 
        to keep a regular grid structure.

    -O --offset-radius-ratio <float>        [default: 0.9]
        For grid distribution methods, this argument defines the upper limit of
        a random offset for each point in a regular grid by way of fractions of
        radii. Currently only used for 'grid-rectangular and 'grid-hex' options
        for '--cell-distribution-method'. The radius is calculated internally 
        for both the x and y dimensions to space points evenly on the map.

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

def make_rectangular_grid_points(target_n, map_width, map_height, max_random_offset=0.5):
    """
    Create a rectangular grid of points that are evenly spaced from 0 to 
    map_width or map_height plus a random offset. The dimensions of the point 
    grid will approximate the aspect ratio of the max dimensions and contain a 
    total number of points near the target number.
    Random offsets are kept within max_random_offset * radius.
    """

    # First calculate the exact dimensions that maintain the map aspect ratio 
    # then round to the nearest integer.
    # Given equations:
    #   target_n = nx * ny                  # Points in grid
    #   nx / ny = map_width / map_height    # Keep aspect ratio
    
    # Solve for nx
    #   ny = nx * map_height / map_width
    #   target_n = nx**2 * map_height / map_width
    nx = round(sqrt(target_n * map_width / map_height))

    # Solve for ny
    #   nx = ny * map_width / map_height
    #   target_n = ny**2 * map_width / map_height
    ny = round(sqrt(target_n * map_height / map_width))
    
    # Evenly space the points between the map extent such that cells would all 
    # be the same size
    radius_x = 1 / (2 * nx) * map_width # distance from cell center to cell edge
    radius_y = 1 / (2 * ny) * map_height # distance from cell center to cell edge
    coor_x = np.linspace(radius_x, map_width - radius_x, nx)
    coor_y = np.linspace(radius_y, map_height - radius_y, ny)
    mesh_coordinates = np.meshgrid(coor_x, coor_y)
    grid_points = np.dstack(mesh_coordinates).reshape(nx * ny, 2)

    # Add a random offset to each point
    # Uses a uniform distribution to determine direction and magnitude of 
    # offsets
    #max_random_offset = 9/10 # e.g. 1/2 means offset stays within a half radius
    polar_offsets = np_rng.uniform((0,0), (2*np.pi, max_random_offset), (nx * ny, 2))
    offsets = np.column_stack((
        np.cos(polar_offsets[:,0]) * polar_offsets[:,1] * radius_x,
        np.sin(polar_offsets[:,0]) * polar_offsets[:,1] * radius_y,
        ))

    if nx*ny != target_n:
        print(f"Creating {nx*ny} cells instead of {target_n} cells to create a {nx}x{ny} rectangular grid of cells")

    return grid_points + offsets

def _estimate_hex_grid_dimensions(target_n, map_width, map_height):

    assert target_n > 0
    assert map_width > 0
    assert map_height > 0

    # First calculate the exact dimensions that maintain the map aspect ratio 
    # then round to the nearest integer.
    # Note: nx and ny are the number of long columns and rows, respectively
    #       The number of short columns and rows would be (nx-1) and (ny-1), 
    #       respectively
    #
    # Note: hex grid organized such than tiles in the y direction touch
    #       eachother and horizontal direction have a gap.
    #
    #  Example of a 3x4 grid (nx=4, ny=3); n_cells would be 18
    #  *   *   *   *
    #    *   *   *
    #  *   *   *   *
    #    *   *   *
    #  *   *   *   *
    #  _   _   _   _
    # / \_/ \_/ \_/ \
    # \_/ \_/ \_/ \_/
    # / \_/ \_/ \_/ \
    # \_/ \_/ \_/ \_/
    # / \_/ \_/ \_/ \
    # \_/ \_/ \_/ \_/
    #
    # Grid dimensions
    #   r == radius; center to vertex on perimeter
    #   grid_w = nx*2*r + (nx-1)*r      # Tile width + gap
    #   grid_w = (3*nx - 1)*r
    #   grid_h = ny*2*r*cos(30)         # Tile height
    #   grid_h = sqrt(3)*ny*r

    # Governing equations:
    #   # Points in the hex grid (outer grid + inner grid):
    #       target_n = nx * ny + (nx - 1) * (ny - 1)
    #                = nx * ny + nx * ny - nx - ny + 1
    #                = 2 * nx * ny - nx - ny + 1
    #   # Aspect Ratio (exact):
    #       AR = map_width / map_height = grid_w / grid_h
    #       AR = ((3*nx - 1)*r) / (sqrt(3)*ny*r)
    #       AR = (3*nx - 1) / (sqrt(3)*ny)
    AR = map_width / map_height

    # Solve for nx
    #   # Rearrange aspect eq:
    #   ny = (3*nx -1) / (sqrt(3)*AR)
    #   # Rearrange points eq:
    #   0 = 2 * nx * ny - nx - ny + 1 - target_n
    #   # Substitute ny and simplify:
    #   0 = 2 * nx * ((3*nx - 1) / (sqrt(3)*AR))
    #       - nx - ((3*nx - 1) / (sqrt(3)*AR)) + 1 - target_n
    #   0 = 2 * nx * (3*nx - 1) - nx * (sqrt(3)*AR) - (3*nx - 1)
    #       + (1 - target_n) * (sqrt(3)*AR)
    #   0 = 6 * nx**2 - 2 * nx - nx * sqrt(3)*AR - 3*nx - 1
    #       + (1 - target_n) * sqrt(3)*AR
    #   0 = 6 * nx**2 + (- 2 - sqrt(3)*AR - 3) * nx
    #       - 1 + (1 - target_n) * sqrt(3)*AR
    #   0 = 6 * nx**2 + (-sqrt(3)*AR - 5) * nx
    #       - 1 + (1 - target_n) * sqrt(3)*AR
    #
    #   # Quadratic eq: x = (-b +- sqrt(b**2 - 4ac)) / (2a)
    #       a = 6;      b = -sqrt(3)*AR - 5;
    #       c = - 1 + (1 - target_n) * sqrt(3)*AR
    #   nx_discriminant = (-sqrt(3)*AR - 5)**2
    #                     - 4*(6)*(-1 + (1 - target_n) * sqrt(3)*AR)
    #   nx_discriminant = (sqrt(3)*AR)**2 + 10*sqrt(3)*AR + 25
    #                     + 24 - 24*(1 - target_n) * sqrt(3)*AR
    #   nx_discriminant = 3*AR**2 + 10*sqrt(3)*AR + (-24 + 24*target_n)
    #                     * sqrt(3)*AR + 25 + 24
    #   nx_discriminant = 3 * AR**2 + (24 * target_n - 14) * sqrt(3) * AR + 49
    #   nx = (-(-sqrt(3)*AR - 5) +- sqrt(nx_discriminant)) / (2*6)
    #   nx = (sqrt(3)*AR + 5 +- sqrt(nx_discriminant)) / 12
    nx_discriminant = 3 * AR**2 + (24 * target_n - 14) * sqrt(3) * AR + 49
    #nx_add = (sqrt(3)*AR + 5 + sqrt(nx_discriminant)) / 12
    #nx_sub = (sqrt(3)*AR + 5 - sqrt(nx_discriminant)) / 12

    # Solve for ny
    #   # Rearrange aspect eq:
    #   3*nx - 1 = AR * sqrt(3) * ny
    #   nx = (AR * sqrt(3) * ny + 1) / 3
    #   # Rearrange points eq:
    #   0 = 2 * nx * ny - nx - ny + 1 - target_n
    #   # Substitute nx and simplify:
    #   0 = 2 * ((AR * sqrt(3) * ny + 1) / 3) * ny
    #       - ((AR * sqrt(3) * ny + 1) / 3) - ny + 1 - target_n
    #   0 = 2 * AR * sqrt(3) * ny**2 + 2 * ny
    #       - AR * sqrt(3) * ny - 1 - 3 * ny + 3 - 3*target_n
    #   0 = 2 * AR * sqrt(3) * ny**2
    #       + (2 - AR * sqrt(3) - 3) * ny - 1 + 3 - 3*target_n
    #   0 = 2 * AR * sqrt(3) * ny**2
    #       + (-AR * sqrt(3) - 1) * ny + 2 - 3*target_n
    #
    #   # Quadratic eq: x = (-b +- sqrt(b**2 - 4ac)) / (2a)
    #       a = 2 * AR * sqrt(3);
    #       b = -AR * sqrt(3) - 1;
    #       c = 2 - 3*target_n;
    #   ny_discriminant = (-AR * sqrt(3) - 1)**2
    #                     - 4 * (2 * AR * sqrt(3)) * (2 - 3*target_n)
    #   ny_discriminant = (AR * sqrt(3) + 1)**2
    #                     - 8 * AR * sqrt(3) * (2 - 3*target_n)
    #   ny_discriminant = (AR * sqrt(3))**2 + 2 * AR * sqrt(3) + 1
    #                     - 16 * AR * sqrt(3) + 24 * AR * sqrt(3) * target_n
    #   ny_discriminant = 3 * AR**2 + 2 * sqrt(3) * AR - 16 * sqrt(3) * AR
    #                     + 24 * sqrt(3) * target_n * AR + 1
    #   ny_discriminant = 3 * AR**2 + (2 - 16 + 24 * target_n) * sqrt(3)*AR + 1
    #   ny_discriminant = 3 * AR**2 + (24 * target_n - 14) * sqrt(3)*AR + 1
    #   ny = (-(-AR * sqrt(3) - 1) +- sqrt(ny_discriminant))
    #        / (2 * (2 * AR * sqrt(3)))
    #   ny = (AR * sqrt(3) + 1 +- sqrt(ny_discriminant)) / (4 * AR * sqrt(3))
    ny_discriminant = 3 * AR**2 + (24 * target_n - 14) * sqrt(3) * AR + 1
    #ny_add = (AR * sqrt(3) + 1 + sqrt(ny_discriminant)) / (4 * AR * sqrt(3))
    #ny_sub = (AR * sqrt(3) + 1 - sqrt(ny_discriminant)) / (4 * AR * sqrt(3))

    print(f"  Given {target_n} cells, {map_width} width, and {map_height} height")
    if nx_discriminant < 0 or ny_discriminant < 0:
        # One or both of nx and ny does not have a solution
        # Print stuff then terminate
        print(f"  Unhandled situation: too few solutions!")
        print(f"  nx_discriminant = {nx_discriminant}")
        print(f"  ny_discriminant = {nx_discriminant}")
        assert False

    else:
        # Both nx and ny have one or more solutions
        too_many = False
        nx, nx_add, nx_sub = None, None, None
        if nx_discriminant == 0 :
            # One solution for nx, ignore nx_add and nx_sub
            #nx = (1 + 1/AR +- sqrt(0)) / (4/AR)
            nx = (AR + 1) / 4
            print(f"  One solution for nx {nx}")
        else:
            # Two solutions for nx
            nx_add = (sqrt(3)*AR + 5 + sqrt(nx_discriminant)) / 12
            nx_sub = (sqrt(3)*AR + 5 - sqrt(nx_discriminant)) / 12
            if (nx_sub > 0) != (nx_add > 0):
                # Only one is positive
                nx = max(nx_sub, nx_add)
                print(f"  One positive solution for nx: {nx_sub} or {nx_add}")
            else:
                too_many = True

        ny, ny_add, ny_sub = None, None, None
        if ny_discriminant == 0 :
            # One solution for ny, ignore ny_add and ny_sub
            #ny = (1 + AR + sqrt(0)) / (4*AR)
            ny = (1/AR + 1) / 4
            print(f"  One solution for ny {ny}")
        else:
            ny_add = (AR * sqrt(3) + 1 + sqrt(ny_discriminant)) / (4 * AR * sqrt(3))
            ny_sub = (AR * sqrt(3) + 1 - sqrt(ny_discriminant)) / (4 * AR * sqrt(3))
            if (ny_sub > 0) != (ny_add > 0):
                # Only one is positive
                ny = max(ny_sub, ny_add)
                print(f"  One positive solution for ny: {ny_sub} or {ny_add}")
            else:
                too_many = True

        if too_many:
            print(f"  Unhandled situation: too many positive solutions!")
            print(f"  nx = {nx_sub} or {nx_add} (or {nx})")
            print(f"  ny = {ny_sub} or {ny_add} (or {ny})")
            assert False

    assert nx > 0 and ny > 0
    print()
    print(f"Exact dimensions are {nx} x {ny}")
    nx, ny = round(nx), round(ny)
    n_grid = nx*ny + (nx-1)*(ny-1)
    AR_grid = (3*nx - 1) / (sqrt(3)*ny)
    print(f"Integer dimensions are {nx} x {ny} for n_cells = {n_grid}")
    print(f"Grid aspect ratio = {AR_grid} vs map aspect ratio {AR}")
    print()
    #assert False

    return nx, ny
    
def make_hex_grid_points(target_n, map_width, map_height, max_random_offset=0.5):
    """
    Create a hex grid of points that are evenly spaced from 0 to map_width or
    map_height plus a random offset. The dimensions of the point grid will 
    approximate the aspect ratio of the max dimensions and contain a total 
    number of points near the target number. The grid will always have longer 
    rows on the top and bottom, with alternating short and long rows in between. 
    Random offsets are kept within max_random_offset * radius.
    """

    nx, ny = _estimate_hex_grid_dimensions(target_n, map_width, map_height)

    # Note: nx and ny are the dimensions of the long rows and columns (outer 
    # dimensions of the grid). Shorter rows/columns (nxs or nys) will be placed 
    # in between longer rows/cols.
    
    # rx and ry == radius in x and y directions
    # map_width = nx * 2 * rx + (nx - 1) * rx   # Tile width + gap
    # map_width = (3 * nx - 1) * rx
    radius_x = map_width / (3 * nx - 1)
    # map_height = ny * 2 * ry * cos(30)        # Tile height
    # map_height = sqrt(3) * ny * ry
    radius_y = map_height / (sqrt(3) * ny)

    # Create outer grid
    # Evenly space the points between the map extent such that all cells are 
    # consistent with the same hexagon shape and size (border cells will be cut 
    # off some). Must account for layout of hex grids (points on long rows have 
    # gaps for short columns)
    outer_coor_x = np.linspace(0, map_width, nx)
    outer_coor_y = np.linspace(0, map_height, ny)
    outer_mesh_coordinates = np.meshgrid(outer_coor_x, outer_coor_y)
    outer_grid_points = np.dstack(outer_mesh_coordinates).reshape(nx * ny, 2)

    # Create inner grid
    inner_coor_x = np.linspace(1.5 * radius_x, map_width - 1.5 * radius_x, nx-1)
    inner_coor_y = np.linspace(radius_y, map_height - radius_y, ny-1)
    inner_mesh_coordinates = np.meshgrid(inner_coor_x, inner_coor_y)
    inner_grid_points = np.dstack(inner_mesh_coordinates).reshape((nx-1) * (ny-1), 2)

    # Complete grid
    grid_points = np.vstack((outer_grid_points, inner_grid_points))

    # Add a random offset to each point
    # Uses a uniform distribution to determine direction and magnitude of 
    # offsets
    #max_random_offset = 9/10 # e.g. 1/2 means offset stays within a half radius
    n_points = grid_points.shape[0]
    polar_offsets = np_rng.uniform((0,0), (2*np.pi, max_random_offset), (n_points,2))
    offsets = np.column_stack((
        np.cos(polar_offsets[:,0]) * polar_offsets[:,1] * radius_x,
        np.sin(polar_offsets[:,0]) * polar_offsets[:,1] * radius_y,
        ))
    grid_points = grid_points + offsets

    # Keep points within map extent
    grid_points[grid_points[:,0] < 0         , 0] = 0
    grid_points[grid_points[:,0] > map_width , 0] = map_width
    grid_points[grid_points[:,1] < 0         , 1] = 0
    grid_points[grid_points[:,1] > map_height, 1] = map_height

    if n_points != target_n:
        print(f"Creating {n_points} cells instead of {target_n} cells "
                + f"to create a {nx}x{ny} hex grid of cells")

    return grid_points

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
offset_radius_ratio = float(args['--offset_radius_ratio'])
if distribution_method == 'uniform':
    points = np_rng.uniform((0,0), (map_width, map_height), (n_cells, 2))

elif distribution_method == 'grid-rectangular':
    points = make_rectangular_grid_points(
            n_cells, map_width, map_height, offset_radius_ratio)

elif distribution_method == 'grid-hex':
    points = make_hex_grid_points(
            n_cells, map_width, map_height, offset_radius_ratio)

elif distribution_method == 'min-distance':
    # Similar to a uniform distribution but enforce a minimum distance between 
    # any two points.
    # Use a K-D Tree to efficiently organize the points for lookup?
    raise NotImplementedError

else:
    print(f"Distribution type '{distribution_method}' is not recognized")
    assert False
n_cells = points.shape[0]
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

    screen_max_length = 10
    AR = map_width / map_height
    if AR > 1: # wide map
        vor_fig.set_size_inches((screen_max_length, screen_max_length / AR))
    elif AR < 1: # tall map
        vor_fig.set_size_inches((screen_max_length * AR, screen_max_length))
    else: # square map
        vor_fig.set_size_inches((screen_max_length, screen_max_length))
    #plt.tight_layout()

    label_offset = (map_width/100, map_height/100) # 1% of map extent
    for label, p in zip(tile_labels, points):
        if label in tile_resources:
            msg = f"{label}-{''.join(tile_resources.loc[label])}"
        else:
            msg = f"{label}--"
        vor_ax.text(p[0] + label_offset[0], p[1] + label_offset[1], msg)

    plt.show()

