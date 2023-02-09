#!/usr/bin/env python3

"""
Generate an image of the map.

Usage:
    view_map.py
"""

import seacow
import networkx as nx

doc = seacow.load_doc()

tiles = seacow.load_map_tiles(doc)
edges = seacow.load_map_edges(doc)
resources = seacow.load_map_resources(doc)

# The coordinates will be interpreted as inches by GraphViz.  Here we scale the 
# coordinates to fit in a reasonable area:

xy = ['X', 'Y']
size = tiles[xy].max().max()
tiles[xy] *= 10 / size

# Make a graph with all the information from the spreadsheets:

map = nx.Graph()
for id, (x, y) in tiles.iterrows():
    map.add_node(id, pos=f'{x},{y}!', resources=[])

for _, (a, b) in edges.iterrows():
    map.add_edge(a, b)

for id, (resource,) in resources.iterrows():
    map.nodes[id]['resources'].append(resource)

for id in map.nodes:
    node = map.nodes[id]
    node['label'] = ','.join(sorted(node['resources']))

viz = nx.nx_agraph.to_agraph(map)
viz.draw('map.svg', prog='neato')


