#!/usr/bin/env python3

"""
Generate an image of the map.

Usage:
    draw_map.py
"""

import seacow
import networkx as nx

doc = seacow.load_doc()
map = seacow.load_map(doc)

for tile in map.nodes:
    node = map.nodes[tile]
    node['pos'] = f'{node["x"]},{node["y"]}!'
    node['label'] = f'{tile}: {node["resources"]}'

viz = nx.nx_agraph.to_agraph(map)
viz.draw('map.svg', prog='neato')


