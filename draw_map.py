#!/usr/bin/env python3

"""
Generate an image of the map.

Usage:
    draw_map.py [-F] [-S]

Options:
    -F --full-map
        Draw the entire map regardless of player.

    -S --no-show
        Don't automatically open up an application for viewing the svg image.
"""

import docopt
import seacow
import networkx as nx
import subprocess

args = docopt.docopt(__doc__)

# Useful links:
# https://graphviz.org/doc/info/colors.html

## Settings
player_info = {
        0 : {
            'user' : None,
            'fillcolor' : 'grey90',
            },
        1 : {
            'user' : 'kale',
            'fillcolor' : 'yellow2',
            },
        2 : {
            'user' : 'alex',
            'fillcolor' : 'springgreen2',
            }
        }
user_info = {info['user'] : p_id for p_id, info in player_info.items()}

## Load data
print("Loading data...")
user_id = subprocess.check_output(['whoami']).decode().strip()
assert user_id in user_info
player_id = user_info[user_id]

doc = seacow.load_doc()
map = seacow.load_map(doc)

## Process map
"""
print("Processing map...")
for tile in map.nodes:
    node = map.nodes[tile]
    try:
        owner_id = seacow.who_controls(map, tile)
    except ValueError:
        owner_id = 0
    if not args['--full-map']:
        # Delete points depending on who runs the script.
        pass
"""

## Plot map
print("Plotting...")
#map.graph['overlap'] = 'scalexy'
for tile in map.nodes:
    node = map.nodes[tile]
    owner_id = seacow.who_controls(map, tile, return_none=True) or 0

    exploration = node['explore']
    label = ''
    if args['--full-map'] or player_id == owner_id:
        # Owned by player or script directed to show everything
        # Tile fully visible.
        label = f'{tile}: {node["resources"]}'
        node['fillcolor'] = player_info[owner_id]['fillcolor']
    elif player_id in exploration:
        # Explored tile
        # Show tile id and resources
        # Show foreign ownership only if explored after they took control
        label = f'{tile}: {node["resources"]}'
        exploration_turn = seacow.when_explored(map, tile, player_id)
        if owner_id > 0:
            ownership_turn = seacow.when_controlled(map, tile, owner_id)
            if ownership_turn <= exploration_turn:
                node['fillcolor'] = player_info[owner_id]['fillcolor']
    else:
        # Unexplored adjacent tile
        # Show tile id
        label = f'{tile}'
    #else:
        # Unknown tile
        # Hide/delete

    node['label'] = label
    node['pos'] = f'{node["x"]},{node["y"]}!'
    node['fontsize'] = '150'
    node['style'] = 'filled'

    #print(node)
    #print(vars(node))
    #assert False

viz = nx.nx_agraph.to_agraph(map)
viz.draw('map.svg', prog='neato')

if not args['--no-show']:
    print("Opening eog...")
    subprocess.run(['eog', 'map.svg'])
