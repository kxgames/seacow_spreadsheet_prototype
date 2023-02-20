#!/usr/bin/env python3

"""
Generate an image of the map.

Usage:
    draw_map.py [-F] [-A] [-S] [--show <viewer>]

Options:
    -F --full-map
        Draw the entire map.

    -A --all-players
        Draw separate maps for all players.

    -S
        Open up eog for viewing the svg image. Do not use with --show .

    --show <str>
        Open up an application for viewing the svg image. Do not use with -S .
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
        '0' : {
            'user' : 'game',
            'fillcolor' : 'grey90',
            },
        '1' : {
            'user' : 'kale',
            'fillcolor' : 'yellow2',
            },
        '2' : {
            'user' : 'alex',
            'fillcolor' : 'springgreen2',
            }
        }
user_info = {info['user'] : p_id for p_id, info in player_info.items()}

## Load data
print("Loading data...")
doc = seacow.load_doc()
map = seacow.load_map(doc)

## Plot map
def make_label(tile, node, is_explored=False):
    if is_explored:
        return '\n'.join([f'{tile}:', *[f'{r}:{v}' for r, v in node['resources'].items()]])
    else:
        return f'{tile}:\n?'

def plot_player(map, player_id):
    map = map.copy()
    username = player_info[player_id]['user']
    print(f"Plotting map for player {player_id} ({username})...")
    #map.graph['overlap'] = 'scalexy'
    tiles_to_remove = []
    for tile in map.nodes:
        node = map.nodes[tile]
        owner_id = seacow.who_controls(map, tile, return_none=True) or '0'

        node['pos'] = f'{node["x"]},{node["y"]}!'
        node['fontsize'] = '150'
        node['style'] = 'filled'

        if seacow.is_active_battle(map, tile):
            node['shape'] = 'doubleoctagon'
            node['penwidth'] = 10

        tile_exploration = node['explore']
        if player_id == '0' or player_id == owner_id:
            # Script directed to show all (player_id == 0) or owned by player
            # Tile fully visible.
            node['label'] = make_label(tile, node, is_explored=True)
            node['fillcolor'] = player_info[owner_id]['fillcolor']

        elif player_id in tile_exploration:
            # Explored tile
            # Show tile id and resources
            # Show foreign ownership only if explored after they took control
            node['label'] = make_label(tile, node, is_explored=True)
            exploration_turn = seacow.when_explored(map, tile, player_id)
            if owner_id != '0':
                ownership_turn = seacow.when_controlled(map, tile, owner_id)
                if ownership_turn <= exploration_turn:
                    node['fillcolor'] = player_info[owner_id]['fillcolor']
        else:
            # Unexplored tile
            # Determine if it is adjacent to explored area
            #tile_edges = map.edges(tile)
            #is_explored_by = lambda adj: seacow.is_explored_by(map, adj, player_id)
            #any_explored = any([is_explored_by(adj) for _, adj in tile_edges])
            any_explored = any([
                seacow.is_explored_by(map, adj, player_id)
                for _, adj in map.edges(tile)
                ])
            if any_explored:
                # Tile is adjacent to already explored area
                # Only show the tile id
                node['label'] = make_label(tile, node)
            else:
                # Unknown tile
                # Hide/delete
                 tiles_to_remove.append(tile)

    map.remove_nodes_from(tiles_to_remove)

    filename = f'map_p{player_id}_{username}.svg'
    viz = nx.nx_agraph.to_agraph(map)
    viz.draw(filename, prog='neato')

    print(f'map saved as {filename}')
    print()
    return filename

img_filenames = []
if args['--all-players']:
    # Make a plot for each player
    for player_id in player_info.keys():
        img_filenames.append(plot_player(map, player_id))

else:
    # Only plot current player
    user_id = subprocess.check_output(['whoami']).decode().strip()
    assert user_id in user_info
    player_id = user_info[user_id]
    img_filenames.append(plot_player(map, player_id))

if args['-S'] or args['--show']:
    viewer = args['--show'] or 'eog'
    for filename in img_filenames:
        print(f"Using {viewer} to open {filename}...")
        subprocess.Popen([viewer, filename])
    print("Note: image viewers may be slow to open")

