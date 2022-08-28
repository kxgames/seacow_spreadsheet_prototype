#!/usr/bin/env python3

"""\
Usage:
    make_elasticities.py [-r <seed>] [-U]

Options:
    -r --random-seed <int>      [default: 0]
        Use the given random seed when generating the interaction graph.

    -U --no-upload
        Don't upload the resulting market interactions to Google drive.
"""

# Right now, this script creates a graph by picking 2 random interactions (1 
# positive, 1 negative) for each market.  These interactions will probably be 
# fairly ridiculous, but hopefully we'll at least see what kinds of graph 
# structures are strategically interesting.
#
# Writing this script, I can see that we might want to use use weights other 
# than +1/-1 to scale how closely two markets interact.  Note that this is 
# similar to the implicit supply/demand parameters we already have.  Those 
# parameters operate on individual markets: they make the market in 
# question more/less dependent on the supply/demand created by the players.  In 
# contrast, the interaction weights would operate on pairs of markets.  This 
# would give us more knobs to control, but would also make the game more 
# complicated.

import docopt
import seacow
import random
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from math import *
from itertools import cycle
from more_itertools import pairwise

args = docopt.docopt(__doc__)

seed = int(args['--random-seed'])
random.seed(seed)

doc = seacow.load_doc()
markets = seacow.load_markets(doc)

# Figure out which markets make up each round:

rounds = {}

for name, row in markets.iterrows():
    round = row["Round"]
    rounds.setdefault(round, []).append(name)

# Decide which markets will interact with each other:

interactions = nx.DiGraph()

round_colors = [
        'white',
        'gray80',
        'gray60',
]
for round_names, color in zip(rounds.values(), round_colors):
    interactions.add_nodes_from(round_names, fillcolor=color, style='filled')

for name, row in markets.iterrows():
    interactions.add_edge(name, name, supply=1, demand=1, style='invis')

    round = row["Round"]
    weights = [0.1, -0.1]
    round_mates = [x for x in rounds[round] if x != name]
    partners = [x for x in random.sample(round_mates, len(weights))]

    for partner, weight in zip(partners, weights):
        interactions.add_edge(
                partner, name,
                supply=weight,
                demand=0,
                color='green' if weight > 0 else 'red',
        )

# Add interactions between rounds:

for k, v in rounds.items():
    random.shuffle(v)

for old, new in pairwise(sorted(rounds)):
    for old_name, new_name in zip(rounds[old], cycle(rounds[new])):
        interactions.add_edge(
                new_name, old_name,
                supply=-1,
                color='lightpink',
        )

# Print the adjacency matrix to double check with if needed:

print(nx.to_pandas_adjacency(interactions).astype(int))

# Upload the market interactions:

if not args['--no-upload']:
    print("Uploading to Google drive...")
    df = pd.DataFrame([
            {
                'Market': k1,
                'Related Market': k2,
                'Supply Elasticity': v['supply'],
                'Demand Elasticity': v['demand'],
            }
            for (k1, k2), v in interactions.edges().items()
    ])
    seacow.record_elasticities(doc, df)
else:
    print("Skipped upload")

# Plot the market interactions:
#
# Note that GraphViz is the library that's actually making the images.  So 
# search for "graphviz" when looking for for info on how to change the 
# appearance of the graph.  For example, here are the colors you can use:
#
# https://graphviz.org/doc/info/colors.html

viz = nx.nx_agraph.to_agraph(interactions)
viz.draw('market_interactions.svg', prog='dot')
print("Updated SVG file")
