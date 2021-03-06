#!/usr/bin/env python3

"""\
Usage:
    make_industry_interactions.py [-r <seed>] [-U]

Options:
    -r --random-seed <int>      [default: 0]
        Use the given random seed when generating the interaction graph.

    -U --no-upload
        Don't upload the resulting industry interactions to Google drive.
"""

# Right now, this script creates a graph by picking 2 random interactions (1 
# positive, 1 negative) for each industry.  These interactions will probably be 
# fairly ridiculous, but hopefully we'll at least see what kinds of graph 
# structures are strategically interesting.
#
# Writing this script, I can see that we might want to use use weights other 
# than +1/-1 to scale how closely two industries interact.  Note that this is 
# similar to the implicit supply/demand parameters we already have.  Those 
# parameters operate on individual industries: they make the industry in 
# question more/less dependent on the supply/demand created by the players.  In 
# contrast, the interaction weights would operate on pairs of industries.  This 
# would give us more knobs to control, but would also make the game more 
# complicated.

import docopt
import seacow
import random
import networkx as nx
import matplotlib.pyplot as plt

from math import *
from itertools import cycle
from more_itertools import pairwise

args = docopt.docopt(__doc__)

seed = int(args['--random-seed'])
random.seed(seed)

doc = seacow.load_doc()
industries = seacow.load_industries(doc)
names = industries["Industry"]

# Figure out which industries make up each round:

rounds = {}

for i, row in industries.iterrows():
    name = row["Industry"]
    round = row["Round"]
    rounds.setdefault(round, []).append(name)

# Decide which industries will interact with each other:

interactions = nx.DiGraph()

round_colors = [
        'white',
        'gray80',
        'gray60',
]
for round_names, color in zip(rounds.values(), round_colors):
    interactions.add_nodes_from(round_names, fillcolor=color, style='filled')

for i, row in industries.iterrows():
    name = row["Industry"]
    round = row["Round"]

    weights = [1, -1]
    round_mates = [x for x in rounds[round] if x != name]
    partners = [x for x in random.sample(round_mates, len(weights))]

    for partner, weight in zip(partners, weights):
        interactions.add_edge(
                partner, name,
                weight=weight,
                color='green' if weight > 0 else 'red',
        )

# Add interactions between rounds:

for k, v in rounds.items():
    random.shuffle(v)

for old, new in pairwise(sorted(rounds)):
    for old_name, new_name in zip(rounds[old], cycle(rounds[new])):
        interactions.add_edge(
                new_name, old_name,
                weight=-1,
                color='lightpink',
        )

# Print the adjacency matrix to double check with if needed:

print(nx.to_pandas_adjacency(interactions).astype(int))

# Upload the industry interactions:

if not args['--no-upload']:
    print("Uploading to Google drive...")
    records = [
            {'Industry 1': k1, 'Industry 2': k2, 'Interaction': v['weight']}
            for (k1, k2), v in interactions.edges().items()
    ]
    seacow.record_industry_interactions(doc, records)
else:
    print("Skipped upload")

# Plot the industry interactions:
#
# Note that GraphViz is the library that's actually making the images.  So 
# search for "graphviz" when looking for for info on how to change the 
# appearance of the graph.  For example, here are the colors you can use:
#
# https://graphviz.org/doc/info/colors.html

viz = nx.nx_agraph.to_agraph(interactions)
viz.draw('industry_interactions.svg', prog='dot')
print("Updated SVG file")
