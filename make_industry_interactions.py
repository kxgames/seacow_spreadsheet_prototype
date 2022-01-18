#!/usr/bin/env python3

"""\
Usage:
    make_industry_interactions.py [-r <seed>] [-U] [-a]

Options:
    -r --random-seed <int>      [default: 0]
        Use the given random seed when generating the interaction graph.

    -U --no-upload
        Don't upload the resulting industry interactions to Google drive.

    -a --print-adjacency
        Print the final adjacency matrix.
"""

# Advanced Industries update goals:
# - Introduce Game State inputs. In the real game, these would be variables 
#   like total population or war/peace states. For the prototype, these will be 
#   simple functions that change in predictable ways over the turns (e.g.  
#   linearly increasing, oscillating, etc.). The Game States can affect 
#   multiple industries, but are not directly impacted by any industries and 
#   cannot be invested in by players. In older version of the prototype, these 
#   Game States would be akin to the industries with only outgoing interaction 
#   arrows (which we got rid of because they were boring to invest in).
#
# - Define a hierarchy of industry categories. Initial thoughts are Raw, 
#   Processing, and Luxury. These are closely related to industry stability.  
#   Raw draw from only the Game States for now, but will eventually be 
#   spatially dependent too once maps are implemented. Generally Raw Industries 
#   do not compete with each other (except maybe new ones appearing later in 
#   the gameplay). Processing Industries have the most complex set of positive 
#   and negative interactions with each other, the Raw , and the Luxury 
#   Industries. The Luxury Industries are the downstream "output" of some 
#   Processing Industries. These are volatile but potentially very lucrative.
#
# - Design industries to "chain" supply and demand. For example, given a 
#   positive interaction sequence of A -> B -> C, both the supply and demand in 
#   A must increase to increase demand in B. However, if B had more supply than 
#   demand already, increasing demand in B will also drive up demand in C 
#   without the player needing to make a separate investment. Similarly 
#   suppressing the demand in A below the supply level can start to drag down B 
#   and C too. This requires rethinking the governing equations for the 
#   industries. Perhaps more like a neural network.

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
import numpy as np
import matplotlib.pyplot as plt

from math import *
from itertools import cycle
from more_itertools import pairwise
from collections import defaultdict

args = docopt.docopt(__doc__)

seed = int(args['--random-seed'])
random.seed(seed)
np.random.seed(seed)

doc = seacow.load_doc()
industries = seacow.load_industries(doc)
names = industries["Industry"]

# Add nodes to the graph and figure out which industries make up each round:
interactions = nx.DiGraph()

round_colors = defaultdict(lambda: 'white', {
    1 : 'white',
    2 : 'gray80',
    3 : 'gray60',
    })
category_shapes = defaultdict(lambda: 'oval', {
    'GS' : 'Mcircle',
    'R'  : 'oval',
    'P'  : 'house',
    'L'  : 'diamond',
    })

rounds_dict = defaultdict(list)
categories_dict = defaultdict(list)
for i, row in industries.iterrows():
    name = row["Industry"]
    category = row["Category"]
    round = row["Round"]
    color = round_colors[round]
    shape = category_shapes[category]

    rounds_dict[round].append(name)
    categories_dict[category].append(name)

    interactions.add_node(name, fillcolor=color, style='filled', shape=shape)

# Decide which industries will interact with each other:
def get_partners(
        query_expr, 
        n_partner_range=(0,1), 
        weight_choices=[-1,1], 
        min_n_pos_weights=0,
        min_n_neg_weights=0,
        ):
    # Choose a random number (within provided range) of partners that meet the 
    # search criteria.
    weight_choices = np.array(weight_choices).astype(float)

    partners = np.array([])
    weights = np.array([])

    industry_options = industries.query(query_expr)['Industry'].values

    # Figure out how many partners there should be
    npr = n_partner_range
    n_partners = min(
            industry_options.size, 
            npr[0] if np.equal(*npr) else random.randint(*npr)
            )

    if n_partners > 0:
        # Choose partners
        partners = np.random.choice(industry_options, n_partners, replace=False)

        # Assign weights, respecting sign requirements
        pos_weights, neg_weights, rand_weights = [np.array([]) for i in [1,2,3]]
        assert partners.size >= min_n_pos_weights + min_n_neg_weights

        if min_n_pos_weights > 0:
            assert any(weight_choices > 0)
            pos_choices = weight_choices[weight_choices > 0]
            pos_weights = np.random.choice(pos_choices, min_n_pos_weights)

        if min_n_neg_weights > 0:
            assert any(weight_choices < 0)
            neg_choices = weight_choices[weight_choices < 0]
            neg_weights = np.random.choice(neg_choices, min_n_neg_weights)

        n_random_sign = partners.size - min_n_pos_weights - min_n_neg_weights
        if n_random_sign > 0:
            rand_weights = np.random.choice(weight_choices, n_random_sign)

        weights = np.random.permutation(
                np.hstack([pos_weights, neg_weights, rand_weights])
                )

    return partners, weights

def influenced_by_tuples(name, partners, weights):
    # Note, these are current industry influenced by partners ( c <- p )
    return [(p, name, w) for p, w in zip(partners, weights)]

def influences_tuples(name, partners, weights):
    # Note, these are current industry influences partners ( c -> p )
    return [(name, p, w) for p, w in zip(partners, weights)]

new_links = [] # [(independent_industry , dependent_industry, weight), ...]
for _, row in industries.iterrows():
    name = row["Industry"]
    category = row["Category"]
    round = row["Round"]

    if category == 'GS':
        # Game states do not depend on other industries
        continue

    elif category == 'R':
        # All raw industries are influenced by 1-2 game states chosen randomly
        GS_partners, GS_weights = get_partners(
                f"Industry != '{name}' and Category == 'GS'",
                n_partner_range=(1,2), min_n_pos_weights=1,
                )
        new_links += influenced_by_tuples(name, GS_partners, GS_weights)

        # Raw industries from later rounds can influence earlier industries
        # For now only pick up to one industry and give it a negative weight. 
        # This may change in the future.
        R_partners, R_weights = get_partners(
                f"Industry != '{name}' and Category == 'R' and Round < {round}",
                n_partner_range=(0,1), weight_choices=[-1],
                )
        new_links += influences_tuples(name, R_partners, R_weights)

        # Each Raw industry influences one to three Processing industries from 
        # any round.  At least one interaction will be positive.
        P_partners, P_weights = get_partners(
                f"Industry != '{name}' and Category == 'P'",
                n_partner_range=(1,3),
                min_n_pos_weights=1,
                )
        new_links += influences_tuples(name, P_partners, P_weights)

    elif category == 'P':
        # Processing industries are influenced by 2-3 other processing 
        # industries from any round. At least one interaction will be positive 
        # and one will be negative.
        P_partners, P_weights = get_partners(
                f"Industry != '{name}' and Category == 'P'",
                n_partner_range=(2,3),
                min_n_pos_weights=1, min_n_neg_weights=1,
                )
        new_links += influenced_by_tuples(name, P_partners, P_weights)

    elif category == 'L':
        # Luxury industries are influenced by 2-3 Processing industries from 
        # the same or earlier rounds. At least one interaction will be positive 
        # and one will be negative.

        P_partners, P_weights = get_partners(
                f"Industry != '{name}' and Category == 'P' and Round <= {round}",
                n_partner_range=(2,3),
                min_n_pos_weights=1, min_n_neg_weights=1,
                )
        new_links += influenced_by_tuples(name, P_partners, P_weights)

        # Luxury industries are negatively influenced by all other luxury 
        # industries from any round.
        L_partners, L_weights = get_partners(
                f"Industry != '{name}' and Category == 'L'",
                n_partner_range=(np.inf,np.inf), weight_choices=[-1],
                )
        new_links += influenced_by_tuples(name, L_partners, L_weights)

    else:
        print(f"Unkown industry category {category}")
        continue

for link_A, link_B, weight in new_links:
    interactions.add_edge(
            link_A, link_B,
            weight=weight,
            color='green' if weight > 0 else 'red',
    )


# Print the adjacency matrix to double check with if needed:

if args['--print-adjacency']:
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
