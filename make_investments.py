#!/usr/bin/env python3

import seacow
from itertools import count

doc = seacow.load_doc()
industries = seacow.load_industries(doc)
interactions = seacow.load_industry_interactions(doc)

investments = []
effects = []
prices = [50, 100, 200, 400]
supplies = [10, 50, 200, 800]
demands = [5, 25, 100, 400]

for industry in industries['Industry']:
    for i, price, supply, demand in zip(count(1), prices, supplies, demands):
        name = f'{industry}{i}'

        investments.append({
            'Investment': name,
            'Industry': industry,
            'Price': price,
        })

        effects.append({
            'Investment': name,
            'Industry': industry,
            'Effect': 'Supply',
            'Value': supply,
        })

        relevant = (interactions['Industry 1'] == industry)
        for _, interaction in interactions[relevant].iterrows():
            effects.append({
                'Investment': name,
                'Industry': interaction['Industry 2'],
                'Effect': 'Demand',
                'Value': demand * interaction['Interaction'],
            })

seacow.record_investments(doc, investments)
seacow.record_investment_effects(doc, effects)

