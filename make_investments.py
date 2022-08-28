#!/usr/bin/env python3

import seacow
import pandas as pd
from itertools import count

doc = seacow.load_doc()
markets = seacow.load_markets(doc)

investments = []
effects = []
prices = [50, 100, 200, 400]
supplies = [10, 50, 200, 800]
demands = [0, 0, 0, 0]

for market in markets.index:
    for i, price, supply, demand in zip(count(1), prices, supplies, demands):
        name = f'{market}{i}'

        investments.append({
            'Investment': name,
            'Market': market,
            'Price': price,
        })

        effects.append({
            'Investment': name,
            'Market': market,
            'Marginal Supply at $1': supply,
            'Marginal Demand at $1': demand,
        })

seacow.record_investments(doc, pd.DataFrame(investments))
seacow.record_investment_effects(doc, pd.DataFrame(effects))

