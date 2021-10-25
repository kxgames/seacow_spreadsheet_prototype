#!/usr/bin/env python3

import seacow

doc = seacow.load_doc()
industries = seacow.load_industries(doc)

names = industries['Industry']
prices = [50, 100, 200, 400]

investments = []

for name in names:
    for i, price in enumerate(prices, 1):
        investments.append({
            'Investment': f'{name}{i}',
            'Industry': name,
            'Price': price,
        })

seacow.record_investments(doc, investments)

