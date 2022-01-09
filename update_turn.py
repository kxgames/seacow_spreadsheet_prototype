#!/usr/bin/env python3

import seacow
import pandas as pd

# - Update player income

doc = seacow.load_doc()
industries = seacow.load_industries(doc)
global_market = industries.loc[:, 'Industry':'Demand']

def calc_income(row, player):
    my_supply = row[f'Player {player} Supply']
    global_supply = row['Supply']
    global_demand = row['Demand']
    price = row['Price']

    units_sold = max(0, min(
            my_supply,
            global_demand * my_supply / global_supply,
    ))
    return price * units_sold

for player in [1, 2]:
    ledger = seacow.load_player_income(doc, player)
    income = industries.apply(calc_income, axis=1, player=player).sum()
    row = pd.Series(
        [ledger.iloc[-1]['Turn'] + 1, income],
        index=ledger.columns,
    )
    ledger = ledger.append(row, ignore_index=True)
    seacow.record_player_income(doc, player, ledger)

    #seacow.record_global_market(doc, player, global_market)

    purchases = seacow.load_player_purchases(doc, player)
    intel_receiver = (player % 2) + 1
    seacow.record_player_intel(doc, intel_receiver, purchases)




