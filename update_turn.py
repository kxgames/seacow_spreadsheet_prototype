#!/usr/bin/env python3

import seacow
import pandas as pd
from time import sleep

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

def remake_purchases(purchases_df):
    # Get both columns and identify new purchases
    new_purchases = purchases_df.loc[:,'New Purchases']
    existing_purchases = purchases_df.loc[:,'Existing Purchases']

    new_purchases = new_purchases[new_purchases != '']
    existing_purchases = existing_purchases[existing_purchases != '']

    # Combine and sort
    combined = pd.concat([new_purchases, existing_purchases], ignore_index=True)
    sorted = combined.sort_values(ignore_index=True)

    # Recreate the two-column dataframe
    final_df = sorted.to_frame(name='Existing Purchases')
    final_df.insert(0, 'New Purchases', '')

    return final_df

seacow.record_status(doc, "Updating purchases...")
for player in [1, 2]:
    # Update purchases
    old_purchases_df = seacow.load_player_purchases(doc, player)
    new_purchases_df = remake_purchases(old_purchases_df)
    seacow.record_player_purchases(doc, player, new_purchases_df)

    # Update sheet for spying
    intel_receiver = (player % 2) + 1
    seacow.record_player_intel(doc, intel_receiver, new_purchases_df['Existing Purchases'])

seacow.record_status(doc, "Finished updating purchases")

for t in range(3,0,-1):
    # Pause for a bit to let sheets finish syncing changes
    seacow.record_status(doc, f"Taking a breather ({t})")
    sleep(1)

seacow.record_status(doc, "Updating ledgers")
for player in [1, 2]:
    # Update income
    ledger = seacow.load_player_income(doc, player)
    income = industries.apply(calc_income, axis=1, player=player).sum()
    row = pd.Series(
        [ledger.iloc[-1]['Turn'] + 1, income],
        index=ledger.columns,
    )
    ledger = ledger.append(row, ignore_index=True)
    seacow.record_player_income(doc, player, ledger)

    # Update global market info
    #seacow.record_global_market(doc, player, global_market)

seacow.record_status(doc, "Finished updating ledgers")

seacow.record_status(doc, "Play your turn")
