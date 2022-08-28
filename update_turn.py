#!/usr/bin/env python3

import seacow
import numpy as np
import pandas as pd

from numpy import log, exp

def update_turn(doc):
    markets = seacow.load_markets(doc)
    player_investments = seacow.load_player_existing_investments(doc)
    player_new_investments = seacow.load_player_new_investments(doc)
    investment_effects = seacow.load_investment_effects(doc)

    market_shares = calc_market_shares(
            markets,
            player_investments,
            investment_effects,
    )

    elasticities = seacow.load_elasticities(doc)

    update_market_state(markets, market_shares, elasticities)
    seacow.record_markets(doc, markets)

    player_incomes = seacow.load_player_incomes(doc)
    update_player_incomes(markets, market_shares, player_incomes)
    seacow.record_player_incomes(doc, player_incomes)

    df = concat_player_investments(player_new_investments, player_investments)
    seacow.record_player_investments(doc, df)
    seacow.clear_player_new_investments(doc)

def calc_market_shares(markets, player_investments, investment_effects):
    """
    Work out what quantity of supply and demand each player is individually 
    responsible for.

    These quantities are given in terms of "Supply at $1" and "Demand at $1".  
    In other words, these are the quantities that would be supplied/demanded if 
    the price were fixed.  The actual quantities supplied/demanded will be 
    adjusted once the prices are set.
    """
    market_shares = pd.DataFrame(
            data=0,
            index=pd.MultiIndex.from_product(
                (markets.index, seacow.PLAYERS + ['World']),
                names=['Market', 'Player'],
                ),
            columns=[
                'Supply at $1',
                'Demand at $1',
                'Supply Share',
                'Demand Share',
                ],
            )

    for market, row in markets.iterrows():
        i = market, 'World'
        market_shares.loc[i, 'Supply at $1'] += row['Supply at $1']
        market_shares.loc[i, 'Demand at $1'] += row['Demand at $1']

    for player, row in player_investments.iterrows():
        investment = row['Existing Investments']

        for market, row in investment_effects.loc[investment].iterrows():
            i = market, player
            market_shares.loc[i, 'Supply at $1'] += row['Marginal Supply at $1']
            market_shares.loc[i, 'Demand at $1'] += row['Marginal Demand at $1']

    # It seems like I should be able to do this calculation with `groupby()`, 
    # but I can't figure out how.

    for market, in markets.index:
        # If I keep *s* as a Series (instead of converting it to a numpy 
        # array), all the market shares end up getting set to NaN.  I don't 
        # understand why.
        s = market_shares.loc[market, 'Supply at $1'].values
        market_shares.loc[market, 'Supply Share'] = s / s.sum()

        d = market_shares.loc[market, 'Demand at $1'].values
        market_shares.loc[market, 'Demand Share'] = d / d.sum()

    debug()
    print(market_shares); print()
    return market_shares

def update_market_state(markets, market_shares, elasticities):
    """
    Calculate up-to-date prices for each market.

    The given market data frame is modified in place.
    """
    n = markets.shape[0]
    A = np.zeros((2*n, 2*n))
    B = np.zeros((2*n, 1))

    indices = {k: v for v, k in enumerate(markets.index)}

    debug()
    print(markets); print()
    print(market_shares); print()
    print(elasticities); print()

    for market in markets.index:
        i = 2 * indices[market]

        B[i+0,0] = market_shares.loc[market, 'Demand at $1'].sum()
        B[i+1,0] = market_shares.loc[market, 'Supply at $1'].sum()

    for (market, rel_market), row in elasticities.iterrows():
        i = 2 * indices[market]
        j = 2 * indices[rel_market]

        A[i+0,j+0] = (i == j)
        A[i+1,j+0] = (i == j)
        A[i+0,j+1] = row['Demand Elasticity']
        A[i+1,j+1] = -row['Supply Elasticity']

    x = exp(np.linalg.solve(A, log(B)))

    print(A); print()
    print(B); print()
    print(x); print()

    for market, index in indices.items():
        i = 2 * index

        markets.loc[market, 'Quantity'] = x[i+0,0]
        markets.loc[market, 'Price'] = x[i+1,0]

    markets['Total Value'] = markets['Quantity'] * markets['Price']
    print(markets); print()

def update_player_incomes(markets, market_shares, player_incomes):
    """
    Calculate how much income the players earned from their investments this 
    turn.

    The total value of each market is the quantity times the price.  Both of 
    these values are given by the *markets* data frame.  The players earn this 
    amount time their share of the supply for the relevant market.  These 
    shares are given by the *market_shares* data frame.

    The *player_incomes* data frame is updated in place.  Specifically, a new 
    row representing the current turn is added for each player.
    """
    market_shares = market_shares.drop('World', level='Player')

    df = pd.merge(
            market_shares, markets['Total Value'],
            left_index=True,
            right_index=True,
    )
    df['Income'] = df['Total Value'] * df['Supply Share']

    incomes = df['Income'].groupby('Player').agg('sum')
    turn = player_incomes.index.unique('Turn').max() + 1

    # There aren't a lot of good ways to update a data frame in place.  (Which 
    # is reasonable; growing a data frame always requires copying the entire 
    # thing, and should generally be avoided).  However, in this case it's what 
    # we want, so we have to resort to a for-loop to make it happen.

    for player, income in incomes.iteritems():
        player_incomes.loc[(player, turn), 'Income'] = income

    player_incomes.sort_index(inplace=True)

def concat_player_investments(player_new_investments, player_existing_investments):
    s = pd.concat([
        player_new_investments['New Investments'],
        player_existing_investments['Existing Investments'],
    ])
    s.name = 'Existing Investments'
    return s.to_frame()


if __name__ == '__main__':
    doc = seacow.load_doc()
    update_turn(doc)

