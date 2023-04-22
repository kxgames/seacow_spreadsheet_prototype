#!/usr/bin/env python3

"""
Update the state of the game world after each player has taken their turn.

Usage:
    update_turn.py [-i]

Options:
    -i --interactive
        Run in a loop, performing an update whenever the user presses [Enter].  
        This minimizes the number of API requests that have to be made to 
        Google Sheets, because a lot of the sheets are read-only and can just 
        be downloaded once at the beginning (as opposed to every turn).
"""

import seacow
import numpy as np
import pandas as pd
import random

from numpy import log, exp
from itertools import chain, repeat

class World:

    def __init__(self, doc):
        self.turn = seacow.load_turn(doc)

        self.markets = seacow.load_markets(doc)
        self.market_shares = None  # filled in by `update_market_*()`
        self.elasticities = seacow.load_elasticities(doc)

        self.map_tiles = seacow.load_map_tiles(doc)
        self.map_edges = seacow.load_map_edges(doc)
        self.map_resources = seacow.load_map_resources(doc)
        self.map_exploration = seacow.load_map_exploration(doc)
        self.map_control = seacow.load_map_control(doc)

        self.player_logs = seacow.load_player_logs(doc)
        self.player_balances = seacow.load_player_balances(doc)
        self.player_actions = seacow.load_player_actions(doc)
        self.player_soldiers = seacow.load_player_soldiers(doc)
        self.pending_actions = None  # filled in by `update_actions()`

        self.technologies = seacow.load_technologies(doc)
        self.technology_prices = seacow.load_technology_prices(doc)

        self.buildings = seacow.load_buildings(doc)
        self.building_prices = seacow.load_building_prices(doc)
        self.building_resources = seacow.load_building_resources(doc)
        self.building_effects = seacow.load_building_effects(doc)

        self.battles = seacow.load_battles(doc)

        self.refresh_map()

    def log(self, message, player=None):
        players = [player] if player else seacow.PLAYERS
        turn = self.turn

        df = pd.DataFrame([
                {'Turn': turn, 'Player': player, 'Log': message}
                for player in players
        ])

        self.player_logs = pd.concat([self.player_logs, df])

    def refresh_turn(self):
        self.turn += 1

    def refresh_map(self):
        self.map = seacow.compose_map(
                tiles=self.map_tiles,
                edges=self.map_edges,
                resources=self.map_resources,
                exploration=self.map_exploration,
                control=self.map_control,
                battles=self.battles,
        )

class ActionError(Exception):
    pass

class TooExpensive(ActionError):

    def __init__(self, balance, required):
        self.balance = balance
        self.required = required

class SkipWithWarning(ActionError):

    def __init__(self, message):
        self.message = message

def update_turn(world):
    world.refresh_turn()

    update_actions(world)
    update_market_shares(world)
    update_market_prices(world)
    update_incomes(world)
    update_battles(world)

def update_actions(world):
    # Copy the actions into a more malleable data structure:
    actions = {}
    pending_actions = []

    for player, group in world.player_actions.groupby('Player'):
        actions[player] = list(group['Actions'])

    while actions:
        # I don't think it matters what order the players take their actions 
        # in, but in case it does (in some way that I'm overlooking), it'll 
        # always be fair to randomize the order.
        player = random.choice(list(actions.keys()))
        action = actions[player][0]

        try:
            take_action(world, player, action)

        except TooExpensive as warn:
            message = chain(
                    [f"needed ${warn.required:.0f}, only had ${warn.balance:.0f}"],
                    repeat(''),
            )
            pending_actions += [
                    {'Player': player, 'Actions': action, 'Warning': next(message)}
                    for action in actions[player]
            ]
            del actions[player][:]

        except SkipWithWarning as warn:
            pending_actions.append({
                'Player': player,
                'Actions': action,
                'Warning': warn.message,
            })
            del actions[player][0]

        else:
            del actions[player][0]

        if not actions[player]:
            del actions[player]

    world.pending_actions = pd.DataFrame(
            pending_actions,
            columns=['Player', 'Actions', 'Warning'],
    )

def update_market_shares(world):
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
                (world.markets.index, seacow.PLAYERS + ['World']),
                names=['Market', 'Player'],
                ),
            columns=[
                'Supply at $1',
                'Demand at $1',
                'Supply Share',
                'Demand Share',
                ],
            )

    for market, row in world.markets.iterrows():
        i = market, 'World'
        market_shares.loc[i, 'Supply at $1'] += row['Supply at $1']
        market_shares.loc[i, 'Demand at $1'] += row['Demand at $1']

    for (tile, building), quantity in world.buildings.iterrows():
        player = seacow.who_controls(world.map, tile)

        for market, row in world.building_effects.loc[building].iterrows():
            i = market, player
            market_shares.loc[i, 'Supply at $1'] += row['Marginal Supply at $1']
            market_shares.loc[i, 'Demand at $1'] += row['Marginal Demand at $1']

    # It seems like I should be able to do this calculation with `groupby()`, 
    # but I can't figure out how.

    for market in world.markets.index:
        # If I keep *s* as a Series (instead of converting it to a numpy array 
        # with `.values`), all the market shares end up getting set to NaN.  I 
        # don't understand why.
        s = market_shares.loc[market, 'Supply at $1'].values
        market_shares.loc[market, 'Supply Share'] = s / s.sum()

        d = market_shares.loc[market, 'Demand at $1'].values
        market_shares.loc[market, 'Demand Share'] = d / d.sum()

    world.market_shares = market_shares

def update_market_prices(world):
    """
    Calculate up-to-date prices for each market.

    The given *markets* data frame is modified in place.  A modified version of 
    the *market_shares* data frame (with "Total Value" and "Income" columns) is 
    returned.
    """
    n = world.markets.shape[0]
    A = np.zeros((2*n, 2*n))
    B = np.zeros((2*n, 1))

    indices = {k: v for v, k in enumerate(world.markets.index)}

    for market in world.markets.index:
        i = 2 * indices[market]

        B[i+0,0] = world.market_shares.loc[market, 'Demand at $1'].sum()
        B[i+1,0] = world.market_shares.loc[market, 'Supply at $1'].sum()

    for (market, rel_market), row in world.elasticities.iterrows():
        i = 2 * indices[market]
        j = 2 * indices[rel_market]

        A[i+0,j+0] = (i == j)
        A[i+1,j+0] = (i == j)
        A[i+0,j+1] = row['Demand Elasticity']
        A[i+1,j+1] = -row['Supply Elasticity']

    x = exp(np.linalg.solve(A, log(B)))

    for market, index in indices.items():
        i = 2 * index

        world.markets.loc[market, 'Quantity'] = x[i+0,0]
        world.markets.loc[market, 'Price'] = x[i+1,0]

    world.markets['Total Value'] = \
            world.markets['Quantity'] * world.markets['Price']

    world.market_shares = pd.merge(
            world.market_shares,
            world.markets['Total Value'],
            left_index=True,
            right_index=True,
    )
    world.market_shares['Income'] = (
            world.market_shares['Total Value'] *
            world.market_shares['Supply Share']
    ).round()

def update_incomes(world):
    """
    Calculate how much income the players earned from their buildings this 
    turn, and update their balances accordingly.
    """
    market_shares = world.market_shares.drop('World', level='Player')
    incomes = market_shares['Income'].groupby('Player').sum()

    for player, income in incomes.items():
        world.log(f"Income: ${income:.0f}", player=player)
        world.player_balances[player] += income

def update_battles(world):
    # - Each turn, each player gets `1 + log2(number of soldiers)` attacks.  
    #   Each attack has a 50% chance of killing an enemy soldier.  The number 
    #   of attacks is calculated before any soldiers are killed.
    #
    #   - The `log2` makes it so that bigger armies get more attacks, but there 
    #     are diminishing returns, so a smaller army can at least hold on for a 
    #     while.
    #
    #   - The 50% kill-probability keeps things a bit unpredictable.
    #
    # - The battle lasts until one side is out of soldiers.  The remaining side 
    #   gains control of the tile in question.
    #
    # - Reinforcements are more expensive for the attacker than the defender.
    #
    #   - This encourages the attacker to commit more troops then they might 
    #     need, possibly leaving the door open for a counter-attack.
    #
    # - Retreats are expensive, and incur casualties.

    def fight(num_attackers, num_defenders):
        num_attacks = 1 + np.ceil(np.log2(num_attackers))
        num_kills = np.random.binomial(num_attacks, seacow.KILL_PROBABILITY)
        return max(num_defenders - num_kills, 0)

    # Don't simulate battles on the turns they were created, because the 
    # defender won't have had a chance to commit soldiers yet.
    active_battles = world.battles.query('Begin < @world.turn and End == 0')

    for i, battle in active_battles.iterrows():
        tile = battle['Tile']
        num_attackers = num_surviving_attackers = battle['Attacking Soldiers']
        num_defenders = num_surviving_defenders = battle['Defending Soldiers']

        # It's possible that one side will have no soldiers, e.g. if that side 
        # retreated this turn.  In this case, don't bother simulating the fight 
        # because (i) it will have no effect anyways and (ii) the logarithm in 
        # `fight()` can't handle 0.
        if (num_attackers > 0) and (num_defenders > 0):
            num_surviving_attackers = fight(num_defenders, num_attackers)
            num_surviving_defenders = fight(num_attackers, num_defenders)

            world.log(f"Battle on tile {tile}:")
            world.log(f"{num_attackers - num_surviving_attackers} attackers killed, {num_surviving_attackers} remaining")
            world.log(f"{num_defenders - num_surviving_defenders} defenders killed, {num_surviving_defenders} remaining")

            world.battles.loc[i, 'Attacking Soldiers'] = num_surviving_attackers
            world.battles.loc[i, 'Defending Soldiers'] = num_surviving_defenders

        if (num_surviving_attackers == 0) or (num_surviving_defenders == 0):
            world.battles.loc[i, 'End'] = world.turn

            # Return any surviving soldiers to their players:
            attacker = battle['Attacker']
            defender = battle['Defender']

            world.player_soldiers[attacker] += num_surviving_attackers
            world.player_soldiers[defender] += num_surviving_defenders

            # Update the map:
            if num_surviving_attackers:
                world.log(f"Player {attacker} took control of tile {tile}!")

                world.map_control.loc[(tile, world.turn), 'Player'] = attacker
                world.refresh_map()
                
            else:
                # If both players run out of soldiers, the defender wins by 
                # default.
                world.log(f"Player {defender} defended the attack on tile {tile}!")


def take_action(world, player, action):
    """
    Parse and then execute the given action.

    Each action will either:
    - Update the state of the world somehow
    - Raise a `TooExpensive` exception, indicating that the player is out of 
      money and should take no more actions this turn.
    - Raise a `SkipWithWarning` exception, indicating that something about the 
      action doesn't make sense.  The action will be skipped, and the player 
      will be presented with a warning message, so they know what happened.
    """
    match action.split():
        case ("explore", *args):
            tile = parse_arguments(world, [parse_tile], args)
            explore(world, player, tile)

        case ("expand", *args):
            tile = parse_arguments(world, [parse_tile], args)
            expand(world, player, tile)

        case ("research", *args):
            tech = parse_arguments(world, [parse_technology], args)
            research(world, player, tech)

        case ("build", *args):
            tile, building = parse_arguments(world, [parse_tile, parse_building], args)
            build(world, player, tile, building)

        case ("recruit", *args):
            soldiers = parse_arguments(world, [parse_soldiers], args)
            recruit(world, player, soldiers)

        case ("attack", *args):
            tile, soldiers = parse_arguments(world, [parse_tile, parse_soldiers], args)
            attack(world, player, tile, soldiers)

        case ("defend", *args):
            tile, soldiers = parse_arguments(world, [parse_tile, parse_soldiers], args)
            defend(world, player, tile, soldiers)

        case ("retreat", *args):
            tile = parse_arguments(world, [parse_tile], args)
            retreat(world, player, tile)

        case (cmd, *args):
            raise SkipWithWarning(f"unknown action {cmd!r}")

def explore(world, player, tile):
    require_explored_neighbor(world.map, tile, player)
    require_payment(world, player, seacow.EXPLORE_PRICE)

    # Update the "Map Exploration" table in place:
    i = len(world.map_exploration)
    world.map_exploration.loc[i] = {
            'Tile': tile,
            'Player': player,
            'Turn': world.turn,
    }

    # Other actions may depend on the map being up-to-date:
    world.refresh_map()

def expand(world, player, tile):
    require_explored_tile(world.map, tile, player)
    require_controlled_neighbor(world.map, tile, player)
    require_uncontrolled_tile(world.map, tile)
    require_payment(world, player, seacow.EXPAND_PRICE)

    # Update the "Map Control" table in place:
    world.map_control.loc[(tile, world.turn), 'Player'] = player

    # Other actions may depend on the map being up-to-date:
    world.refresh_map()

def research(world, player, tech):
    price = world.technology_prices.at[tech, 'Price']

    require_unresearched(world, player, tech)
    require_payment(world, player, price)

    # Update the "Technologies" table in place:

    df = world.technologies
    players_list = df.loc[tech,'Players'].split(',') if tech in df.index else []
    players_list.append(f"{player}")
    df.loc[tech,'Players'] = ','.join(players_list)

def build(world, player, tile, building):
    price = world.building_prices.at[building, 'Price']
    required_tech = world.building_prices.at[building, 'Prereq Tech']

    require_research(world, player, required_tech)
    require_controlled_tile(world.map, tile, player)
    require_resources(world, tile, building)
    require_payment(world, player, price)

    # Update the "Buildings" table in place:

    df = world.buildings
    k = (tile, building),

    if df.index.isin(k).any():
        df.loc[k] += 1
    else:
        df.loc[k] = 1

def recruit(world, player, soldiers):
    require_payment(world, player, seacow.RECRUIT_PRICE * soldiers)
    world.player_soldiers[player] += soldiers

def attack(world, player, tile, soldiers):
    require_opponent_tile(world.map, tile, player)
    require_soldiers(world, player, soldiers)

    try:
        battle = require_battle(world.battles, tile)

    # Start a new battle:
    except SkipWithWarning:
        require_explored_tile(world.map, tile, player)
        require_payment(world, player, seacow.ATTACK_PRICE)

        opponent = seacow.who_controls(world.map, tile)
        world.log(f"Tile {tile} is being attacked!", player=opponent)

        world.player_soldiers[player] -= soldiers
        world.battles.loc[len(world.battles)] = {
                'Tile': tile,
                'Begin': world.turn,
                'End': 0,
                'Attacker': player,
                'Defender': opponent,
                'Attacking Soldiers': soldiers,
                'Defending Soldiers': 0,
        }

    # Add soldiers to an existing battle:
    else:
        require_payment(world, player, seacow.ATTACK_PRICE)

        world.player_soldiers[player] -= soldiers
        world.battles.loc[battle, 'Attacking Soldiers'] += soldiers

def defend(world, player, tile, soldiers):
    battle = require_battle(world.battles, tile)
    require_controlled_tile(world.map, tile, player)
    require_soldiers(world, player, soldiers)
    require_payment(world, player, seacow.DEFEND_PRICE)

    world.player_soldiers[player] -= soldiers
    world.battles.loc[battle, 'Defending Soldiers'] += soldiers

def retreat(world, player, tile):
    i = require_battle(world.battles, tile)
    require_payment(world, player, seacow.RETREAT_PRICE)

    battle = world.battles.iloc[i]
    soldier_cols = {
            battle['Attacker']: 'Attacking Soldiers',
            battle['Defender']: 'Defender Soldiers',
    }
    soldier_col = soldier_cols[player]

    world.battles.loc[i, soldier_col] = 0
    world.player_soldiers[player] += battle[soldier_col] // 2

    world.log(f"Player {player} retreated from tile {tile}!", player=battle['Defender'])


def parse_arguments(world, parsers, args):
    if len(args) != len(parsers):
        raise SkipWithWarning(f"expected {len(parsers)} arguments, got {len(args)}")

    values = [f(world, x) for f, x in zip(parsers, args)]

    return values[0] if len(values) == 1 else values

def parse_tile(world, tile_str):
    try:
        tile = int(tile_str)
    except:
        raise SkipWithWarning(f"can't interpret {tile_str!r} as a tile")

    if tile not in world.map.nodes:
        raise SkipWithWarning(f"tile {tile} doesn't exist; valid tile IDs are {min(world.map.nodes)}-{max(world.map.nodes)}")

    return tile

def parse_building(world, building):
    if building not in world.building_prices.index:
        raise SkipWithWarning(f"no such building {building!r}")

    return building

def parse_soldiers(world, soldiers_str):
    try:
        return int(soldiers_str)
    except:
        raise SkipWithWarning(f"can't interpret {soldiers_str!r} as a number of soldiers")

def parse_technology(world, tech):
    if tech not in world.technology_prices.index:
        raise SkipWithWarning(f"no such technology {tech}")

    return tech


def require_unresearched(world, player, tech):
    has_researched  = (
            (tech in world.technologies.index)
            and (player in world.technologies.at[tech, 'Players'].split(','))
            )
    if tech and has_researched:
        raise SkipWithWarning(f"'{tech}' already researched")

def require_research(world, player, tech):
    has_researched  = (
            (tech in world.technologies.index)
            and (player in world.technologies.at[tech, 'Players'].split(','))
            )
    if tech and not has_researched:
        raise SkipWithWarning(f"must research '{tech}'")

def require_payment(world, player, amount):
    if world.player_balances[player] < amount:
        raise TooExpensive(world.player_balances[player], amount)
    else:
        world.player_balances[player] -= amount

def require_explored_tile(map, tile, player):
    if not seacow.player_explored_tile(map, tile, player):
        raise SkipWithWarning(f"must explore tile {tile}")

def require_controlled_tile(map, tile, player):
    if not seacow.player_controls_tile(map, tile, player):
        raise SkipWithWarning(f"must control tile {tile}")

def require_opponent_tile(map, tile, player):
    if not seacow.opponent_controls_tile(map, tile, player):
        raise SkipWithWarning(f"opponent must control tile {tile}")

def require_uncontrolled_tile(map, tile):
    if c := map.nodes[tile]['control']:
        raise SkipWithWarning(f"tile {tile} is already controlled by player {c[-1].player}")

def require_explored_neighbor(map, tile, player):
    for neighbor in map.neighbors(tile):
        if seacow.player_explored_tile(map, neighbor, player):
            return
    raise SkipWithWarning(f"must explore a tile adjacent to {tile}")

def require_controlled_neighbor(map, tile, player):
    for neighbor in map.neighbors(tile):
        if seacow.player_controls_tile(map, neighbor, player):
            return
    raise SkipWithWarning(f"must control a tile adjacent to {tile}")

def require_resources(world, tile, building):
    unused_resources = seacow.count_unused_resources(
            world.map,
            tile,
            world.buildings,
            world.building_resources,
    )
    exhausted_resources = [
            resource
            for resource, (quantity,) in world.building_resources.loc[building].iterrows()
            if quantity > unused_resources.get(resource, 0)
    ]
    if exhausted_resources:
        raise SkipWithWarning(f"not enough {','.join(exhausted_resources)} resources in tile {tile}")

def require_soldiers(world, player, num_soldiers):
    n = world.player_soldiers[player]
    if n < num_soldiers:
        raise SkipWithWarning(f"must have {num_soldiers} soldiers; only have {n}")

def require_battle(battles, tile):
    battle = battles.query('Tile == @tile and End == 0')

    if len(battle) == 0:
        raise SkipWithWarning(f"no battle happening in tile {tile}")
    elif len(battle) == 1:
        return battle.index[0]
    else:
        raise AssertionError("multiple battles happening simultaneously?")


def record_world(doc, world):
    seacow.record_turn(doc, world.turn)

    seacow.record_market_history(doc, world.markets)
    seacow.record_market_shares(doc, world.market_shares)

    seacow.record_map_exploration(doc, world.map_exploration)
    seacow.record_map_control(doc, world.map_control)

    seacow.record_technologies(doc, world.technologies)

    seacow.record_buildings(doc, world.buildings)

    seacow.record_player_logs(doc, world.player_logs)
    seacow.record_player_balances(doc, world.player_balances)
    seacow.record_player_actions(doc, world.pending_actions)
    seacow.record_player_soldiers(doc, world.player_soldiers)
    seacow.record_player_soldiers(doc, world.player_soldiers)
    seacow.record_player_engaged_soldiers(doc, seacow.count_engaged_soldiers(world.battles))

    seacow.record_battles(doc, world.battles)

    try:
        doc.commit()
    except AttributeError:
        pass


if __name__ == '__main__':
    import docopt
    args = docopt.docopt(__doc__)

    doc = seacow.load_doc()
    world = World(doc)

    if args['--interactive']:
        while True:
            update_turn(world)
            record_world(doc, world)
            input(f"Turn {world.turn} complete!  Press [Enter] to start the next turn...")

    else:
        update_turn(world)
        record_world(doc, world)
        print(f"Turn {world.turn} complete!")


