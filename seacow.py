#!/usr/bin/env python3

import gspread
import pandas as pd
import backoff
import os, sys

from dataclasses import dataclass
from collections import defaultdict

PLAYERS = ['1', '2']
EXPLORE_PRICE = 50
EXPAND_PRICE = 50
RECRUIT_PRICE = 50
ATTACK_PRICE = 100
DEFEND_PRICE = 50
RETREAT_PRICE = 100
KILL_PROBABILITY = 0.5
RESOURCE_FACTOR = 5

retry_politely = backoff.on_exception(
        backoff.expo,
        gspread.exceptions.APIError,
        on_backoff=lambda details: print(
            "Backing off {wait:.2f} secconds after {tries} tries".format_map(details),
            file=sys.stderr,
        ),
)

@retry_politely
def load_doc(version=None):
    auth = gspread.oauth()

    # These keys come directly from the URLs of the spreadsheets.
    keys = {
            2: '126oXnb_MY7JUeRQpGgN6YVsGkKThBzf7udPuKhO1kzw',
            3: '12AbTTUJYCDZp5wqK4vwILPR3h-qdJlaKNRbeqkpCyT8',
            4: '1tSY8ikutY26WTXYAln9sqQh62_uaOZFEE0tsTYZ9Aec',
            5: '1RmRUbX_GW3aQ3zkZk4QSGB8cuuF-fNDqo4riq4O0nO4',
            6: '1VSEL1xCtHwKtDx1umrhD1Hh1Km3cvWj8HLjxL_C87Pg',
    }

    if not version:
        version = int(os.environ.get('SEACOW_VERSION', 0))
    if not version:
        version = list(keys)[-1]

    return auth.open_by_key(keys[version])

def load_turn(doc):
    return int(load_cell(doc, 'Game Info', 'B1', default=1))

def load_markets(doc):
    return load_df(doc, 'Markets', index=['Market'])

def load_elasticities(doc):
    return load_df(doc, 'Elasticities', index=['Market', 'Related Market'])

def load_map(doc, actions):
    return compose_map(
            tiles=seacow.load_map_tiles(doc),
            edges=seacow.load_map_edges(doc),
            resources=seacow.load_map_resources(doc),
            exploration=seacow.load_map_exploration(doc),
            control=seacow.load_map_control(doc),
            battles=seacow.load_battles(doc),
    )

def load_map_tiles(doc):
    return load_df(doc, 'Map Tiles', index=['Tile'])

def load_map_edges(doc):
    return load_df(doc, 'Map Edges')

def load_map_resources(doc):
    return load_df(doc, 'Map Resources', index=['Tile'])

def load_map_exploration(doc):
    return load_df(doc, 'Map Exploration')

def load_map_control(doc):
    return load_df(doc, 'Map Control', index=['Tile', 'Turn'])

def load_player_logs(doc):
    return load_player_df(doc, range='A:B')

def load_player_balances(doc):
    return load_player_dict(doc, 'B1', required=True, cast=float)

def load_player_actions(doc):
    return load_player_df(doc, range='D:D')

def load_player_soldiers(doc):
    return load_player_dict(doc, 'B2', default=0, cast=int)

def load_buildings(doc):
    return load_df(doc, 'Buildings', index=['Tile', 'Building'])

def load_building_prices(doc):
    return load_df(doc, 'Building Prices', index=['Building'])

def load_building_resources(doc):
    return load_df(doc, 'Building Resources', index=['Building', 'Resource'])

def load_building_effects(doc):
    return load_df(doc, 'Building Effects', index=['Building', 'Market'])

def load_battles(doc):
    return load_df(doc, 'Battles')

@retry_politely
def load_df(doc, sheet_name, *, range=None, index=None):
    sheet = doc.worksheet(sheet_name)
    cells = sheet.get_values(range, value_render_option='UNFORMATTED_VALUE')
    df = pd.DataFrame(cells[1:], columns=cells[0])

    if index:
        return df.set_index(index).sort_index()
    else:
        return df

@retry_politely
def load_player_df(doc, *, range=None, index=None):
    """
    Load a data frame containing information on each player.

    This is a special case, because the information for each player is stored 
    in a different spreadsheet.  So this method loads all the relevant 
    spreadsheets, concatenates the data contained therein, and annotates which 
    player each datum refers to.
    """
    dfs = []

    for player in PLAYERS:
        df = load_df(doc, f'Player {player}', range=range)
        df['Player'] = player
        dfs.append(df)

    df = pd.concat(dfs).dropna(how='all')

    if index:
        df = df.set_index(index).sort_index()

    return df

def load_player_dict(doc, cell, *, default=None, required=False, cast=None):
    return {
            player: load_cell(
                doc,
                f'Player {player}',
                cell,
                default=default,
                required=required,
                cast=cast,
            )
            for player in PLAYERS
    }

@retry_politely
def load_cell(doc, sheet_name, cell, *, default=None, required=False, cast=None):
    sheet = doc.worksheet(sheet_name)
    value = sheet.acell(cell, value_render_option='UNFORMATTED_VALUE').value

    if value is None:
        value = default

    if value is None and required:
        raise ValueError(f"expected a value for {sheet_name!r}!{cell}")

    if cast:
        value = cast(value)

    return value


def record_turn(doc, turn):
    record_cell(doc, 'Game Info', 'B2', turn)

def record_market_history(doc, df):
    """
    Add a row to the "Market History" spreadsheet, given the updated market data 
    frame.
    """
    old_records = load_df(doc, 'Market History')

    cols = ['Quantity', 'Price', 'Total Value']
    new_record = df.drop(columns=df.columns.difference(cols)).reset_index()
    new_record['Turn'] = 1 if old_records.empty else old_records['Turn'].max() + 1
    new_record = new_record[['Turn', 'Market', *cols]]

    all_records = pd.concat([new_record, old_records])
    record_df(doc, 'Market History', all_records)

def record_market_shares(doc, df):
    record_df(doc, 'Market Shares', df, index=True)

def record_elasticities(doc, df):
    clear_sheet(doc, 'Elasticities')
    record_df(doc, 'Elasticities', df)

def record_map_tiles(doc, tile_locations_df, edges_df):
    clear_sheet(doc, 'Map Tiles')
    record_df(doc, 'Map Tiles', tile_locations_df)

    clear_sheet(doc, 'Map Edges')
    record_df(doc, 'Map Edges', edges_df)

def record_map_resources(doc, df):
    clear_sheet(doc, 'Map Resources')
    record_df(doc, 'Map Resources', df)

def record_map_exploration(doc, df):
    record_df(doc, 'Map Exploration', df)

def record_map_control(doc, df):
    clear_sheet(doc, 'Map Control')
    record_df(doc, 'Map Control', df, index=True)

def record_player_logs(doc, df):
    # TODO:
    # - Grey-out older log entries:
    #
    #   https://docs.gspread.org/en/latest/api/models/worksheet.html#gspread.worksheet.Worksheet.format
    #   https://developers.google.com/sheets/api/reference/rest/v4/spreadsheets/cells#cellformat
    #
    #   >>> sheet.format(cell, {...})

    df = df.sort_values('Turn', ascending=False)

    record_player_df(doc, df, range='A:B')

def record_player_balances(doc, values):
    record_player_dict(doc, 'B1', values)

def record_player_actions(doc, df):
    for player in PLAYERS:
        clear_sheet(doc, f'Player {player}', 'D2:D')

    record_player_df(doc, df, range='D:E')

def record_player_soldiers(doc, values):
    record_player_dict(doc, 'B2', values)

def record_player_engaged_soldiers(doc, values):
    record_player_dict(doc, 'B3', values)

def record_buildings(doc, df):
    record_df(doc, 'Buildings', df, index=True)

def record_building_prices(doc, df):
    clear_sheet(doc, 'Building Prices')
    record_df(doc, 'Building Prices', df)

def record_building_resources(doc, df):
    clear_sheet(doc, 'Building Resources')
    record_df(doc, 'Building Resources', df)

def record_building_effects(doc, df):
    clear_sheet(doc, 'Building Effects')
    record_df(doc, 'Building Effects', df)

def record_battles(doc, df):
    record_df(doc, 'Battles', df)

@retry_politely
def record_df(doc, sheet_name, df, *, range=None, cols=None, index=False):
    """
    Upload the given data frame to the given Google Docs spreadsheet.

    Arguments:
        range:
            Write the data frame values to the given range, e.g. 'A:B'.  By 
            default, the data frame will overwrite the whole sheet.

        cols:
            Only upload the given columns (in the given order).  By default, 
            every column in the given data frame will be uploaded.

        index:
            Upload the index of the data frame (along with its columns).  If 
            the index has multiple levels, each will be treated as a separate 
            column.  If *cols* is specified, only the specified index levels 
            will be recorded.
    """
    
    if index:
        df = df.reset_index()
    if cols:
        df = df[cols]

    sheet = doc.worksheet(sheet_name)
    cells = [df.columns.values.tolist()] + df.values.tolist()

    if range:
        sheet.update(range, cells)
    else:
        sheet.update(cells)

@retry_politely
def record_player_df(doc, df, *, range=None, cols=None, index=False):
    for player in PLAYERS:

        # The double-brackets force `.loc` to generate a data frame instead of 
        # converting to a series when the data have one one column.  This is 
        # important because `record_df()` res an actual data frame.  It's also 
        # not trivial to recreate a data frame from a series.  I tried to do 
        # this, but kept getting a bug where pandas would mix up the rows and 
        # columns of the data frame if the series only had one entry.

        try:
            dfi = df.loc[[player]]
        except KeyError:
            continue

        # If the user asks for the index to be recorded, we don't want to 
        # include the "Player" level.  Removing index levels is complicated, 
        # though (you have to do it differently depending on how many index 
        # levels there are), so here we just convert any index levels into 
        # columns, then remove the player column.

        if index:
            dfi = dfi.reset_index().drop('Player', axis=1)

        record_df(
                doc,
                f'Player {player}',
                dfi,
                range=range,
                cols=cols,
                index=False,
        )

def record_player_dict(doc, cell, values):
    for player, value in values.items():
        record_cell(doc, f'Player {player}', cell, value)

@retry_politely
def record_cell(doc, sheet_name, cell_index, value):
    sheet = doc.worksheet(sheet_name)
    sheet.update(cell_index, value)


@retry_politely
def clear_sheet(doc, sheet_name, range=None):
    sheet = doc.worksheet(sheet_name)

    if range:
        sheet.batch_clear([range])
    else:
        sheet.clear()


def compose_map(*, tiles, edges, resources, exploration, control, battles):
    # Node attributes:
    # - x, y: location of the tile
    # - resources: list of strings, can have duplicates
    # - control: list of (name, turn) tuples
    # - explore: the most recent turn each player explored this tile
    #
    map = nx.Graph()
    for tile, (x, y) in tiles.iterrows():
        map.add_node(
                tile,
                x=x,
                y=y,
                resources={},
                control=[],
                explore={},
        )

    for _, (a, b) in edges.iterrows():
        map.add_edge(a, b)

    for tile, (resource,) in resources.iterrows():
        map.nodes[tile]['resources'].setdefault(resource, 0)
        map.nodes[tile]['resources'][resource] += RESOURCE_FACTOR

    for _, (tile, turn, player) in control.iterrows():
        map.nodes[tile]['control'].append()

    return map

def who_controls(map, tile):
    control = map.modes[tile]['control']
    if not control:
        raise ValueError(f"tile {tile} is not controlled by either player")
    return control[-1].player

def count_resources(map, tile):
    resources = defaultdict(lambda: 0)

    for resource in map.nodes[tile]['resources']:
        resources[resource] += RESOURCE_FACTOR

    return resources

def player_controls_tile(map, tile, player):
    control = map[tile]['control']
    return control and control[-1].player == player

def player_explored_tile(map, tile, player):
    """
    Return the most recent turn that the given player explored the given tile, 
    or 0 if the player never explored the tile.
    """
    return map[tile]['explore'].get(player, [0])[-1]

def opponent_controls_tile(map, tile, player):
    control = map[tile]['control']
    return control and control[-1].player != player

def count_used_resources(map, tile, buildings, building_resources):
    player = who_controls(map, tile)
    df = pd.merge(
            player_buildings.loc[tile].rename(
                {'Quantity': 'Num Buildings'}, axis=1,
            ),
            building_resources.rename(
                {'Quantity': 'Resources Per'}, axis=1,
            ),
            left_index=True,
            right_index=True,
    )
    df['Resources Used'] = df['Num Buildings'] * df['Resources Per']
    return df

def count_unused_resources(map, tile, buildings, building_resources):
    resources = map.nodes[tile]['resources']
    resources_used_per_building = count_used_resources(
            tile,
            buildings,
            building_resources,
    )
    resources_used = resources_used_per_building\
            .groupby('Resource')['Resources Used'].sum()

    for k in resources:
        resources[k] -= resources_used[k]

    return resources

def count_engaged_soldiers(battles):
    soldiers = {k: 0 for k in PLAYERS}

    for _, battle in battles.iterrows():
        soldiers[battle['Attacker']] += battle['Attacking Soldiers']
        soldiers[battle['Defender']] += battle['Defending Soldiers']

    return soldiers


@dataclass(order=True)
class Control:
    turn: int
    player: str



