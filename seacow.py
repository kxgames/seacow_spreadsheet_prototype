#!/usr/bin/env python3

import gspread
import offline_mode
import pandas as pd
import networkx as nx
import backoff
import os, sys
import networkx as nx

from dataclasses import dataclass

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
        giveup=lambda err: err.response.json()['error']['code'] != 429,
        on_backoff=lambda details: print(
            "Backing off {wait:.2f} seconds after {tries} tries".format_map(details),
            file=sys.stderr,
        ),
)

@retry_politely
def load_doc(version=None):
    if xlsx_in := os.environ.get('SEACOW_OFFLINE_MODE'):
        xlsx_out = os.environ.get('SEACOW_OFFLINE_MODE_OUTPUT_PATH')
        return offline_mode.load_doc(xlsx_in, xlsx_out)
        
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

def load_map(doc):
    return compose_map(
            tiles=load_map_tiles(doc),
            edges=load_map_edges(doc),
            resources=load_map_resources(doc),
            exploration=load_map_exploration(doc),
            control=load_map_control(doc),
            battles=load_battles(doc),
    )

def load_map_tiles(doc):
    return load_df(doc, 'Map Tiles', index=['Tile'])

def load_map_edges(doc):
    return load_df(doc, 'Map Edges')

def load_map_resources(doc):
    return load_df(doc, 'Map Resources', index=['Tile'])

def load_map_exploration(doc):
    return load_df(doc, 'Map Exploration', dtypes={'Player': 'string'})

def load_map_control(doc):
    return load_df(
            doc,
            'Map Control',
            index=['Tile', 'Turn'],
            dtypes={'Player': 'string'},
    )

def load_player_logs(doc):
    return load_player_df(doc, range='G:H')

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
    return load_df(
            doc,
            'Battles',
            dtypes={'Attacker': 'string', 'Defender': 'string'},
    )

@retry_politely
def load_df(doc, sheet_name, *, range=None, index=None, dtypes=None):
    sheet = doc.worksheet(sheet_name)
    cells = sheet.get_values(range, value_render_option='UNFORMATTED_VALUE')
    df = pd.DataFrame(cells[1:], columns=cells[0])

    if dtypes:
        for col, dtype in dtypes.items():
            df[col] = df[col].astype(dtype)

    if index:
        return df.set_index(index).sort_index()
    else:
        return df

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

    record_player_df(doc, df, range='G:H')

def record_player_balances(doc, values):
    record_player_dict(doc, 'B1', values)

def record_player_actions(doc, df):
    for player in PLAYERS:
        clear_sheet(doc, f'Player {player}', 'D2:D')

    print(df)

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

def record_player_df(doc, df, *, range=None, cols=None, index=False):
    for player, group in df.groupby('Player'):
        record_df(
                doc,
                f'Player {player}',
                group.drop(columns='Player'),
                range=range,
                cols=cols,
                index=False,
        )

def record_player_dict(doc, cell, values):
    for player, value in values.items():
        record_cell(doc, f'Player {player}', cell, value)

@retry_politely
def record_cell(doc, sheet_name, cell, value):
    sheet = doc.worksheet(sheet_name)
    sheet.update(cell, value)


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
                control=[], # [control_obj, ...]
                explore=defaultdict(list), #{player : [turns explored]}
        )

    for _, (a, b) in edges.iterrows():
        map.add_edge(a, b)

    for tile, row in resources.iterrows():
        resource = row['Resource']
        map.nodes[tile]['resources'].setdefault(resource, 0)
        map.nodes[tile]['resources'][resource] += RESOURCE_FACTOR

    for (tile, turn), row in control.sort_index().iterrows():
        ctrl = Control(turn, row['Player'])
        map.nodes[tile]['control'].append(ctrl)

    for _, (tile, turn, player) in exploration.iterrows():
        map.nodes[tile]['explore'][player].append(turn)

    return map

def who_controls(map, tile, return_none=False):
    control = map.nodes[tile]['control']
    if not control:
        if return_none:
            return None
        else:
            raise ValueError(f"tile {tile} is not controlled by either player")
    return control[-1].player

def when_controlled(map, tile, player):
    control = map.nodes[tile]['control']
    if (not control) or control[-1].player != player:
        raise ValueError(f"tile {tile} is not controlled by player {player}")
    return control[-1].turn

def when_explored(map, tile, player):
    exploration = map.nodes[tile]['explore']
    if player not in exploration:
        raise ValueError(f"tile {tile} not explored yet by player")
    return exploration[player][-1]

def count_resources(map, tile):
    resources = defaultdict(lambda: 0)

    for resource in map.nodes[tile]['resources']:
        resources[resource] += RESOURCE_FACTOR

    return resources
def player_controls_tile(map, tile, player):
    try:
        return who_controls(map, tile) == player
    except ValueError:
        return False

def player_explored_tile(map, tile, player):
    """
    Return the most recent turn that the given player explored the given tile, 
    or 0 if the player never explored the tile.
    """
    return player_controls_tile(map, tile, player) or \
            map.nodes[tile]['explore'].get(player, [0])[-1]

def opponent_controls_tile(map, tile, player):
    try:
        return who_controls(map, tile) != player
    except ValueError:
        return False

def count_used_resources(tile, buildings, building_resources):
    if buildings.empty:
        return pd.DataFrame([], columns=['Resources Used'])

    df = pd.merge(
            buildings.loc[tile].rename(
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
    if not resources_used_per_building.empty:
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

