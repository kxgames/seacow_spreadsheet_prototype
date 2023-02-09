#!/usr/bin/env python3

import os, gspread
import pandas as pd

PLAYERS = ['1', '2']

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

def load_is_finished(doc):
    sheet = doc.worksheet(f'Game Info')
    is_finished = sheet.acell('A6').value
    return is_finished == 'TRUE'

def load_status(doc):
    sheet = doc.worksheet('Game Info')
    status = sheet.acell('A3').value

    return status

def load_markets(doc):
    return load_df(doc, 'Markets', index=['Market'])

def load_elasticities(doc):
    return load_df(doc, 'Elasticities', index=['Market', 'Related Market'])

def load_investments(doc):
    return load_df(doc, 'Investments', index=['Investment'])

def load_investment_effects(doc):
    return load_df(doc, 'Investment Effects', index=['Investment', 'Market'])

def load_map_tiles(doc):
    return load_df(doc, 'Map Tiles', index=['Tile'])

def load_map_edges(doc):
    return load_df(doc, 'Map Edges')

def load_map_resources(doc):
    return load_df(doc, 'Map Resources', index=['Tile'])

def load_player_incomes(doc):
    return load_player_df(doc, range='A:B', index=['Player', 'Turn'])

def load_player_new_investments(doc):
    return load_player_df(doc, range='D:D')

def load_player_existing_investments(doc):
    return load_player_df(doc, range='E:E')

def load_df(doc, sheet_name, *, range=None, index=None):
    sheet = doc.worksheet(sheet_name)
    cells = sheet.get_values(range, value_render_option='UNFORMATTED_VALUE')
    df = pd.DataFrame(cells[1:], columns=cells[0])

    if index:
        return df.set_index(index).sort_index()
    else:
        return df

def load_player_df(doc, *, range=None, index=None):
    """
    Load a dataframe containing information of each player.

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

    return df.set_index(index or 'Player').sort_index()


def record_status(doc, status):
    record_cell(doc, 'Game Info', 'A3', status)

def record_markets(doc, df):
    record_df(doc, "Markets", df, index=True)

def record_elasticities(doc, df):
    clear_sheet(doc, 'Elasticities')
    record_df(doc, 'Elasticities', df)

def record_investments(doc, df):
    clear_sheet(doc, 'Investments')
    record_df(doc, 'Investments', df)

def record_investment_effects(doc, df):
    clear_sheet(doc, 'Investment Effects')
    record_df(doc, 'Investment Effects', df)

def record_map_tiles(doc, tile_locations_df, edges_df):
    clear_sheet(doc, 'Map Tiles')
    record_df(doc, 'Map Tiles', tile_locations_df)

    clear_sheet(doc, 'Map Edges')
    record_df(doc, 'Map Edges', edges_df)

def record_map_resources(doc, resources_df):
    clear_sheet(doc, 'Map Resources')
    record_df(doc, 'Map Resources', resources_df)

def record_player_incomes(doc, df):
    record_player_df(doc, df, range='A:B', index=True)

def record_player_income_breakdown(doc, player, df):
    record_df(doc, f'Player {player}', df, range='I:J')

def record_global_market(doc, player, df):
    record_df(doc, f'Player {player}', df, range='H:K')

def record_player_intel(doc, player, df):
    clear_sheet(doc, f'Player {player} Intel')
    record_df(doc, f'Player {player} Intel', df)

def record_player_investments(doc, df):
    df = df.sort_values('Existing Investments')
    record_player_df(doc, df, range='E:E')

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

def record_cell(doc, sheet_name, cell_index, value):
    sheet = doc.worksheet(sheet_name)
    sheet.update(cell_index, value)


def clear_player_new_investments(doc):
    for player in PLAYERS:
        clear_sheet(doc, f'Player {player}', 'D2:D')

def clear_sheet(doc, sheet_name, range=None):
    sheet = doc.worksheet(sheet_name)

    if range:
        sheet.batch_clear([range])
    else:
        # clear the entire sheet
        sheet.clear()

def reset_is_finished(doc, player):
    record_cell(doc, f'Player {player}', 'G5', False)
