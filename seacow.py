#!/usr/bin/env python3

import gspread
import pandas as pd

def load_doc():
    auth = gspread.oauth()

    # From the URL of the "Seacow market prototype v2" spreadsheet.
    #return auth.open_by_key('126oXnb_MY7JUeRQpGgN6YVsGkKThBzf7udPuKhO1kzw') # Version 2
    return auth.open_by_key('12AbTTUJYCDZp5wqK4vwILPR3h-qdJlaKNRbeqkpCyT8') # Version 3

def load_is_finished(doc):
    sheet = doc.worksheet(f'Game Info')
    is_finished = sheet.acell('A6').value
    return is_finished == 'TRUE'

def load_status(doc):
    sheet = doc.worksheet('Game Info')
    status = sheet.acell('A3').value

    return status

def load_industries(doc):
    return load_df(doc, "Industries")

def load_industry_interactions(doc):
    return load_df(doc, "Industry Interactions")

def load_investments(doc):
    return load_df(doc, "Investments")

def load_investment_effects(doc):
    return load_df(doc, "Investment Effects")

def load_player_income(doc, player):
    df = load_df(doc, f"Player {player}", 'A:B')
    return df

def load_player_purchases(doc, player):
    df = load_df(doc, f"Player {player}", 'D:E')
    return df

def load_df(doc, sheet_name, range=None):
    sheet = doc.worksheet(sheet_name)
    cells = sheet.get_values(range, value_render_option='UNFORMATTED_VALUE')
    return pd.DataFrame(cells[1:], columns=cells[0])


def record_status(doc, status):
    record_cell(doc, 'Game Info', 'A3', status)

def record_industries(doc, df):
    record_df(doc, "Industries", df)

def record_industry_interactions(doc, df):
    clear_sheet(doc, "Industry Interactions")
    record_df(doc, "Industry Interactions", df)

def record_investments(doc, df):
    clear_sheet(doc, "Investments")
    record_df(doc, "Investments", df)

def record_investment_effects(doc, df):
    clear_sheet(doc, "Investment Effects")
    record_df(doc, "Investment Effects", df)

def record_player_income(doc, player, df):
    record_df(doc, f"Player {player}", df, range='A:B')

def record_player_income_breakdown(doc, player, df):
    record_df(doc, f"Player {player}", df, range='I:J')

def record_global_market(doc, player, df):
    record_df(doc, f"Player {player}", df, range='H:K')

def record_player_intel(doc, player, df):
    clear_sheet(doc, f"Player {player} Intel")
    record_df(doc, f"Player {player} Intel", df)

def record_player_purchases(doc, player, df):
    #clear_sheet(doc, f"Player {player}", range='D:E')
    record_df(doc, f"Player {player}", df, range='D:E')

def record_df(doc, sheet_name, df, range=None):
    df = pd.DataFrame(df)
    sheet = doc.worksheet(sheet_name)
    cells = [df.columns.values.tolist()] + df.values.tolist()

    if range:
        sheet.update(range, cells)
    else:
        sheet.update(cells)

def record_cell(doc, sheet_name, cell_index, value):
    sheet = doc.worksheet(sheet_name)
    sheet.update(cell_index, value)


def clear_sheet(doc, sheet_name, range=None):
    sheet = doc.worksheet(sheet_name)

    if range:
        sheet.batch_clear([range])
    else:
        # clear the entire sheet
        sheet.clear()
def reset_is_finished(doc, player):
    record_cell(doc, f'Player {player}', 'G5', False)
