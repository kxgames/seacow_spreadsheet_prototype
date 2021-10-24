#!/usr/bin/env python3

import gspread
import pandas as pd

def load_doc():
    auth = gspread.oauth()

    # From the URL of the "Seacow market prototype v2" spreadsheet.
    return auth.open_by_key('126oXnb_MY7JUeRQpGgN6YVsGkKThBzf7udPuKhO1kzw')

def load_industries(doc):
    return load_df(doc, "Industries")

def load_industry_interactions(doc):
    return load_df(doc, "Industry Interactions")

def load_investments(doc):
    return load_df(doc, "Investments")

def load_investment_effects(doc):
    return load_df(doc, "Investment Effects")

def load_df(doc, sheet_name):
    sheet = doc.worksheet(sheet_name)
    records = sheet.get_all_records()
    return pd.DataFrame(records)


def record_industries(doc, df):
    record_df(doc, "Industries", df)

def record_industry_interactions(doc, df):
    record_df(doc, "Industry Interactions", df)

def record_investments(doc, df):
    record_df(doc, "Investments", df)

def record_investment_effects(doc, df):
    record_df(doc, "Investment Effects", df)

def record_df(doc, sheet_name, df):
    df = pd.DataFrame(df)
    sheet = doc.worksheet(sheet_name)
    sheet.update([df.columns.values.tolist()] + df.values.tolist())


