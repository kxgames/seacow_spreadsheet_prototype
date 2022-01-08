#!/usr/bin/env python3

import gspread

auth = gspread.oauth()

# From the URL of the "Seacow market prototype v2" spreadsheet.
doc = auth.open_by_key('126oXnb_MY7JUeRQpGgN6YVsGkKThBzf7udPuKhO1kzw')

sheet_name = "Hello World"
try:
    sheet = doc.worksheet(sheet_name)
    print(f"Cell A1 contains: {sheet.acell('A1').value}")

except gspread.exceptions.WorksheetNotFound:
    print((f'Spreadsheet document "{doc.title}" '
        'does not have the sheet "{sheet_name}"'))
