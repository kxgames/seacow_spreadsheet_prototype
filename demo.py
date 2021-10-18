#!/usr/bin/env python3

import gspread

auth = gspread.oauth()

# From the URL of the "Seacow market prototype v2" spreadsheet.
doc = auth.open_by_key('126oXnb_MY7JUeRQpGgN6YVsGkKThBzf7udPuKhO1kzw')

sheet = doc.worksheet("Hello World")

print(sheet.acell('A1').value)


