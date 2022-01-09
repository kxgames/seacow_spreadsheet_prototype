Seacow Spreadsheet Prototype
============================
This repository contains a handful scripts to help run the seacow prototype.

Usage
-----
Calculate a new industry interaction graph:

1. Fill in the "Industries" spreadsheet with as many industries as you'd like. 
   Enter values for Industry (A), Price (B), and Implicit Supply/Demand (I/J). 
   Pull down the formulas for the remaining columns (C-H).

2. `./make_industry_interactions.py -h`

   This script will fill in the "Industry Interactions" spreadsheet and output 
   an SVG file illustrating the interaction graph.  You can set a custom random 
   seed to generate different graphs.

Create "investments" for the players to purchase:

1. Fill in the "Industries" and "Industry Interactions" graphs as described 
   above.

2. `./make_investments.py`

   This script will fill in the "Investments" and "Investment Effects" 
   spreadsheets.

Play the prototype:

1. Each player should open their spreadsheet (e.g. "Player 1" or "Player 2") 
   and reset any purchases and all incomes except for the first.

2. On each turn, players decide what they want to purchase and enter it into 
   the "Purchases" column on their spreadsheet.  The "Balance" value should 
   update automatically.

3. When both players have made all the purchases they want to, one player 
   should run the `update_turn.py` script.  This will evaluate the state of the 
   economy and add a row to each players' turn/income table accordingly.

Authentication
--------------
`gspread` authentication tokens expire after a week.  You need to manually 
remove the authentication file to repeat the authentication process:
```
rm ~/.config/gspread/authorized_user.json
```

After that, just run any of the scripts like normal and you will be prompted 
for permission again.
