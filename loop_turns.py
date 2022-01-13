#!/usr/bin/env python3

"""\
Usage:
    loop_turns.py [-t <seconds>]

Options:
    -t --time-limit <int>      [default: 0]
        Set the time limit for turns (should be multiple of 10). Zero means no time limit.

"""

import docopt
from time import sleep
import numpy as np

import seacow
import update_turn

def reset_both_players_finished(doc):
    seacow.reset_is_finished(doc, 1)
    seacow.reset_is_finished(doc, 2)

args = docopt.docopt(__doc__)
time_limit = int(args['--time-limit'])
time_limit -= time_limit % 10

doc = seacow.load_doc()

# Usage limits for Google Sheets
# (from https://developers.google.com/sheets/api/reference/limits)
#
#       This version of the Google Sheets API has a limit of 500 requests per 
#       100 seconds per project, and 100 requests per 100 seconds per user.  
#       Limits for reads and writes are tracked separately. There is no daily 
#       usage limit.
#
sleep_time = 10
# Do not make sleep_time less than 5 or the loop will easily hit the google 
# read/write limits. It sometime crashes with 5 s depending on how quickly 
# turns are finished. For now 10 s seems to be the reliable choice.

# Loop forever. The host has to Ctrl-C to end the game.
while True:
    # Start turn
    seacow.record_status(doc, "Starting turn")

    time_left = time_limit if time_limit > 0 else np.inf
    reset_both_players_finished(doc)
    players_finished = False

    # Wait for the turn clock to run out or both players have finished early
    while time_left > 0 and not players_finished:
        print(f"time_left {time_left}")
        
        # Let the players know how much time is left (if there are time limits)
        time_msg = " " if time_left == np.inf else f" (~{time_left} s)"
        seacow.record_status(doc, f"Play your turn{time_msg}")

        # Sleep so we don't hit the read/write limits
        sleep(sleep_time)

        # Update the timer and check if players are finished
        time_left -= sleep_time
        try:
            players_finished = seacow.load_is_finished(doc)
        except gspread.exceptions.APIError as err:
            if "try again in 30 seconds" in err.message:
                print("Unknown error:")
                print(err.message)
                print()
                print("Attempting to start again in 30 seconds")
                sleept(30)
                time_left -= 30
            else:
                raise


    # Turn ending, update the spreadsheets
    update_turn.update_turn()
