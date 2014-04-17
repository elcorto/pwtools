#!/bin/bash

# Print some raw results from the database.

sqlite3 -column -header calc.db "select idx,volume,etot,target_press/1e4,pressure from calc"
