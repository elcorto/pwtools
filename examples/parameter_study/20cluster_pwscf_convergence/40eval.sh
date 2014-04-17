#!/bin/bash

# Print some raw results from the database.

sqlite3 -column -header calc.db "select idx,ecutwfc,etot,pressure from calc"
