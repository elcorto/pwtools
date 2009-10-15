# regex.py
#
# Definition of useful regexes thatwe may use for parsing here and there.

# Regex that matched every conveivable form of a float number, also Fortran 
# 1
# 1.0
# +1.0
# -1.0
# 1.0e3
# 1.0e+03
# 1.0E-003
# -.1D03
# ...
float_re = r'[+-]*[\.0-9eEdD+-]+'
