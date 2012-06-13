# The periodic table. 
#
# Got that from [1]. Finding such a list on the net is not as easy as you might
# think. Search for "list of chemical elements".
#
# Replaced masses like
#   9.012182(3) -> 9.012182
#   [227]       -> 227
#
# Covalent radii shamelessly stolen from ASE.
#
# [1] http://chemistry.about.com/od/elementfacts/a/atomicweights.htm

import numpy as np

missing = -1

symbols = [
    'Xx',
    'H',
    'He',
    'Li',
    'Be',
    'B',
    'C',
    'N',
    'O',
    'F',
    'Ne',
    'Na',
    'Mg',
    'Al',
    'Si',
    'P',
    'S',
    'Cl',
    'Ar',
    'K',
    'Ca',
    'Sc',
    'Ti',
    'V',
    'Cr',
    'Mn',
    'Fe',
    'Co',
    'Ni',
    'Cu',
    'Zn',
    'Ga',
    'Ge',
    'As',
    'Se',
    'Br',
    'Kr',
    'Rb',
    'Sr',
    'Y',
    'Zr',
    'Nb',
    'Mo',
    'Tc',
    'Ru',
    'Rh',
    'Pd',
    'Ag',
    'Cd',
    'In',
    'Sn',
    'Sb',
    'Te',
    'I',
    'Xe',
    'Cs',
    'Ba',
    'La',
    'Ce',
    'Pr',
    'Nd',
    'Pm',
    'Sm',
    'Eu',
    'Gd',
    'Tb',
    'Dy',
    'Ho',
    'Er',
    'Tm',
    'Yb',
    'Lu',
    'Hf',
    'Ta',
    'W',
    'Re',
    'Os',
    'Ir',
    'Pt',
    'Au',
    'Hg',
    'Tl',
    'Pb',
    'Bi',
    'Po',
    'At',
    'Rn',
    'Fr',
    'Ra',
    'Ac',
    'Th',
    'Pa',
    'U',
    'Np',
    'Pu',
    'Am',
    'Cm',
    'Bk',
    'Cf',
    'Es',
    'Fm',
    'Md',
    'No',
    'Lr',
    'Rf',
    'Db',
    'Sg',
    'Bh',
    'Hs',
    'Mt',
    'Ds',
    'Rg',
    'Uub',
    'Uut',
    'Uuq',
    'Uup',
    'Uuh',
    'Uuo',
    ]

masses = np.array([\
    missing,
    1.00794,
    4.002602,
    6.941,
    9.012182,
    10.811,
    12.0107,
    14.0067,
    15.9994,
    18.9984032,
    20.1797,
    22.98976928,
    24.3050,
    26.9815386,
    28.0855,
    30.973762,
    32.065,
    35.453,
    39.948,
    39.0983,
    40.078,
    44.955912,
    47.867,
    50.9415,
    51.9961,
    54.938045,
    55.845,
    58.933195,
    58.6934,
    63.546,
    65.38,
    69.723,
    72.64,
    74.92160,
    78.96,
    79.904,
    83.798,
    85.4678,
    87.62,
    88.90585,
    91.224,
    92.90638,
    95.96,
    98,
    101.07,
    102.90550,
    106.42,
    107.8682,
    112.411,
    114.818,
    118.710,
    121.760,
    127.60,
    126.90447,
    131.293,
    132.9054519,
    137.327,
    138.90547,
    140.116,
    140.90765,
    144.242,
    145,
    150.36,
    151.964,
    157.25,
    158.92535,
    162.500,
    164.93032,
    167.259,
    168.93421,
    173.054,
    174.9668,
    178.49,
    180.94788,
    183.84,
    186.207,
    190.23,
    192.217,
    195.084,
    196.966569,
    200.59,
    204.3833,
    207.2,
    208.98040,
    209,
    210,
    222,
    223,
    226,
    227,
    232.03806,
    231.03588,
    238.02891,
    237,
    244,
    243,
    247,
    247,
    251,
    252,
    257,
    258,
    259,
    262,
    267,
    268,
    271,
    272,
    270,
    276,
    281,
    280,
    285,
    284,
    289,
    288,
    293,
    294,
    ])

# Angstrom
covalent_radii = np.array([
    missing,  # Xx  
    0.31,  # H 
    0.28,  # He     
    1.28,  # Li     
    0.96,  # Be     
    0.84,  # B      
    0.76,  # C      
    0.71,  # N      
    0.66,  # O      
    0.57,  # F      
    0.58,  # Ne     
    1.66,  # Na     
    1.41,  # Mg     
    1.21,  # Al     
    1.11,  # Si     
    1.07,  # P      
    1.05,  # S      
    1.02,  # Cl     
    1.06,  # Ar     
    2.03,  # K      
    1.76,  # Ca     
    1.70,  # Sc     
    1.60,  # Ti     
    1.53,  # V      
    1.39,  # Cr     
    1.39,  # Mn     
    1.32,  # Fe     
    1.26,  # Co     
    1.24,  # Ni     
    1.32,  # Cu     
    1.22,  # Zn     
    1.22,  # Ga     
    1.20,  # Ge     
    1.19,  # As     
    1.20,  # Se     
    1.20,  # Br     
    1.16,  # Kr     
    2.20,  # Rb     
    1.95,  # Sr     
    1.90,  # Y      
    1.75,  # Zr     
    1.64,  # Nb     
    1.54,  # Mo     
    1.47,  # Tc     
    1.46,  # Ru     
    1.42,  # Rh     
    1.39,  # Pd     
    1.45,  # Ag     
    1.44,  # Cd     
    1.42,  # In     
    1.39,  # Sn     
    1.39,  # Sb     
    1.38,  # Te     
    1.39,  # I      
    1.40,  # Xe     
    2.44,  # Cs     
    2.15,  # Ba     
    2.07,  # La     
    2.04,  # Ce     
    2.03,  # Pr     
    2.01,  # Nd     
    1.99,  # Pm     
    1.98,  # Sm     
    1.98,  # Eu     
    1.96,  # Gd     
    1.94,  # Tb     
    1.92,  # Dy     
    1.92,  # Ho     
    1.89,  # Er     
    1.90,  # Tm     
    1.87,  # Yb     
    1.87,  # Lu     
    1.75,  # Hf     
    1.70,  # Ta     
    1.62,  # W      
    1.51,  # Re     
    1.44,  # Os     
    1.41,  # Ir     
    1.36,  # Pt     
    1.36,  # Au     
    1.32,  # Hg     
    1.45,  # Tl     
    1.46,  # Pb     
    1.48,  # Bi     
    1.40,  # Po     
    1.50,  # At     
    1.50,  # Rn     
    2.60,  # Fr     
    2.21,  # Ra     
    2.15,  # Ac     
    2.06,  # Th     
    2.00,  # Pa     
    1.96,  # U      
    1.90,  # Np     
    1.87,  # Pu     
    1.80,  # Am     
    1.69,  # Cm     
    missing,  # Bk  
    missing,  # Cf  
    missing,  # Es  
    missing,  # Fm  
    missing,  # Md  
    missing,  # No  
    missing,  # Lr  
    missing,  # Rf  
    missing,  # Db  
    missing,  # Sg  
    missing,  # Bh  
    missing,  # Hs  
    missing,  # Mt  
    missing,  # Ds  
    missing,  # Rg  
    missing,  # Uub 
    missing,  # Uut 
    missing,  # Uuq 
    missing,  # Uup 
    missing,  # Uuh 
    missing,  # Uuh
    ])

numbers = {}
pt = {}
for num, sym in enumerate(symbols):
    pt[sym] = {'number': num, 
               'mass': masses[num], 
               'cov_rad': covalent_radii[num]}
    numbers[sym] = num
del num, sym
