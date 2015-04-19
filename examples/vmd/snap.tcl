# Basic VMD example script for making snapshots and save them.

proc snap {base} {
    set tga $base.tga
    set png $base.png
    render snapshot $tga "convert $tga $png; rm $tga"
}

source very_nice_bonds.tcl
display resetview 
axes location lowerright
##axes location off
scale to 0.120

# y
# z x
snap axis_c

# z
# xy
rotate y by -90; rotate z by -90
snap axis_a

#  z
# xy
rotate y by -90
snap axis_b

exit
