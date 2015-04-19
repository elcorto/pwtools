# VMD script.
#
# Usage:
#     Load molecule on cmd line (or from the VMD File menu), set the variable
#     $molid (number of the current molecule, =0 for the first one) and source
#     the script on the VMD prompt.
#         $ vmd file1.axsf
#         vmd > set molid 0
#         vmd > source script.tcl
#         # load second file (VMD prompt or File menu), it gets ID 1 after
#         # loading, set $molid to refer to this mol and source script again
#         vmd > mol new file2.axsf type xsf first 0 last -1 step 1 waitfor -1
#         vmd > set molid 1
#         vmd > source script.tcl
#
# This script is a modified VMD log file (User Guide: 3.9 Tracking Script
# Command Versions of the GUI Actions). In the original file, there were lines
# like 
#     mol modstyle 0 0 CPK 1.000000 0.000000 20.000000 6.000000
#     mol representation CPK 1.000000 0.000000 20.000000 6.000000
# We replaced them by   
#    set rep [mol modstyle 0 0 CPK 1.000000 0.000000 20.000000 6.000000]
#    mol representation $rep

if !{[info exists molid]} then {
     set molid 0
}     

set repnum -1
display resetview
animate style Loop
display projection Orthographic
color Display Background white
display depthcue off
axes location lowerright

# Draw unit cell. This is a comand from the PBCtools plugin.
pbc box

# CPK <sphere scale> <bond radius> <sphere resolution> <bond resolution>
#
# Atoms with no bonds (bond radius 0.0). These bonds are calculated once when
# the mol is loaded. If we have PBC wrap-around, some bonds span the whole box,
# so make them invisible. The sphere scale is 1.0. Use 0.4 for atoms with the
# same diameter as the bonds.
incr repnum
mol modcolor $repnum $molid Name
mol modselect $repnum $molid all
set rep [mol modstyle $repnum $molid CPK 1.000000 0.000000 20.000000 6.000000]
mol representation $rep

# Activate dynamic bond calculation in each frame. Set bond radius to 0.1
# (default 0.3). Distance cut-off of 2.6.
incr repnum
mol addrep $molid
mol color Name
mol modselect $repnum $molid all
mol modcolor $repnum $molid Name
set rep [mol modstyle $repnum $molid DynamicBonds 2.60000 0.100000 6.000000]
mol representation $rep
