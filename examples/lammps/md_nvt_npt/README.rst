Run a short LAMMPS MD (rock salt AlN solid state system) either NPT or NVT and
plot various quantities parsed from txt and dcd files or calculated from parsed
data.

This is used to show subtle differences in various output quantities in NPT
runs which should in theory be exactly the same.

Usage
-----
./run.py nvt | npt

We plot a quantity obtained by 3 different ways:

txt  = parsed from lmp.out.dump, i.e. the printed value directly from lammps
calc = set tr.<quantity>=None and calculate it by the methods defined in
       Trajectory (i.e. by calling set_all() -> call get_<quantity>())
dcd  = parse dcd file with only coords and cryst_const, rest is calculated in
       Trajectory

Here is a detailed overview of what is parsed and what is calculated from
parsed data.

txt
---
parsed: (from lmp.out.dump)
  coords      (xu  yu  zu)
  coords_frac (xsu ysu zsu)
  cell        (ITEM BOX BOUNDS)
  velocity    (vx vy vz)
calculated:
  cryst_const   from cell

calc (based on txt values above)
----
parsed:
  --
calculated:
  coords        from coords_frac+cell
  coords_frac   from coords+cell
  cryst_const   from cell
  cell          from cryst_const
  velocity      from coords

dcd
---
parsed: (from lmp.out.dcd)
  coords
  cryst_const
calculated:
  cell          from cryst_const
  coords_frac   from coords+cell
  velocity      from coords

Results
-------
Everything is the same for NVT.

For NPT we find substantial(+) and small(-) differences for some quantities:

    cell        : txt  = dcd   = calc
    cryst_const : txt  = dcd   = calc
    coords      : txt  = dcd  != calc     +
    coords_frac : txt != dcd   = calc     +
    velocity    : txt != dcd   = calc     -

=> coords(txt)      = coords(dcd)
=> cell(txt)        = cell(dcd)
=> cryst_const(txt) = cryst_const(dcd)
BUT:
=> coords_frac(txt), coords(txt) and cell(txt) don't fit together! Who is
   right? Should we rely on coords(txt) or coords_frac(txt)??

Pragmatic choice: Use only dcd, which is what we need to use anyway for real
world MD runs. Then we have no choice other than to believe that coords(txt)
is The Truth. We should therefore ignore and never use coords_frac(txt).
