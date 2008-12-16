import matplotlib.pyplot as plt

import pydos

pwifn = 'AlN.md.in'
pwofn = 'AlN.md.out.gz'

#--- minimal pydos.main() -----------------------------------------------------

pwin_nl = pydos.conf_namelists(pwifn)
atspec = pydos.atomic_species(pwifn)
atpos_in = pydos.atomic_positions(pwifn, atspec)
pwout = pydos.parse_pwout(fn_out=pwofn,
                    pwin_nl=pwin_nl, 
                    atspec=atspec,
                    atpos_in=atpos_in)

massvec = atpos_in['massvec']
R = pwout['R']

V = pydos.velocity(R, copy=False)
dt = pydos._float(pwin_nl['control']['dt']) * pydos.constants.tryd

#--- compare PDOS methods -----------------------------------------------------

# [mass] >    False   True
# v [method]   
# vacf        vd      vdm
# direct      dd      ddm

M = None
faxis_vd, vd = pydos.vacf_pdos(V, dt=dt, m=M, mirr=True)
faxis_dd, dd = pydos.direct_pdos(V, dt=dt, m=M)

M = massvec
faxis_vd, vdm = pydos.vacf_pdos(V, dt=dt, m=M, mirr=True)
faxis_dd, ddm = pydos.direct_pdos(V, dt=dt, m=M)

fig=[]
ax=[]

fig.append(plt.figure())
ax.append(fig[-1].add_subplot(111))
ax[-1].plot(faxis_vd, vd, label='vd')
ax[-1].plot(faxis_vd, vdm, label='vdm')
ax[-1].plot(faxis_dd, dd, label='dd')
ax[-1].plot(faxis_dd, ddm, label='ddm')
ax[-1].legend()

# Plot the ratio [w/o mass] / [with mass] for each method.
fig.append(plt.figure())
ax.append(fig[-1].add_subplot(111))
ax[-1].plot(faxis_vd, vd/vdm, label='vd/vdm')
ax[-1].plot(faxis_dd, dd/ddm, label='dd/ddm')
ax[-1].legend()


plt.show()
