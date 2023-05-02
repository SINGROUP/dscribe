import numpy as np
import matplotlib.pyplot as plt


fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(8,12))

# Plot pow with different settings
t = 1e-2
r_cut = 5
settings = [
    [2, r_cut/(1/t*(1-t)) ** (1 / 2), 1],
    [4, r_cut/(1/t*(1-t)) ** (1 / 4), 1],
    [8, r_cut/(1/t*(1-t)) ** (1 / 8), 1],
]
rmin = 0
rmax = 5.2
for setting in settings:
    m = setting[0]
    r0 = setting[1]
    c = setting[2]
    d = c
    r = np.arange(rmin, rmax, 0.01)
    polym = c / (d + (r / r0) ** m)
    ax2.plot(r, polym, label="m = {}, r0 = {:.3}, c = d = {}".format(m, r0, c))
ax2.axvline(r_cut, color='k', linestyle='--')
ax2.text(
    r_cut*0.99,
    0.5,
    "r_cut inferred from threshold",
    verticalalignment='center',
    horizontalalignment='right',
    rotation="vertical",
)
ax2.set_title("pow")
ax2.set_xlabel("r")
ax2.set_ylabel("w(r)")

# Plot poly with different settings
settings = [
    [r_cut, 3],
    [r_cut, 2],
    [r_cut, 1],
]
for setting in settings:
    r0 = setting[0]
    m = setting[1]
    c = 1
    poly3m = []
    for ri in r:
        if ri < r0:
            poly3m.append(c*(1 + 2 * (ri / r0) ** 3 - 3 * (ri / r0) ** 2) ** m)
        else:
            poly3m.append(0)
    ax1.plot(r, poly3m, label="m = {}, r0 = {}, c={}".format(m, r0, c))
ax1.axvline(r_cut, color='k', linestyle='--')
ax1.text(
    r_cut*0.99,
    0.5,
    "r_cut inferred from r0".format(t),
    verticalalignment='center',
    horizontalalignment='right',
    rotation="vertical",
)
ax1.set_title("poly")
ax1.set_xlabel("r")
ax1.set_ylabel("w(r)")

# Plot exp with different settings
settings = [
    [r_cut/np.log(1/t - 0), 1, 0],
    [r_cut/np.log(10/t - 9), 10, 9],
    [r_cut/np.log(100/t - 99), 100, 99],
]
for setting in settings:
    r = np.arange(rmin, rmax, 0.01)
    r0 = setting[0]
    c = setting[1]
    d = setting[2]
    exp = c/(d + np.exp(r/r0))
    ax3.plot(r, exp, label="r0={:.3}, c={}, d={}".format(r0, c, d))
ax3.axvline(r_cut, color='k', linestyle='--')
ax3.text(
    r_cut*0.99,
    0.5,
    "r_cut inferred from threshold",
    verticalalignment='center',
    horizontalalignment='right',
    rotation="vertical",
)
ax3.set_title("exp")
ax3.set_xlabel("r")
ax3.set_ylabel("w(r)")

l = "upper right"
anchor = (0.9, 1)
ax1.set_xlim(rmin, rmax)
ax1.set_ylim(0, 1)
ax2.set_ylim(0, 1)
ax3.set_ylim(0, 1)
ax1.legend(loc=l, bbox_to_anchor=anchor)
ax2.legend(loc=l, bbox_to_anchor=anchor)
ax3.legend(loc=l, bbox_to_anchor=anchor)
plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.05)
plt.show()
