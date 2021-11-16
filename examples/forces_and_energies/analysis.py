import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error

# Load data as produded by the trained model
r_whole = np.load("r.npy")
r_train_full = np.load("r_train_full.npy")
order = np.argsort(r_whole)
E_whole = np.load("E.npy")
E_train_full = np.load("E_train_full.npy")
E_whole_pred = np.load("E_whole_pred.npy")
F_whole = np.load("F.npy")
F_train_full = np.load("F_train_full.npy")
F_whole_pred = np.load("F_whole_pred.npy")
F_x_whole_pred = F_whole_pred[order, 0, 0]
F_x_whole = F_whole[:, 0, 0][order]
F_x_train_full = F_train_full[:, 0, 0]

# Plot energies for the whole range
order = np.argsort(r_whole)
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 10))
ax1.plot(r_whole[order], E_whole[order], label="True", linewidth=3, linestyle="-")
ax1.plot(r_whole[order], E_whole_pred[order], label="Predicted", linewidth=3, linestyle="-")
ax1.set_ylabel('Energy', size=15)
mae_energy = mean_absolute_error(E_whole_pred, E_whole)
ax1.text(0.95, 0.5, "MAE: {:.2} eV".format(mae_energy), size=16, horizontalalignment='right', verticalalignment='center', transform=ax1.transAxes)

# Plot forces for whole range
ax2.plot(r_whole[order], F_x_whole, label="True", linewidth=3, linestyle="-")
ax2.plot(r_whole[order], F_x_whole_pred, label="Predicted", linewidth=3, linestyle="-")
ax2.set_xlabel('Distance', size=15)
ax2.set_ylabel('Forces', size=15)
mae_force = mean_absolute_error(F_x_whole_pred, F_x_whole)
ax2.text(0.95, 0.5, "MAE: {:.2} eV/Å".format(mae_force), size=16, horizontalalignment='right', verticalalignment='center', transform=ax2.transAxes)

# Plot training points
ax1.scatter(r_train_full, E_train_full, marker="o", color="k", s=20, label="Training points", zorder=3)
ax2.scatter(r_train_full, F_x_train_full, marker="o", color="k", s=20, label="Training points", zorder=3)

# Show plot
ax1.legend(fontsize=12)
plt.subplots_adjust(left=0.08, right=0.97, top=0.97, bottom=0.08, hspace=0)
plt.show()
