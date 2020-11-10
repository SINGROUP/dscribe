import numpy as np
import torch
from matplotlib import pyplot as mpl
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
torch.manual_seed(7)

# Load the dataset
D_numpy = np.load("D.npy")
n_samples, n_features = D_numpy.shape
E_numpy = np.array([np.load("E.npy")]).T
F_numpy = np.load("F.npy")
dD_dr_numpy = np.load("dD_dr.npy")
r_numpy = np.load("r.npy")

# Split data into training and test sets
D_train, D_test, E_train, E_test, F_train, F_test, dD_dr_train, dD_dr_test, r_train, r_test = train_test_split(
    D_numpy,
    E_numpy,
    F_numpy,
    dD_dr_numpy,
    r_numpy,
    test_size=150,
    train_size=50,
    random_state=7,
)

# Standardize input for improved learning. Fit is done only on training data,
# scaling is applied to both descriptors and their derivatives on training and
# test sets.
scaler = StandardScaler().fit(D_train)
D_train = scaler.transform(D_train)
D_whole = scaler.transform(D_numpy)
dD_dr_train = dD_dr_train / scaler.scale_[None, None, None, :]

# Split off 10% validation data from the training data for early stopping
D_train, D_valid, E_train, E_valid, F_train, F_valid, dD_dr_train, dD_dr_valid, r_train, r_valid = train_test_split(
    D_train,
    E_train,
    F_train,
    dD_dr_train,
    r_train,
    test_size=0.2,
    random_state=42,
)

# Create tensors for pytorch
D_whole = torch.Tensor(D_whole)
D_train = torch.Tensor(D_train)
D_valid = torch.Tensor(D_valid)
E_whole = torch.Tensor(E_numpy)
E_train = torch.Tensor(E_train)
E_valid = torch.Tensor(E_valid)
F_whole = torch.Tensor(F_numpy)
F_train = torch.Tensor(F_train)
F_valid = torch.Tensor(F_valid)
r_whole = r_numpy
dD_dr_whole = torch.Tensor(dD_dr_numpy / scaler.scale_[None, None, None, :])
dD_dr_train = torch.Tensor(dD_dr_train)
dD_dr_valid = torch.Tensor(dD_dr_valid)

# Calculate the variance of energy and force values for the training set. These
# are used to balance their contribution to the MSE loss
var_energy_train = torch.var(E_train)
var_force_train = torch.var(F_train)

# We explicityly require that the gradients should be calculated for the input
# variables. Pytorch will not do this by default as it is typically not needed.
D_whole.requires_grad = True
D_valid.requires_grad = True


class FFNet(torch.nn.Module):
    """A simple feed-forward network with one hidden layer, randomly
    initialized weights, ReLU activation and a linear output layer.
    """
    def __init__(self, n_features, n_hidden, n_out):
        super(FFNet, self).__init__()
        self.linear1 = torch.nn.Linear(n_features, n_hidden)
        torch.nn.init.normal_(self.linear1.weight, mean=0, std=1.0)
        self.relu1 = torch.nn.Sigmoid()
        self.linear3 = torch.nn.Linear(n_hidden, n_out)
        torch.nn.init.normal_(self.linear3.weight, mean=0, std=1.0)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear3(x)

        return x


def energy_force_loss(E_pred, E_train, F_pred, F_train, with_forces=True):
    """Custom loss function that targets both energies and forces.
    """
    energy_loss = torch.mean((E_pred - E_train)**2) / var_energy_train
    if with_forces:
        force_loss = torch.mean((F_pred - F_train)**2) / var_force_train
        return energy_loss + force_loss
    return energy_loss

# Initialize model
model = FFNet(n_features, n_hidden=5, n_out=1)

# The Adam optimizer is used for training the model parameters
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

# Train!
n_epochs = 1000
batch_size = 16
patience = 100
i_worse = 0
old_valid_loss = float("Inf")
best_valid_loss = float("Inf")

# Epochs
for i_epoch in range(n_epochs):

    # Batches
    permutation = torch.randperm(D_train.size()[0])
    for i in range(0, D_train.size()[0], batch_size):

        indices = permutation[i:i+batch_size]
        D_train_batch, E_train_batch = D_train[indices], E_train[indices]
        D_train_batch.requires_grad = True
        F_train_batch, dD_dr_train_batch = F_train[indices], dD_dr_train[indices]

        # Forward pass: Predict energies from the descriptor input
        E_train_pred_batch = model(D_train_batch)

        # Get derivatives of model output with respect to input variables. The
        # backward()-function defined by pytorch does exactly this. As the output
        # is non-scalar, we need to give the gradients explicitly. Also as we will
        # call the backward function again to calculate the gradients with respect
        # to the loss, we need to use retain_graph=True.
        E_train_pred_batch.backward(gradient=torch.ones(E_train_pred_batch.size()), retain_graph=True)
        df_dD_train_batch = D_train_batch.grad

        # Get derivatives of input variables (=descriptor) with respect to atom
        # positions = forces
        F_train_pred_batch = -torch.einsum('ijkl,il->ijk', dD_dr_train_batch, df_dD_train_batch)

        # Zero gradients, perform a backward pass, and update the weights.
        D_train_batch.grad.data.zero_()
        optimizer.zero_grad()
        loss = energy_force_loss(E_train_pred_batch, E_train_batch, F_train_pred_batch, F_train_batch, with_forces=True)
        loss.backward()
        optimizer.step()
            
    # Check early stopping criterion and save best model
    E_valid_pred = model(D_valid)
    E_valid_pred.backward(gradient=torch.ones(E_valid_pred.size()), retain_graph=True)
    df_dD_valid = D_valid.grad
    F_valid_pred = -torch.einsum('ijkl,il->ijk', dD_dr_valid, df_dD_valid)
    valid_loss = energy_force_loss(E_valid_pred, E_valid, F_valid_pred, F_valid, with_forces=True)
    if valid_loss < best_valid_loss:
        print("Saving at epoch {}".format(i_epoch))
        torch.save(model.state_dict(), "best_model.pt")
        best_valid_loss = valid_loss
    if valid_loss >= old_valid_loss:
        i_worse += 1
    else:
        i_worse = 0
    if i_worse > patience:
        print("Early stopping at epoch {}".format(i_epoch))
        break
    old_valid_loss = valid_loss

    if i_epoch % 500 == 0:
        print("  Finished epoch: {} with loss: {}".format(i_epoch, loss.item()))
    D_valid.grad.data.zero_()

# Way to tell pytorch that we are entering the evaluation phase
model.load_state_dict(torch.load("best_model.pt"))
model.eval()

# Calculate energies and force for the entire range
E_whole_pred = model(D_whole)
E_whole_pred.backward(gradient=torch.ones(E_whole_pred.size()))
df_dD_whole = D_whole.grad
F_whole_pred = -torch.einsum('ijkl,il->ijk', dD_dr_whole, df_dD_whole)

# Plot energies for the whole range
order = np.argsort(r_whole)
fig, (ax1, ax2) = mpl.subplots(2, 1, sharex=True, figsize=(10, 10))
ax1.plot(r_whole[order], E_whole[order], label="True", linewidth=3, linestyle="-")
ax1.plot(r_whole[order], E_whole_pred.detach().numpy()[order], label="Predicted", linewidth=3, linestyle="-")
ax1.set_ylabel('Energy', size=15)
ax1.legend(fontsize=12)

# Plot forces for whole range
F_x_whole_pred = F_whole_pred.detach().numpy()[order, 0, 0]
F_x_whole = F_whole[:, 0, 0][order]
ax2.plot(r_whole[order], F_x_whole, label="True", linewidth=3, linestyle="-")
ax2.plot(r_whole[order], F_x_whole_pred, label="Predicted", linewidth=3, linestyle="-")
ax2.legend(fontsize=12)
ax2.set_xlabel('Distance', size=15)
ax2.set_ylabel('Forces', size=15)

# Plot training points
ax1.scatter(r_train, E_train, marker="o", color="k", s=20, label="Training points", zorder=3)
F_x_train = F_train[:, 0, 0]
ax2.scatter(r_train, F_x_train, marker="o", color="k", s=20, label="Training points", zorder=3)

# Show plot
mpl.show()
