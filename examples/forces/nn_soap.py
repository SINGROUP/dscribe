import numpy as np
import torch
from matplotlib import pyplot as mpl
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the dataset
D_numpy = np.load("D.npy")
n_samples, n_features = D_numpy.shape
E_numpy = np.array([np.load("E.npy")]).T
F_numpy = np.load("F.npy")
dD_dr_numpy = np.load("dD_dr.npy")
r_numpy = np.load("r.npy")
indices = np.arange(0, n_samples)

# Split data into training and test sets
D_train, D_test, E_train, E_test, F_train, F_test, dD_dr_train, dD_dr_test, r_train, r_test = train_test_split(
    D_numpy,
    E_numpy,
    F_numpy,
    dD_dr_numpy,
    r_numpy,
    test_size=0.20,
    random_state=42,
)

# Standardize input for improved learning. Fit is done only on training data,
# scaling is applied to both descriptors and their derivatives on training and
# test sets.
scaler = StandardScaler().fit(D_train)
D_train = scaler.transform(D_train)
D_test = scaler.transform(D_test)
dD_dr_train = dD_dr_train / scaler.scale_[None, None, None, :]
dD_dr_test = dD_dr_test / scaler.scale_[None, None, None, :]

# Create tensors for pytorch
D_train = torch.Tensor(D_train)
D_test = torch.Tensor(D_test)
E_train = torch.Tensor(E_train)
E_test = torch.Tensor(E_test)
F_train = torch.Tensor(F_train)
dD_dr_train = torch.Tensor(dD_dr_train)
dD_dr_test = torch.Tensor(dD_dr_test)

# We explicityly require that the gradients should be calculated for the input
# variables. Pytorch will not do this by default as it is typically not needed.
D_train.requires_grad = True
D_test.requires_grad = True


class FFNet(torch.nn.Module):
    """A simple one hidden layer feed-forward network with randomly initialized
    weights, ReLU activation and linear output layer.
    """
    def __init__(self, n_features, n_hidden, n_out):
        super(FFNet, self).__init__()
        self.linear1 = torch.nn.Linear(n_features, n_hidden)
        torch.nn.init.normal_(self.linear1.weight, mean=0, std=1.0)
        self.relu1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(n_hidden, n_out)
        torch.nn.init.normal_(self.linear2.weight, mean=0, std=1.0)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)

        return x


def energy_force_loss(E_pred, E_train, F_pred, F_train, with_forces=True):
    """Custom loss function that targets both energies and forces.
    """
    energy_loss = torch.mean((E_pred - E_train)**2)
    if with_forces:
        force_loss = torch.mean((F_pred - F_train)**2)
        return energy_loss + force_loss
    return energy_loss

# Initialize model
model = FFNet(n_features, n_hidden=5, n_out=1)

# The Adam optimizer is used for training the model parameters
optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

# Train!
n_epochs = 15000
for i_epoch in range(n_epochs):
    # Forward pass: Predict energies from the desciptor input
    E_train_pred = model(D_train)

    # Get derivatives of model output with respect to input variables. The
    # backward()-function defined by pytorch does exactly this. As the output
    # is non-scalar, we need to give the gradients explicitly. Also as we will
    # call the backward function again to calculate the gradients with respect
    # to the loss, we need to use retain_graph=True.
    E_train_pred.backward(gradient=torch.ones(E_train_pred.size()), retain_graph=True)
    df_dD_train = D_train.grad

    # Get derivatives of input variables (=descriptor) with respect to atom
    # positions = forces
    F_train_pred = -torch.einsum('ijkl,il->ijk', dD_dr_train, df_dD_train)

    # Zero gradients, perform a backward pass, and update the weights.
    D_train.grad.data.zero_()
    optimizer.zero_grad()
    loss = energy_force_loss(E_train_pred, E_train, F_train_pred, F_train, with_forces=True)
    loss.backward()
    optimizer.step()

    if i_epoch % 500 == 0:
        print("  Finished epoch: {} with loss: {}".format(i_epoch, loss.item()))

# Way to tell pytorch that we are entering the evaluation phase
model.eval()

# Calculate energies and force for the test set
E_test_pred = model(D_test)
E_test_pred.backward(gradient=torch.ones(E_test_pred.size()))
df_dD_test = D_test.grad
F_test_pred = -torch.einsum('ijkl,il->ijk', dD_dr_test, df_dD_test)

# Plot energies for test set
order = np.argsort(r_test)
fig, (ax1, ax2) = mpl.subplots(2, 1, sharex=True)
ax1.plot(r_test[order], E_test[order], label="True")
ax1.plot(r_test[order], E_test_pred.detach().numpy()[order], label="Predicted")
ax1.set_ylabel('Energy')
ax1.legend()

# Plot forces for test set
F_x_test_pred = F_test_pred.detach().numpy()[order, 0, 0]
F_x_test = F_test[:, 0, 0][order]
ax2.plot(r_test[order], F_x_test, label="True")
ax2.plot(r_test[order], F_x_test_pred, label="Predicted")
ax2.legend()
ax2.set_xlabel('Distance')
ax2.set_ylabel('Forces')

# Show plot
mpl.legend()
mpl.show()
