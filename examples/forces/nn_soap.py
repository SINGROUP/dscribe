import numpy as np
import torch
from matplotlib import pyplot as mpl
from sklearn.preprocessing import StandardScaler

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
n_epochs = 1000
n_out = 1
n_atoms = 2
n_hidden = 5
n_features = 4
n_samples = 20

# Load numpy data
D_numpy = np.load("D.npy")
n_samples, n_features = D_numpy.shape
E_numpy = np.array([np.load("E.npy")]).T
F_numpy = np.load("F.npy")
# print(D_numpy.shape)
# print(E_numpy.shape)
# print(F_numpy.shape)

# Test multiplication
dD_dr = np.zeros((n_samples, n_atoms, 3, n_features))
df_dD = np.zeros((n_samples, n_features))
F_pred = np.einsum('ijkl,il->ijk', dD_dr, df_dD)
# print(F_pred.shape)
# print(F_pred)

# Standardize input
# print(x.shape)
scaler = StandardScaler().fit(D_numpy)
D_numpy_scaled = scaler.transform(D_numpy)

# Create Tensors
D = torch.Tensor(D_numpy_scaled)
E = torch.Tensor(E_numpy)
F = torch.Tensor(F_numpy)
dD_dr = torch.ones(n_samples, n_atoms, n_features, 3)
D.requires_grad = True

# # Use the nn package to define our model as a sequence of layers. nn.Sequential
# # is a Module which contains other Modules, and applies them in sequence to
# # produce its output. Each Linear Module computes output from input using a
# # linear function, and holds internal Tensors for its weight and bias.
class ToyNet(torch.nn.Module):
    def __init__(self, n_features, n_hidden, n_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(ToyNet, self).__init__()
        self.linear1 = torch.nn.Linear(n_features, n_hidden)
        torch.nn.init.normal_(self.linear1.weight, mean=0, std=1.0)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(n_hidden, n_out)
        torch.nn.init.normal_(self.linear2.weight, mean=0, std=1.0)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)

        return x

model = ToyNet(n_features, n_hidden, n_out)


def custom_loss(E_pred, E_train, F_pred, F_train, forces=True):
    """Custom loss function that targets both energies and forces.
    """
    energy_loss = torch.mean((E_pred - E_train)**2)
    if forces:
        force_loss = torch.mean((F_pred - F_train)**2)
        return energy_loss + force_loss
    return energy_loss

# The nn package also contains definitions of popular loss functions; in this
# case we will use Mean Squared Error (MSE) as our loss function.
# loss_fn = torch.nn.MSELoss(reduction='mean')

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
# loss_function = torch.nn.MSELoss(reduction='mean')

# Initialize the gradients
E_pred = model(D)
loss = custom_loss(E_pred, E, None, None, forces=False)
optimizer.zero_grad()
loss.backward()
optimizer.step()

# Train!
for i_epoch in range(n_epochs):
    # Forward pass: Compute predicted y by passing x to the model
    E_pred = model(D)

    # Get derivatives of model output with respect to input variables
    # (=descriptor)
    df_dD = D.grad

    # Get derivatives of input variables (=descriptor) with respect to atom
    # positions = forces
    F_pred = torch.einsum('ijkl,il->ijk', dD_dr, df_dD)

    # Compute and print loss
    loss = custom_loss(E_pred, E, F_pred, F)

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i_epoch % 100 == 0:
        print("  Finished epoch: {} with loss: {}".format(i_epoch, loss.item()))

# See how it trained
model.eval()
E_test = model(D)
mpl.plot(E_numpy, label="Training data")
mpl.plot(E_test.detach().numpy(), label="Model prediction")
mpl.legend()
mpl.show()
