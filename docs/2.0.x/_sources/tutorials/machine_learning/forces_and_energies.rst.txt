Supervised Learning: Training an ML Force-field
===============================================

This tutorial covers how descriptors can be effectively used as input for a
machine learning model that will predict energies and forces. There are several
design choices that you have to make when building a ML force-field: which ML
model, which descriptor, etc. In this tutorial we will use the following, very
simple setup:

    - Dataset of two atoms interacting through a Lennard-Jones potential. This
      is pretty much as simple as it gets. Real systems will be much more
      complicated thus requiring a more complicated machine learning model and
      longer training times.
    - SOAP descriptor calculated directly between the two atoms. Once again
      this is for simplicity and in real systems you would have many more
      centers, possibly on top of each atom.
    - We will use a fully connected neural network to perform the prediction.
      In principle any machine learning method will do, but neural networks can
      very conveniently calculate the analytical derivatives of the output with
      respect to the input. This allows us to train an energy prediction model
      from which we will automatically get the forces as long as we also know
      the derivatives of the descriptor with respect to the atomic positions.
      This is exactly what the :code:`derivatives`-function provided by DScribe
      returns (notice that DScribe does not yet provide these derivatives for
      all descriptors).

Setup
-----
We will use a dataset of feature vectors :math:`\mathbf{D}`, their derivatives
:math:`\nabla_{\mathbf{r_i}} \mathbf{D}` and the associated system energies :math:`E` and
forces :math:`\mathbf{F}` for training. We will use a neural network :math:`f`
to predict the energies: :math:`\hat{E} = f(\mathbf{D})`. Here
variables with a "hat" on top indicate predicted quantities to distinguish them
from the real values. The predicted forces can be directly computed as the
negative gradients with respect to the atomic positions. For example the force
for atom :math:`i` can be computed as (using row vectors):

.. math::
    \hat{\mathbf{F}}_i &= - \nabla_{\mathbf{r_i}} f(\mathbf{D}) \\
                 &= - \nabla_{\mathbf{D}} f \cdot \nabla_{\mathbf{r_i}} \mathbf{D}\\
                 &= - \begin{bmatrix}
                        \frac{\partial f}{\partial D_1} & \frac{\partial f}{\partial D_2} & \dots
                      \end{bmatrix}
                        \begin{bmatrix}
                        \frac{\partial D_1}{\partial x_i} & \frac{\partial D_1}{\partial y_i} & \frac{\partial D_1}{\partial z_i}\\
                        \frac{\partial D_2}{\partial x_i} & \frac{\partial D_2}{\partial y_i} & \frac{\partial D_2}{\partial z_i}\\
                        \vdots & \vdots & \vdots \\
                      \end{bmatrix}

In these equations :math:`\nabla_{\mathbf{D}} f` is the derivative of the ML
model output with respect to the input descriptor. As mentioned before, neural
networks typically can output these derivatives analytically.
:math:`\nabla_{\mathbf{r_i}} \mathbf{D}` is the descriptor derivative with
respect to an atomic position. DScribe provides these derivatives for the SOAP
descriptor. Notice that in the derivatives provided by DScribe last dimension
loops over the features. This makes calculating the involved dot products
faster in an environment that uses a row-major order, such as numpy or C/C++,
as the dot product is taken over the last, fastest dimension. But you can of
course organize the output in any way you like.

The loss function for the neural network will contain the sum of mean squared
error of both energies and forces. In order to better equalize the contribution
of these two properties in the loss function their values are scaled by their
variance in the training set.

Dataset generation
------------------

The following script generates our training dataset (full script in the GitHub
repository: *examples/forces_and_energies/dataset.py*):

.. literalinclude:: ../../../../examples/forces_and_energies/dataset.py
    :language: python

The energies will look like this:

.. image:: /_static/img/lj.png
   :alt: Lennard-Jones energies
   :align: center
   :width: 90%

Training
--------
.. note::
   The training code shown in this tutorial uses `pytorch
   <https://pytorch.org/>`_, which you need to install if you wish to run this
   example. You can find the full script at
   *examples/forces_and_energies/training_pytorch.py*. Notice that there is
   also an identical implementation using `tensorflow
   <https://www.tensorflow.org/>`_, kindly provided by `xScoschx
   <https://github.com/xScoschx>`_. It can be found under
   *examples/forces_and_energies/training_tensorflow.py*.

Let us first load and prepare the dataset (full script in the GitHub
repository: *examples/forces_and_energies/training_pytorch.py*):

.. literalinclude:: ../../../../examples/forces_and_energies/training_pytorch.py
    :language: python
    :lines: 1-57

Then let us define our model and loss function:

.. literalinclude:: ../../../../examples/forces_and_energies/training_pytorch.py
    :start-at: class FFNet
    :language: python
    :lines: 1-33

Now we can define the training loop that uses batches and early stopping to
prevent overfitting:

.. literalinclude:: ../../../../examples/forces_and_energies/training_pytorch.py
    :start-at: # Train!
    :language: python
    :lines: 1-76

Once the model is trained (takes around thirty seconds), we can load the best
model and produce predicted data that is saved on disk for later analysis:

.. literalinclude:: ../../../../examples/forces_and_energies/training_pytorch.py
   :start-at: # Way to tell
   :language: python
   :lines: 1-

Analysis
--------
In order to quickly assess the model performance, we will simply plot the model
response in the whole dataset input domain and compare it to the correct values
(full script in the GitHub repository:
*examples/forces_and_energies/analysis.py*):

.. literalinclude:: ../../../../examples/forces_and_energies/analysis.py
   :language: python

The plots look something like this:

.. image:: /_static/img/nn_test.png
   :alt: Lennard-Jones energies
   :align: center
   :width: 90%

