# DeepLearing - project 2

## Organisation of the code

#### `test.py`
This is the test script we are asked to provide. Runned without parameters, it execute our model and then a equivalent one implemented with PyTorch. The learning curves are stored in the same folder.

It take optionnals parameters: model nb_samples batch_size nb_units, where:

- model= sn for SimpleNet or pt for PyTorch
- nb_samples= number of samples in the training dataset
- batch_size= length of the training batches
- nb_units= number of units for the first hidden layer

#### `dataset_gen.py`
Contains the function used to generate the toy dataset described in the project specification taken from the correction of professor Fleuret (lab session 3).

### Folder `./simplenet/`
This package contains the implementation of our SimpleNet framework.

#### `simplenet/modules.py`
Contains the implementation of the different layers we were asked to implement and also to Parameter class used to store the parameters of layers alongside their gradients.

#### `simplenet/criterions.py`
Contains the implementation of the loss functions. Currently we only implemented the MSELoss as asked.

#### `simplenet/optimizers.py`
Contains the implementation of the SGD optimizer.

#### `simplenet/training.py`
Contains the ModelTrainer class used for training our models (compatible with SimpleNet and PyTorch models). Takes care of the training loop (batching, reseting the gradients, call optimizer step(), ...) and plot the learning curves.

### Folder `./tests/`
Contains the unit tests that we implemented to check that our framework return correct results. We've done that by comparing our outputs with the one from PyTorch functions.
