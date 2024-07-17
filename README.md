# Neetworks

A low level C++ Neural Network library, designed to allow developers to create their own customized networks with ease.

## Example programs

For demonstration purposes, Neetworks comes packaged with the [mnist data set](https://yann.lecun.com/exdb/mnist/) and a number of classic neural network features. Below is a 784x30x10 MLP using sigmoid activation functions, 
a mean squared error cost function, and stochastic gradient descent with a learning rate of 3 over 5 epochs with a mini-batch size of 10.

```c++
#include "containers/tensor.hpp"
#include "layers/denseLayer.hpp"
#include "activations/sigmoid.hpp"
#include "costs/meanSquaredError.hpp"
#include "optimizers/sgd.hpp"
#include "network.hpp"
#include "dataHandler.hpp"

int main() {
    using namespace nw;

    Network net(784);
    Sigmoid sig;
    MSE mse;
    SGD sgd(3);

    DenseLayer d1(30, &net.inputLayer(), &sig);
    DenseLayer d2(10, &d1, &sig);

    net.addLayer(&d1);
    net.addLayer(&d2);

    net.compile(&mse, &sgd);

    Data trainingData = getMnistTrainingData();
    Data testData     = getMnistTestData();

    net.train(trainingData, 5, 10, testData);

    system("pause");
}
```

Resulting in:

```
Epoch 1: Training = 49360/60000 Testing = 9130/10000 Cost = 0.144993
Epoch 2: Training = 55173/60000 Testing = 9291/10000 Cost = 0.116057
Epoch 3: Training = 55957/60000 Testing = 9371/10000 Cost = 0.106447
Epoch 4: Training = 56382/60000 Testing = 9376/10000 Cost = 0.104076
Epoch 5: Training = 56577/60000 Testing = 9386/10000 Cost = 0.0996542
```
