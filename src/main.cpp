#include <iostream>
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

    net.train(trainingData, 1, 10, testData);

    system("pause");
}