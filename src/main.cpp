#include <iostream>
#include <vector>
#include "containers/tensor.hpp"
#include "layers/denseLayer.hpp"
#include "activations/sigmoid.hpp"
#include "network.hpp"
#include "dataHandler.hpp"

int main() {
    using namespace nw;

    Network net(784);
    Sigmoid sig;

    DenseLayer d1(16, &net.inputLayer(), &sig);
    DenseLayer d2(16, &d1, &sig);
    DenseLayer d3(10, &d2, &sig);
    net.addLayer(&d1);
    net.addLayer(&d2);
    net.addLayer(&d3);

    Data trainingData = getMnistTestData();

    net.train(trainingData, 1, 1000);

    return 0;
}