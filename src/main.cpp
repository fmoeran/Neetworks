#include <iostream>
#include "containers/tensor.hpp"
#include "layers/denseLayer.hpp"
#include "activations/sigmoid.hpp"
#include "costs/meanSquaredError.hpp"
#include "optimizers/sgd.hpp"
#include "optimizers/momentum.hpp"
#include "network.hpp"
#include "dataHandler.hpp"

int main() {
    using namespace nw;

    Network net(784);
    Sigmoid sig;
    MSE mse;
    SGD sgd(0.1);
    Momentum mom(0.1, 0.9);

    DenseLayer d1(128, &net.inputLayer(), &sig);
    DenseLayer d2(128, &d1, &sig);
    DenseLayer d3(10, &d2, &sig);
    net.addLayer(&d1);
    net.addLayer(&d2);
    net.addLayer(&d3);

    net.compile(&mse, &mom);

    Data trainingData = getMnistTrainingData();

    net.train(trainingData, 30, 10);



//    Tensor<1> t({5});
//    t.randomizeNormal();
//    std::cout << t << std::endl;
//    FlatIterator it = t.getFlatIterator();
//    std::cout << it << std::endl;

    system("pause");
}