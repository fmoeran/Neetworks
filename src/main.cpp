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
    net.autoSaveStats("stats.csv", std::ios::app);

    Sigmoid sig;
    MSE mse;
//    SGD sgd(1);
    Momentum mom(0.3, 0.9);

    DenseLayer d1(30, &net.inputLayer(), &sig);
    DenseLayer d2(10, &d1, &sig);

    net.addLayer(&d1);
    net.addLayer(&d2);

    net.compile(&mse, &mom);

    Data trainingData = getMnistTrainingData();
    Data testData     = getMnistTestData();

    net.train(trainingData, 30, 10, testData);

    system("pause");
}