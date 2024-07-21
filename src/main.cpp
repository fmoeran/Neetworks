#include "containers/tensor.hpp"
#include "layers/denseLayer.hpp"
#include "activations/sigmoid.hpp"
#include "activations/relu.hpp"
#include "costs/meanSquaredError.hpp"
#include "costs/crossEntropy.hpp"
#include "optimizers/sgd.hpp"
#include "optimizers/momentum.hpp"
#include "network.hpp"
#include "dataHandler.hpp"

#include "containers/tensor.hpp"

#include <iostream>
#include <memory>

class Parent {
public:
    int v1;
    Parent(): v1(0){};
    Parent(int _v1): v1(_v1) {};

    virtual std::unique_ptr<Parent> copy() {return std::make_unique<Parent>(v1);};

    virtual void o() {std::cout << "Parent" << std::endl;}

};

class A : public Parent {
public:
    int v2;

    A(int _v1, int _v2) :v2(_v2), Parent(_v1) {};


    std::unique_ptr<Parent> copy() {return std::unique_ptr<Parent>(new A(v1, v2));}

    void o() override {std::cout << "Child" << std::endl;}

};


int main() {
    using namespace nw;

    Network net(784);
//    net.autoSaveStats("stats.csv", std::ios::app);

    Sigmoid sig;
    CrossEntropy ce;
    SGD sgd(1);
    Momentum mom(0.1, 0.9);


    DenseLayer d1(30, net.inputLayer(), sig);
    DenseLayer d2(10, d1, sig);

    net.addLayer(&d1);
    net.addLayer(&d2);

    net.compile(&ce, &sgd);

    Data trainingData = getMnistTrainingData();
    Data testData     = getMnistTestData();

    net.train(trainingData, 30, 10, testData);



    system("pause");
}