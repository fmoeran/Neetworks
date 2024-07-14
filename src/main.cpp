#include <iostream>
#include <vector>
#include "containers/tensor.hpp"
#include "layers/denseLayer.hpp"
#include "activations/sigmoid.hpp"
#include "costs/meanSquaredError.hpp"
#include "network.hpp"
#include "dataHandler.hpp"

int main() {
    using namespace nw;

//    Network net(784);
//    Sigmoid sig;
//    MSE mse;
//
//    DenseLayer d1(16, &net.inputLayer(), &sig);
//    DenseLayer d2(16, &d1, &sig);
//    DenseLayer d3(10, &d2, &sig);
//    net.addLayer(&d1);
//    net.addLayer(&d2);
//    net.addLayer(&d3);
//
//    net.compile(&mse);
//
//    Data trainingData = getMnistTestData();
//
//
//    net.train(trainingData, 5, 1000);


    Tensor<2> mat({3, 2});
    mat.assign(FlatIterator({1, 0, 2, 1, 0, 2}));
    Tensor<1> vec1({3});
    vec1.assign(FlatIterator({2, 3, 4}));
    Tensor<1> vec2({3});
    Tensor<1> vec3({3});
    vec3.assign(FlatIterator({3, 4, 5}));

    Tensor<2> mat2({3, 3});

    operators::vecTensorProduct(vec1.getFlatIterator(), vec3.getFlatIterator(), mat2);

    std::cout << mat2 << std::endl;

    system("pause");
}