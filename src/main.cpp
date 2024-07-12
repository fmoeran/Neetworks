#include <iostream>
#include <vector>
#include "containers/tensor.hpp"
#include "layers/denseLayer.hpp"
#include "activations/sigmoid.hpp"
#include "network.hpp"

int main() {
    using namespace nw;

    Network net(5);
    Sigmoid sig;
    DenseLayer d1(5, net.inputLayer(), &sig);
    DenseLayer d2(5, &d1, &sig);
    net.addLayer(&d1);
    net.addLayer(&d2);

    std::vector<float> inputVec = {0, 0, 0, 0, 0};
    Tensor<1> input({5});
    input.assign(inputVec.begin(), inputVec.end());

    net.feedForward(input.getFlatIterator());

    std::cout << net << std::endl;

    return 0;
}