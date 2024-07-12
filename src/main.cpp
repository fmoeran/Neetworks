#include <iostream>
#include <vector>
#include "containers/tensor.hpp"
#include "layers/denseLayer.hpp"
#include "activations/sigmoid.hpp"
#include "network.hpp"

int main() {
    using namespace nw;
    std::cout << "h" << std::endl;

    Network net(5);
    Sigmoid sig;
    DenseLayer dl(5, net.lastLayer(), &sig);
//    net.addLayer(&dl);

    std::cout << "i" << std::endl;
    return 0;
}