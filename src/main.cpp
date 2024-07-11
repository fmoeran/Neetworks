#include <iostream>
#include <vector>
#include "containers/tensor.hpp"
#include "layers/denseLayer.hpp"
#include "activations/sigmoid.hpp"
#include "network.hpp"

int main() {
    using namespace nw;
    std::cout << "h";


//    DenseLayer dl(5, &il, new Sigmoid);

    Network net(5);

    std::cout << "i" << std::endl;
    return 0;
}