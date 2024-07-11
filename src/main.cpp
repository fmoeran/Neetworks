#include <iostream>
#include <vector>
#include "containers/tensor.hpp"
#include "layers/denseLayer.hpp"
#include "layers/inputLayer.hpp"
#include "activations/sigmoid.hpp"

int main() {
    using namespace nw;
    std::cout << "h" << std::endl;

    InputLayer il(5);

    DenseLayer dl(5, &il, new Sigmoid);

    return 0;
}