#include <iostream>
#include "containers.hpp"
#include "layer.hpp"
#define SIZE 50000000

using namespace std::chrono;


int main() {
    __Layer* layer = new InputLayer(10);

    std::cout << layer->getOutputs();


    return 0;
}