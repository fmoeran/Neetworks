//
// Created by Felix Moeran on 17/12/2023.
//

#include "layer.hpp"

#include <random>

InputLayer::InputLayer(size_t size): _values(size), _size(size) {}

void InputLayer::propagate() {}

const Vector &InputLayer::getOutputs() {
    return _values;
}

size_t InputLayer::size() const {
    return _size;
}


DenseLayer::DenseLayer(size_t size, __Layer* prev, __Activation* activation)
: _size(size), _values(size), _weights(size, prev->size()), _biases(size), _activatedValues(size){
    _previous = prev;
    _activation = activation;

    std::random_device rd {};
    std::default_random_engine generator {rd()};
    std::normal_distribution<float> distribution;

    for (int i=0; i<this->size(); i++) {
        _biases[i] = distribution(generator);
        for (int j=0; j<_previous->size(); j++) {
            _weights[i][j] = distribution(generator);
        }
    }

}

void DenseLayer::propagate() {

}

size_t DenseLayer::size() const {
    return _size;
}

const Vector &DenseLayer::getOutputs() {
    return _values;
}




