//
// Created by Felix Moeran on 17/12/2023.
//

#include "layer.hpp"

#include <random>

namespace nw
{
    InputLayer::InputLayer(size_t size) : _size(size), _values({size}) {}

    void InputLayer::propagate() {}

    const FlatIterator InputLayer::getOutputs() {
        return _values.getFlatIterator();
    }

    size_t InputLayer::size() const {
        return _size;
    }


    DenseLayer::DenseLayer(size_t size, __Layer *prev, __Activation *activation)
            : _size(size), _biases({size}), _values({size}),
            _activatedValues({size}), _weights({size, prev->size()}) {
        _previous = prev;
        _activation = activation;

        std::random_device rd{};
        std::default_random_engine generator{rd()};
        std::normal_distribution<float> distribution;

        for (size_t i = 0; i < this->size(); i++) {
            _biases.get({i}) = distribution(generator);
            for (size_t j = 0; j < _previous->size(); j++) {
                _weights.get({i, j}) = distribution(generator);
            }
        }

    }

    void DenseLayer::propagate() {

    }

    size_t DenseLayer::size() const {
        return _size;
    }

    const FlatIterator DenseLayer::getOutputs() {
        return _activatedValues.getFlatIterator();
    }
}



