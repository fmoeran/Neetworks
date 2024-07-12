//
// Created by felix on 11/07/2024.
//

#include "denseLayer.hpp"
#include <random>


namespace nw
{
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
                _weights.get({i, j}) = 1.0f;
            }
        }

    }

    void DenseLayer::propagate() {
        operators::vecMatMul(_weights.getFlatIterator(), _previous->getOutputs(), _values.getFlatIterator());
        _values += _biases;
        _activation->apply(_values.getFlatIterator(), _activatedValues.getFlatIterator());
    }

    size_t DenseLayer::size() const {
        return _size;
    }

    FlatIterator DenseLayer::getOutputs() {
        return _activatedValues.getFlatIterator();
    }
}