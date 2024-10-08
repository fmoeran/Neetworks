//
// Created by felix on 11/07/2024.
//

#include "inputLayer.hpp"

namespace nw
{
    InputLayer::InputLayer(size_t size) : _size(size), _values({size}) {}

    void InputLayer::propagate() {}

    FlatIterator InputLayer::getOutputs() {
        return _values.getFlatIterator();
    }

    size_t InputLayer::size() const {
        return _size;
    }

    void InputLayer::loadInputs(FlatIterator iter) {
        _values.assign(iter);
    }
}
