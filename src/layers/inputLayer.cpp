//
// Created by felix on 11/07/2024.
//

#include "inputLayer.hpp"

namespace nw
{
    InputLayer::InputLayer(size_t size) : _size(size), _values({size}) {}

    size_t InputLayer::size() const {
        return _size;
    }

    std::unique_ptr<__Layer> InputLayer::copyToUnique() {
        return std::make_unique<InputLayer>(size());
    }

    FlatIterator InputLayer::propagate(FlatIterator previousOutput) {
        _values.assign(previousOutput);
        return _values.getFlatIterator();
    }
}
