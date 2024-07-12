//
// Created by Felix Moeran on 19/12/2023.
//

#include "network.hpp"
#include <iostream>

namespace nw
{
    Network::Network(size_t inputSize) {
        _inputLayerPtr = new InputLayer(inputSize);
        _layers.push_back(_inputLayerPtr);
    }

    void Network::addLayer(__Layer *layer) {
        _layers.push_back(layer);
    }

    void Network::feedForward(FlatIterator inputIterator) {
        _inputLayerPtr->loadInputs(inputIterator.begin(), inputIterator.end());
        for (__Layer* layer : _layers) {
            layer->propagate();
        }
    }

    __Layer *Network::lastLayer() {
        return _layers.back();
    }
    FlatIterator Network::getOutput() {
        return lastLayer()->getOutputs();
    }

    FlatIterator Network::getInput() {
        return _inputLayerPtr->getOutputs();
    }

    InputLayer *Network::inputLayer() {
        return _inputLayerPtr;
    }

    std::ostream &operator<<(std::ostream &os, Network &n) {
        for (auto layer : n._layers) {
            os << layer->getOutputs() << '\n';
        }
        return os;
    }
}