//
// Created by Felix Moeran on 19/12/2023.
//

#include "network.hpp"


namespace nw
{
    Network::Network(size_t inputSize) {
        _inputLayerPtr = new InputLayer(inputSize);
        _layers.push_back(_inputLayerPtr);

    }

    void Network::addLayer(__Layer *layer) {
        _layers.push_back(layer);
    }

    void Network::feedForward(FlatIterator iterator) {
        _inputLayerPtr->loadInputs(iterator.begin(), iterator.end());
        for (__Layer* layer : _layers) {
            layer->propagate();
        }
    }
}