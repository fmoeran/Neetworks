//
// Created by Felix Moeran on 19/12/2023.
//

#include "network.hpp"
#include "layers/inputLayer.hpp"


namespace nw
{
    Network::Network(size_t inputSize) {
        _layers.push_back(new InputLayer(inputSize));

    }

    void Network::addLayer(__Layer *layer) {
        _layers.push_back(layer);
    }
}