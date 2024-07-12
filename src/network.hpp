//
// Created by Felix Moeran on 19/12/2023.
//

#pragma once

#include "layer.hpp"
#include "layers/inputLayer.hpp"
#include <vector>

namespace nw
{

    struct Network {
    public:
        Network(size_t inputSize);

        /// Adds a layer to the NN
        /// @param layer pointer to an object inheriting from the __Layer class
        void addLayer(__Layer *layer);

        /// Runs the NN, updating the values stored in each layer
        void feedForward(FlatIterator inputIterator);

        /// Pointer to the final layer in the network
        /// Often used to assign a new layer after this one
        __Layer* lastLayer();

        /// Returns the output tensor from the final layer in the network
        FlatIterator getOutput();

        /// Returns the input tensor most recently used by the network
        FlatIterator getInput();

    private:
        std::vector<__Layer *> _layers;
        InputLayer *_inputLayerPtr;

    };
}


