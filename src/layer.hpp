//
// Created by Felix Moeran on 17/12/2023.
//

#pragma once
#include "activation.hpp"
#include "containers/tensor.hpp"


namespace nw
{

    struct __Layer {
    public:
        /// Uses the layer's weights, biases, etc. to update the layer's values (found in getOutputs)
        virtual void propagate() {};

        /// Retrieves a Tensor iterator of values that the layer has produced AFTER a propagate() call
        virtual FlatIterator getOutputs() { return FlatIterator(); };

        // Returns the size of the output vector of the layer
        [[nodiscard]] virtual size_t size() const { return 0; };

    protected:
        /// non-linear activation function
        __Activation *_activation;
        /// previous layer in the network (set to nullptr with an input layer)
        __Layer *_previous;
    };



;
}

