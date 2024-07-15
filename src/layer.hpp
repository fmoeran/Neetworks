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

        /// Runs the backpropagation algorithm on this layer. Updating the layer's internal derivatives.\n
        /// Also retrieves a Tensor iterator of the derivatives of the previous layer's outputs.\n
        /// (dCost)/(dPrevOut) for every prevOut
        /// \param outputDerivatives The derivative of cost with respect to the outputs of this layer in the last pass.
        virtual FlatIterator backPropagate(FlatIterator outputDerivatives) {return FlatIterator(); }

        /// Resets all derivative values to 0, allowing backpropagation to begin altering them again.
        /// This is used in Network::train.
        virtual void resetDerivatives() {};


        virtual void update(size_t N, float rate) {};

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

