//
// Created by Felix Moeran on 17/12/2023.
//

#pragma once
#include "activation.hpp"
#include "containers/tensor.hpp"

#include <memory>


namespace nw
{
    /// References a tensor and its members' gradients (derivatives)
    struct GradientIterator;


    struct __Layer {
    public:
        __Layer() {};

        /// Returns references to every parameter in the network. e.g. weights and biases.
        virtual std::vector<FlatIterator> getParameters() {return {};};

        /// Returns (dCost)/(dParam) for every paramater in the layer. \n
        /// These should be in the EXACT SAME ORDER as the parameters in getParameters().\n
        /// These are the gradients calculated in backPropagate().
        virtual std::vector<FlatIterator> getParameterGradients() {return {};};

        /// Uses the layer's weights, biases, etc. to return the output to be used in the next layer of the network.
        /// \param previousOutput the outputs from the previous layer's propagate() call.
        virtual FlatIterator propagate(FlatIterator previousOutput) {return FlatIterator();};

        /// Runs the backpropagation algorithm on this layer. Updating the layer's internal derivatives.\n
        /// This finds the derivatives of the previous layer's outputs values. \n
        /// This ALSO finds the derivatives of all of the parameters in this layer.
        /// These derivatives should be available until getParamaterGradients() is called.
        /// \param outputDerivatives The derivative of cost with respect to the outputs of this layer in the last pass:
        /// (dCost)/(dPrevOut) for every prevOut
        virtual FlatIterator backPropagate(FlatIterator outputDerivatives) {return FlatIterator();}

        /// Creates a new, DEEP COPIED, layer of the same class, stored within a unique_ptr.
        virtual std::unique_ptr<__Layer> copyToUnique() {return std::make_unique<__Layer>();};

        /// Returns the size of the output tensor of the layer
        [[nodiscard]] virtual size_t size() const { return 0; };

        /// Resets all derivative values to 0 to ensure that the next backPropagate() call creates new gradients.
        void resetParameterGradients() {
            for (FlatIterator paramIter : getParameters())
                std::fill(paramIter.begin(), paramIter.end(), 0.0f);
        };
    };



}

