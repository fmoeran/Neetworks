//
// Created by Felix Moeran on 17/12/2023.
//

#pragma once
#include "activation.hpp"
#include "containers/tensor.hpp"


namespace nw
{
    /// References a tensor and its members' gradients (derivatives)
    struct GradientIterator;


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
        virtual FlatIterator backPropagate(FlatIterator outputDerivatives) {return FlatIterator();}

        /// Resets all derivative values to 0, allowing backpropagation to begin altering them again.
        /// This is used in Network::train.
        virtual void resetGradients() {};

        /// Retrieves a vector of gradient iterators
        /// storing the gradients of each parameter in the layer. Used by optimizers in compilation and training.
        virtual std::vector<GradientIterator> getParameterGradients() {return {};};

        /// Returns the size of the output vector of the layer
        [[nodiscard]] virtual size_t size() const { return 0; };

    protected:
        /// non-linear activation function
        __Activation *_activation;
        /// previous layer in the network (set to nullptr with an input layer)
        __Layer *_previous;
    };

    struct GradientIterator {

        GradientIterator() {parameters = FlatIterator(); gradients = FlatIterator();}

        GradientIterator(FlatIterator parameterIterator, FlatIterator gradientIterator) {
            if (parameterIterator.size() != gradientIterator.size()) {
                throw std::invalid_argument("Parameter tensor must have same size as gradient tensor.");
            }

            parameters = parameterIterator;
            gradients  = gradientIterator;
        }


        template<size_t RANK>
        GradientIterator(Tensor<RANK>& parameterTensor, Tensor<RANK>& gradientTensor){
            if (parameterTensor.size() != gradientTensor.size()) {
                throw std::invalid_argument("Parameter tensor must have same size as gradient tensor.");
            }

            parameters = parameterTensor.getFlatIterator();
            gradients  = gradientTensor.getFlatIterator();

        }

        size_t size() {return parameters.size();}

        FlatIterator parameters;
        FlatIterator gradients;
    };

}

