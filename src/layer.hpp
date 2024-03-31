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
        virtual const FlatIterator getOutputs() { return FlatIterator(); };

        // Returns the size of the output vector of the layer
        [[nodiscard]] virtual size_t size() const { return 0; };

    protected:
        /// non-linear activation function
        __Activation *_activation;
        /// previous layer in the network (set to nullptr with an input layer)
        __Layer *_previous;
    };

// A layer that will always be at the start of any network
    struct InputLayer : public __Layer {
    public:
        explicit InputLayer(size_t size);

        void propagate() override;

        template<typename InputIter>
        void loadInputs(InputIter begin, InputIter end);

        const FlatIterator getOutputs() override;

        [[nodiscard]] size_t size() const override;

    private:
        size_t _size;
        Tensor<1> _values;
    };


// The basic NN layer, edges between every node and every previous layer node, with weights biases and an activation function
    struct DenseLayer : public __Layer {
    public:
        explicit DenseLayer(size_t size, __Layer *prev, __Activation *activation);

        void propagate() override;

        const FlatIterator getOutputs() override;

        [[nodiscard]] size_t size() const override;

    private:
        size_t _size;
        Tensor<1> _biases, _values, _activatedValues;
        Tensor<2> _weights;
    };
}

