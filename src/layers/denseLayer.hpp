//
// Created by felix on 11/07/2024.
//

#pragma once
#include "../layer.hpp"
namespace nw
{
/// The basic MLP layer, edges between every node and every previous layer node, with weights, biases and an activation function
    struct DenseLayer : public __Layer {
    public:
        explicit DenseLayer(size_t size, __Layer &prev, __Activation &activation);

        std::vector<FlatIterator> getParameters() override;

        std::vector<FlatIterator> getParameterGradients() override;

        FlatIterator propagate(FlatIterator previousOutput) override;

        FlatIterator backPropagate(FlatIterator outputDerivatives) override;


        [[nodiscard]] size_t size() const override;

    private:
        /// Non-linear activation function
        __Activation &_activation;
        /// Number of output values
        size_t _size, _previousSize;
        /// Parameters
        Tensor<1> _biases, _values, _activatedValues;
        Tensor<2> _weights;
        Tensor<1> _biaseDerivatives;
        Tensor<2> _weightDerivatives;
        /// The derivative of the cost w.r.t the previous layer's output
        Tensor<1> _prevLayerDerivatives;
        /// The values most recently given to propagate()
        Tensor<1> _prevLayerOutputs;
    };
}