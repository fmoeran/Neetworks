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
        explicit DenseLayer(size_t size, __Layer *prev, __Activation *activation);

        void propagate() override;

        FlatIterator getOutputs() override;

        FlatIterator backPropagate(FlatIterator outputDerivatives) override;

        [[nodiscard]] size_t size() const override;

        void resetGradients() override;

        std::vector<GradientIterator> getParameterGradients() override;



    private:
        size_t _size;
        Tensor<1> _biases, _values, _activatedValues;
        Tensor<2> _weights;
        Tensor<1> _biaseDerivatives;
        Tensor<2> _weightDerivatives;
        // The derivative of the cost w.r.t the previous layer's output
        Tensor<1> _prevLayerDerivatives;

    };
}