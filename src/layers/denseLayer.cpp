//
// Created by felix on 11/07/2024.
//

#include "denseLayer.hpp"
#include <random>

namespace nw {
    DenseLayer::DenseLayer(size_t size, __Layer *prev, __Activation *activation)
            : _size(size), _biases({size}), _values({size}),
              _activatedValues({size}), _weights({size, prev->size()}),
              _biaseDerivatives({size}), _weightDerivatives({size, prev->size()}), _prevLayerDerivatives({prev->size()})
              {
        _previous = prev;
        _activation = activation;

        _biases.randomizeNormal();
        _weights.randomizeNormal();
    }

    void DenseLayer::propagate() {
        operators::vecMatMul(_weights.getFlatIterator(), _previous->getOutputs(), _values.getFlatIterator());
        _values += _biases;
        _activation->apply(_values.getFlatIterator(), _activatedValues.getFlatIterator());
    }

    size_t DenseLayer::size() const {
        return _size;
    }

    FlatIterator DenseLayer::getOutputs() {
        return _activatedValues.getFlatIterator();
    }

    FlatIterator DenseLayer::backPropagate(FlatIterator outputDerivatives) {
        // Calculate derivative of output w.r.t the parameter of the activation function
        Tensor<1> activationDerivatives({size()});
        _activation->applyDerivative(_values.getFlatIterator(), activationDerivatives.getFlatIterator());

        // Calculate derivative of Cost w.r.t the parameter of the activation function
        // These are also the change to the derivative w.r.t the bias terms
        Tensor<1> internalDerivatives({size()});
        operators::hadamard(outputDerivatives, activationDerivatives.getFlatIterator(),
                            internalDerivatives.getFlatIterator());

        FlatIterator it = internalDerivatives.getFlatIterator();

        // Calculate change to the derivative of Cost w.r.t weights
        Tensor<2> changeWeightDerivatives({size(), _previous->size()});
        operators::vecTensorProduct(internalDerivatives.getFlatIterator(), _previous->getOutputs(),
                                    changeWeightDerivatives);

        // Update parameter derivatives
        _biaseDerivatives += internalDerivatives;
        _weightDerivatives += changeWeightDerivatives;

        // Derivative of Cost w.r.t the previous layer's output.
        // The reason this has to be a member variable is that it needs to stay in scope when this function finishes,
        // Otherwise its iterator would be pointing to free memory.
        operators::vecMatTransposeMul(_weights.getFlatIterator(), internalDerivatives.getFlatIterator(),
                                      _prevLayerDerivatives.getFlatIterator());

        return _prevLayerDerivatives.getFlatIterator();
    }

    void DenseLayer::resetGradients() {
        _weightDerivatives.fill(0);
        _biaseDerivatives.fill(0);
        _prevLayerDerivatives.fill(0);
    }

    std::vector<GradientIterator> DenseLayer::getParameterGradients() {
        std::vector<GradientIterator> out;

        out.emplace_back(_weights, _weightDerivatives);
        out.emplace_back(_biases, _biaseDerivatives);

        return out;
    }

}