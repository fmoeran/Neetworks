//
// Created by felix on 11/07/2024.
//

#include "denseLayer.hpp"
#include <random>

namespace nw {
    DenseLayer::DenseLayer(size_t size, __Layer &prev, __Activation &activation)
            : _size(size), _previousSize(prev.size()), _biases({size}), _values({size}),
              _activatedValues({size}), _weights({size, prev.size()}),
              _biaseDerivatives({size}), _weightDerivatives({size, prev.size()}), _prevLayerDerivatives({prev.size()}),
              _prevLayerOutputs({_previousSize}), _activation(activation) {
        _activation = activation;

        _biases.randomizeNormal();
        _weights.randomizeNormal();
    }

    std::vector<FlatIterator> DenseLayer::getParameters() {
        return {_weights.getFlatIterator(), _biases.getFlatIterator()};
    }

    std::vector<FlatIterator> DenseLayer::getParameterGradients() {
        return {_weightDerivatives.getFlatIterator(), _biaseDerivatives.getFlatIterator()};
    }

    FlatIterator DenseLayer::propagate(FlatIterator previousOutput) {
        _prevLayerOutputs.assign(previousOutput);
        operators::vecMatMul(_weights.getFlatIterator(), previousOutput, _values.getFlatIterator());
        _values += _biases;
        _activation.apply(_values.getFlatIterator(), _activatedValues.getFlatIterator());
        return _values.getFlatIterator();
    }

    size_t DenseLayer::size() const {
        return _size;
    }

    FlatIterator DenseLayer::backPropagate(FlatIterator outputDerivatives) {
        // Calculate derivative of output w.r.t the parameter of the activation function
        Tensor<1> activationDerivatives({size()});
        _activation.applyDerivative(_values.getFlatIterator(), activationDerivatives.getFlatIterator());

        // Calculate derivative of Cost w.r.t the parameter of the activation function
        // These are also the change to the derivative w.r.t the bias terms
        Tensor<1> internalDerivatives({size()});
        operators::hadamard(outputDerivatives, activationDerivatives.getFlatIterator(),
                            internalDerivatives.getFlatIterator());

        FlatIterator it = internalDerivatives.getFlatIterator();

        // Calculate change to the derivative of Cost w.r.t weights
        Tensor<2> changeWeightDerivatives({size(), _previousSize});
        operators::vecTensorProduct(internalDerivatives.getFlatIterator(), _prevLayerOutputs.getFlatIterator(),
                                    changeWeightDerivatives);

        // Update parameter derivatives
        _biaseDerivatives  = internalDerivatives;
        _weightDerivatives = changeWeightDerivatives;

        // Derivative of Cost w.r.t the previous layer's output.
        // The reason this has to be a member variable is that it needs to stay in scope when this function finishes,
        // Otherwise its iterator would be pointing to free memory.
        operators::vecMatTransposeMul(_weights.getFlatIterator(), internalDerivatives.getFlatIterator(),
                                      _prevLayerDerivatives.getFlatIterator());

        return _prevLayerDerivatives.getFlatIterator();
    }
}