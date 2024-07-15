//
// Created by felix on 15/07/2024.
//

#include "sgd.hpp"

namespace nw
{

    SGD::SGD(float learningRate) {
        _learningRate = learningRate;
        _layers = {};
    }

    void SGD::registerLayers(std::vector<__Layer *> layers) {
        _layers = layers;
    }

    void SGD::updateLayers(size_t batchSize) {
        for (__Layer* layer : _layers) {
            std::vector<GradientIterator> iterators = layer->getParameterGradients();
            for (GradientIterator iterator : iterators) {
                _updateParameter(iterator, batchSize);
            }
            layer->resetGradients();
        }
    }

    void SGD::_updateParameter(GradientIterator gradientIterator, size_t batchSize) {
        FlatIterator parameters = gradientIterator.parameters;
        FlatIterator gradients  = gradientIterator.gradients;

        operators::mul(gradients.begin(), -_learningRate / (float)batchSize, gradients.begin(), gradients.size());

        operators::add(parameters.begin(), gradients.begin(), parameters.begin(), parameters.size());
    }

}
