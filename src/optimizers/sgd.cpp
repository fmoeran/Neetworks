//
// Created by felix on 15/07/2024.
//

#include "sgd.hpp"
#include <stdexcept>

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
            std::vector<FlatIterator> parameterIterators = layer->getParameters();
            std::vector<FlatIterator> gradientIterators = layer->getParameterGradients();

            if (parameterIterators.size() != gradientIterators.size()) {
                throw std::runtime_error("Unequal number of parameters and gradients.");
            }

            for (size_t paramInd=0; paramInd<parameterIterators.size(); paramInd++) {
                _updateParameter(parameterIterators[paramInd], gradientIterators[paramInd], batchSize);
            }
        }
    }

    void SGD::_updateParameter(FlatIterator parameters, FlatIterator gradients, size_t batchSize) {
        if (parameters.size() != gradients.size()) {
            throw std::runtime_error("Unequal number of parameters and gradients.");
        }
        // gradients *= -_learningRate / batchSize
        operators::mul(gradients.begin(), -_learningRate / (float)batchSize, gradients.begin(), gradients.size());

        // parameters += gradients
        operators::add(parameters.begin(), gradients.begin(), parameters.begin(), parameters.size());
    }

}
