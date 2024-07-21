//
// Created by felix on 15/07/2024.
//

#include "momentum.hpp"

namespace nw
{

    Momentum::Momentum(float learningRate, float momentumRate) {
        _learningRate = learningRate;
        _momentumRate = momentumRate;
        _moments = std::vector<std::vector<Tensor<1>>>();
    }

    void Momentum::registerLayers(std::vector<__Layer *> layers) {
        _layers = layers;
        for (auto layer : _layers) {
            _moments.emplace_back();
            for (FlatIterator paramIter: layer->getParameters()) {
                _moments.back().push_back(Tensor<1>({paramIter.size()}));
            }
        }
    }

    void Momentum::updateLayers(size_t batchSize) {
        for (size_t layerIndex=0; layerIndex < _layers.size(); layerIndex++) {

            std::vector<Tensor<1>>& layerMoments         = _moments[layerIndex];
            std::vector<FlatIterator> layerParameters    = _layers[layerIndex]->getParameters();
            std::vector<FlatIterator> layerGradients = _layers[layerIndex]->getParameterGradients();

            for (size_t paramIndex=0; paramIndex < layerMoments.size(); paramIndex++) {
                _updateParameter(layerParameters[paramIndex], layerGradients[paramIndex], batchSize,
                                 layerMoments[paramIndex].getFlatIterator());
            }

        }
    }

    void Momentum::_updateParameter(FlatIterator parameters, FlatIterator gradients, size_t batchSize,
                                    FlatIterator momentum) {
        // momentum *= _momentumRate;
        operators::mul(momentum.begin(), _momentumRate, momentum.begin(), momentum.size());

        // gradients *= -_learningRate / batchSize
        operators::mul(gradients.begin(), -_learningRate / (float)batchSize, gradients.begin(), gradients.size());

        // momentum += gradients
        operators::add(momentum.begin(), gradients.begin(), momentum.begin(), momentum.size());

        // parameters += momentum
        operators::add(parameters.begin(), momentum.begin(), parameters.begin(), parameters.size());
    }
}