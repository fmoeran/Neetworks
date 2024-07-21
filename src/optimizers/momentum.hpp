//
// Created by felix on 15/07/2024.
//

#pragma once
#include "../optimizer.hpp"

namespace nw
{
    class Momentum : public __Optimizer {
    public:
        Momentum(float learningRate, float momentumRate);

        void registerLayers(std::vector<__Layer*> layers) override;

        void updateLayers(size_t batchSize) override;

    private:

        float _learningRate, _momentumRate;
        std::vector<std::vector<Tensor<1>>> _moments;
        std::vector<__Layer *> _layers;

        void _updateParameter(FlatIterator parameters, FlatIterator gradients, size_t batchSize, FlatIterator momentum);

    };
}


