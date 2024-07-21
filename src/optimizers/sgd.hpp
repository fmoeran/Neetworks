//
// Created by felix on 15/07/2024.
//

#pragma once

#include "../optimizer.hpp"


namespace nw
{
    class SGD : public __Optimizer {
    public:
        SGD(float learningRate);

        void registerLayers(std::vector<__Layer*> layers) override;

        void updateLayers(size_t batchSize) override;

    private:

        float _learningRate;
        std::vector<__Layer *> _layers = {};

        void _updateParameter(FlatIterator parameters, FlatIterator gradients, size_t batchSize);

    };
}
