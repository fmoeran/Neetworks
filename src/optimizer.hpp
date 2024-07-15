//
// Created by felix on 15/07/2024.
//

#pragma once

#include "layer.hpp"
#include <vector>


namespace nw
{
    class __Optimizer {
    public:

        virtual void registerLayers(std::vector<__Layer*> layers);

        virtual void updateLayers(size_t batchSize);
    };
}

