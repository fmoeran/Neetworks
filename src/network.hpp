//
// Created by Felix Moeran on 19/12/2023.
//

#pragma once

#include "layer.hpp"
#include <vector>

namespace nw
{

    struct Network {
    public:
        Network(size_t inputSize);

        void addLayer(__Layer *layer);

    private:
        std::vector<__Layer *> _layers;


    };
}


