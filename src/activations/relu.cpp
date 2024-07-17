//
// Created by felix on 17/07/2024.
//

#include "relu.hpp"
#include <algorithm>


namespace nw
{
    void ReLU::apply(FlatIterator it, FlatIterator res) {
        for (size_t ind=0; ind<it.size(); ind++) {
            res[ind] = std::max(it[ind], 0.0f);
        }
    }

    void ReLU::applyDerivative(FlatIterator it, FlatIterator res) {
        for (size_t ind=0; ind<it.size(); ind++) {
            res[ind] = (float)(it[ind]>0);
        }
    }
}

