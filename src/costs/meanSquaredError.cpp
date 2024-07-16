//
// Created by felix on 14/07/2024.
//

#include "meanSquaredError.hpp"

namespace nw
{
    float MSE::apply(FlatIterator target, FlatIterator found) {
        assert(target.size() == found.size());
        float out = 0;
        for (size_t i=0; i<target.size(); i++) {
            float dif = target[i] - found[i];
            out += dif * dif;
        }
        return out;
    }

    void MSE::applyDeriv(FlatIterator target, FlatIterator found, FlatIterator result) {
        assert(target.size() == found.size() && found.size() == result.size());
        for (size_t i=0; i<target.size(); i++) {
            result[i] = (found[i] - target[i]);
        }
    }
}
