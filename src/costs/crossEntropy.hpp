//
// Created by felix on 18/07/2024.
//

#pragma once
#include "../cost.hpp"

namespace nw
{
    /// Applies the Cross Entropy loss function on a binary classifier.
    /// Each of the possible outputs of a network that use this cost should be desired to either be 0 or 1.
    /// So in "set" notation: target[i] in {0, 1}, found[i] in (0, 1).
    class CrossEntropy : public __Cost {
    public:
        float apply(FlatIterator target, FlatIterator found) override;

        static float applySingle(float y, float a);

        void applyDerivative(FlatIterator target, FlatIterator found, FlatIterator result) override;

        static float applyDerivativeSingle(float y, float a);
    };
}



