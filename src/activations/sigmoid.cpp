//
// Created by felix on 11/07/2024.
//
#include "sigmoid.hpp"
#include <cmath>
#include <cassert>
#include <algorithm>
#include <execution>

namespace nw
{
    void Sigmoid::apply(FlatIterator it, FlatIterator res) {
        assert(it.size() == res.size());

        std::transform(std::execution::par_unseq,it.begin(), it.end(), res.begin(), applySingle);
    }

    float Sigmoid::applySingle(float z) {
        return 1.0f / (1 + std::exp(-z));
    }

    void Sigmoid::applyDerivative(FlatIterator it, FlatIterator res) {
        assert(it.size() == res.size());

        std::transform(std::execution::par_unseq, it.begin(), it.end(), res.begin(), applyDerivativeSingle);
    }

    float Sigmoid::applyDerivativeSingle(float z) {
        float res = applySingle(z);
        return res * (1 - res);
    }

}
