//
// Created by felix on 11/07/2024.
//
#include "sigmoid.hpp"
#include <cmath>
#include <cassert>

namespace nw
{
    void Sigmoid::apply(FlatIterator it, FlatIterator res) {
        assert(it.size() == res.size());
        for (size_t i = 0; i < it.size(); i++) {
            res[i] = apply(it[i]);
        }
    }

    float Sigmoid::apply(float z) {
        return 1.0f / (1 + std::exp(-z));
    }

    void Sigmoid::applyDeriv(FlatIterator it, FlatIterator res) {
        assert(it.size() == res.size());
        for (size_t i = 0; i < it.size(); i++) {
            res[i] = applyDeriv(it[i]);
        }
    }

    float Sigmoid::applyDeriv(float z) {
        float res = apply(z);
        return res * (1 - res);
    }

}
