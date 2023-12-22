//
// Created by Felix Moeran on 17/12/2023.
//

#include "activation.hpp"
#include <cmath>
#include <cassert>

namespace nw
{
    void Sigmoid::apply(const Vector &vec, Vector &res) {
        assert(vec.size() == res.size());
        for (int i = 0; i < vec.size(); i++) {
            res[i] = apply(vec[i]);
        }
    }

    float Sigmoid::apply(float z) {
        return 1.0f / (1 + std::exp(-z));
    }

    void Sigmoid::applyDeriv(const Vector &vec, Vector &res) {
        assert(vec.size() == res.size());
        for (int i = 0; i < vec.size(); i++) {
            res[i] = applyDeriv(vec[i]);
        }
    }

    float Sigmoid::applyDeriv(float z) {
        float res = apply(z);
        return res * (1 - res);
    }

}
