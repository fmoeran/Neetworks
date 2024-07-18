//
// Created by felix on 18/07/2024.
//

#include "crossEntropy.hpp"
#include <ranges>
#include <algorithm>
#include <execution>

namespace nw
{

    float CrossEntropy::apply(FlatIterator target, FlatIterator found) {
        assert(target.size() == found.size());
        float out = 0;

        auto inds = std::views::iota(0ull, target.size());
        auto func = [&](size_t i) {out += applySingle(target[i], found[i]);};

        std::for_each(std::execution::seq, inds.begin(), inds.end(), func);

        return out;
    }

    void CrossEntropy::applyDerivative(FlatIterator target, FlatIterator found, FlatIterator result) {
        assert(target.size() == found.size() && found.size() == result.size());

        auto inds = std::views::iota(0ull, target.size());
        auto func = [&](size_t i) {result[i] = applyDerivativeSingle(target[i], found[i]);};

        std::for_each(std::execution::seq, inds.begin(), inds.end(), func);
    }

    float CrossEntropy::applySingle(float y, float a) {
        if (a == 0 || a == 1) return 0;
        else if (y == 1)      return - y * std::log(a);
        else                  return - (1-y) * std::log(1-a);
    }

    float CrossEntropy::applyDerivativeSingle(float y, float a) {
        if (a == 0 || a == 1) return 0;
        else if (y == 1)      return -y/a;
        else                  return (1-y)/(1-a);
    }
}