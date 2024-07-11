//
// Created by Felix Moeran on 17/12/2023.
//

#pragma once
#include "containers/tensor.hpp"

namespace nw {
    class __Activation {
    public:
        // applies the function to an entire Tensor, placing the result in res
        virtual void apply(FlatIterator it, FlatIterator res) {};

        // returns the result of the activation function on z
        virtual float apply(float z) { return 0; };

        // applies the derivative to an entire FlatIterator, placing the result in res
        virtual void applyDeriv(FlatIterator it, FlatIterator res) {};

        // returns the result of the derivative on z
        virtual float applyDeriv(float z) { return 0; };
    };



}
