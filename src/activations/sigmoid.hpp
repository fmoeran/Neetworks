//
// Created by felix on 11/07/2024.
//

#pragma once

#include "../activation.hpp"


namespace nw{

    class Sigmoid : public __Activation {
    public:
        void apply(FlatIterator it, FlatIterator res) override;

        static float applySingle(float z);

        void applyDerivative(FlatIterator it, FlatIterator res) override;

        static float applyDerivativeSingle(float z);
    };
}


