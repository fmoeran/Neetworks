//
// Created by felix on 17/07/2024.
//

#pragma once

#include "../activation.hpp"

namespace nw
{
    class ReLU :  public __Activation{
    public:

        void apply(FlatIterator it, FlatIterator res) override;

        void applyDerivative(FlatIterator it, FlatIterator res) override;

    };
}