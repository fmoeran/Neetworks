//
// Created by felix on 14/07/2024.
//

#pragma once
#include "activation.hpp"

namespace nw {
    class __Cost {
    public:
        virtual float apply(FlatIterator target, FlatIterator found) {return 0.0;};

        virtual void applyDeriv(FlatIterator target, FlatIterator found, FlatIterator result) {};


    };
}