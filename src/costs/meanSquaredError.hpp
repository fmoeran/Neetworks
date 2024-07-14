//
// Created by felix on 14/07/2024.
//

#include "../cost.hpp"

namespace nw
{
    class MSE : public __Cost {
    public:
        float apply(FlatIterator target, FlatIterator found) override;

        void applyDeriv(FlatIterator target, FlatIterator found, FlatIterator result) override;


    };
}