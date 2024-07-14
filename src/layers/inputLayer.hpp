//
// Created by felix on 11/07/2024.
//

#pragma once
#include "../layer.hpp"

namespace nw
{
/// A layer that will always be at the start of any network
    struct InputLayer : public __Layer {
    public:
        explicit InputLayer(size_t size);

        void propagate() override;

        template<typename InputIter>
        void loadInputs(InputIter iter);

        FlatIterator getOutputs() override;

        [[nodiscard]] size_t size() const override;

    private:
        size_t _size;
        Tensor<1> _values;
    };

    template<typename It>
    void InputLayer::loadInputs(It iter) {
        _values.assign(iter);
    }

}

