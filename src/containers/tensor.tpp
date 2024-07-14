#pragma once
#include <iostream>

namespace nw
{
    template<size_t RANK>
    Tensor<RANK>::Tensor(std::initializer_list<size_t> dimensions){
        assert(dimensions.size() == RANK);

        _size = std::accumulate(dimensions.begin(), dimensions.end(), 1, std::multiplies<>());
        _data = std::make_unique<float[]>(size());
        _iterator = FlatIterator(_data.get(), _data.get() + _size);

        memcpy(_dimensions, dimensions.begin(), RANK * sizeof(size_t));
    }

    template<size_t RANK>
    std::array<size_t, RANK> Tensor<RANK>::dimensions() const {
        std::array<size_t, RANK> out;
        memcpy(out.begin(), _dimensions, RANK * sizeof(size_t));
        return out;
    }

    template<size_t RANK>
    size_t Tensor<RANK>::size() const {
        return _size;
    }


    template<size_t RANK>
    float &Tensor<RANK>::get(std::initializer_list<size_t> pos) {
        assert(pos.size() == RANK);
        size_t index = 0;
        for (int i=RANK-1, step=1; i>=0; step*=dimensions()[i], i--){
            index += step * pos.begin()[i];
        }
        assert(index < size());
        return _data[index];
    }

    template<size_t RANK>
    template<typename InputIter>
    void Tensor<RANK>::assign(InputIter iter) {
        assert((size_t)std::distance(iter.begin(), iter.end()) == size());
        std::copy(iter.begin(), iter.end(), _data.get());
    }

    template<size_t RANK>
    void Tensor<RANK>::operator+=(float scalar) {
        operators::add(_iterator.begin(), scalar, _iterator.begin(), size());
    }

    template<size_t RANK>
    void Tensor<RANK>::operator+=(Tensor <RANK> &tensor) {
        assert(dimensions() == tensor.dimensions());
        operators::add(_iterator.begin(), tensor.getFlatIterator().begin(), _iterator.begin(), size());
    }

    template<size_t RANK>
    void Tensor<RANK>::operator*=(float scalar) {
        operators::mul(_iterator.begin(), scalar, _iterator.begin(), size());
    }


    template<size_t RANK>
    std::ostream &operator<<(std::ostream &os, Tensor <RANK> &t) {
        if constexpr (RANK == 1) { // VECTOR

            for (int r = 0; r < t.size(); r++) {
                os << std::to_string(t.getFlatIterator().begin()[r]);
                os << std::string(" ");
            }
        } else if constexpr (RANK == 2) { // MATRIX
            for (int r = 0; r < t.dimensions()[0]; r++) {
                for (int c = 0; c < t.dimensions()[1]; c++) {
                    os << std::to_string(*(t.getFlatIterator().begin() + r * t.dimensions()[1] + c));
                    os << std::string(" ");
                }
                os << std::string("\n");
            }
        } else { // 3 onwards Tensor
            os << std::string("Tensor<") << std::to_string(t.rank()) << std::string(">");
            os << std::string("{");
            for (int d = 0; d < t.rank(); d++) {
                os << std::to_string(t.dimensions()[d]);
                if (d != t.rank() - 1) {
                    os << std::string(", ");
                }
            }
            os << std::string("}");
        }
        return os;
    }

    template<size_t RANK>
    FlatIterator Tensor<RANK>::getFlatIterator() {
        return _iterator;
    }

    template<size_t RANK>
    float Tensor<RANK>::dot(const Tensor<RANK>& other){
        assert(size() == other.size());
        return operators::dot(_iterator.begin(), _iterator.end(), size());

    }
}


