//
// Created by Felix Moeran on 20/12/2023.
//
#pragma once


#include <memory>
#include <iterator>
#include <string>
#include <numeric> // accumulate
#include <cassert> // assert
#include <array>
#include <initializer_list>


namespace nw {
    namespace operators {
        void add(const float *a, const float *b, float *result, size_t size);

        void add(const float *a, float b, float *result, size_t size);

        void mul(const float *a, float b, float *result, size_t size);
    }


    // A multidimensional array.
    // e.g. a Tensor with _rank 1 is a vector, and _rank 2 is a matrix, etc...
    template<size_t RANK>
    struct Tensor {
    public:
        Tensor(std::initializer_list<size_t> dimensions);

        // returns the number of floats stored by the tensor
        [[nodiscard]] size_t size() const;

        [[nodiscard]] std::array<size_t, RANK> dimensions() const;

        // returns the array pointer to the first item in the vector
        [[nodiscard]] float *begin() const;

        [[nodiscard]] float *end() const;

        template<typename InputIter>
        void assign(InputIter begin, InputIter end);

        [[nodiscard]] size_t rank() const;

        void operator+=(float scalar);

        void operator+=(const Tensor<RANK> &tensor);

        void operator*=(float scalar);

    private:
        std::unique_ptr<float[]> _data;
        size_t _dimensions[RANK], _size;
    };
}

#include "tensor.tpp"
