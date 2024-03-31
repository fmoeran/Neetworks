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

    /// Iterator class used to iterate over all of the items in a Tensor
    struct FlatIterator {
    public:
        FlatIterator();
        FlatIterator(float* pBegin, float* pEnd);

        float* begin();

        float* end();
    private:
        float *_begin, *_end;
    };
    /// A multidimensional array.
    /// e.g. a Tensor with RANK 1 is a vector, and RANK 2 is a matrix, etc...
    /// \tparam RANK No. of dimensions in the Tensor.
    template<size_t RANK>
    struct Tensor {
    public:
        Tensor(std::initializer_list<size_t> dimensions);

        /// Returns the number of floats stored by the tensor
        [[nodiscard]] size_t size() const;

        /// Returns the size of each dimension as given in the constructor
        [[nodiscard]] std::array<size_t, RANK> dimensions() const;

        float& get(std::initializer_list<size_t> pos);

        FlatIterator getFlatIterator();

        template<typename InputIter>
        void assign(InputIter begin, InputIter end);

        void operator+=(float scalar);

        void operator+=(const Tensor<RANK> &tensor);

        void operator*=(float scalar);

        float dot(const Tensor<RANK>& other);

    private:
        std::unique_ptr<float[]> _data;
        size_t _dimensions[RANK], _size;
        FlatIterator _iterator;
    };


    namespace operators {
        void add(const float *a, const float *b, float *result, size_t size);

        void add(const float *a, float b, float *result, size_t size);

        void mul(const float *a, float b, float *result, size_t size);

        float dot(const float *a, const float *b, size_t size);

        void vecMatMul(Tensor<2> &m, Tensor<1> &v, Tensor<1> &result);
    }

}

#include "tensor.tpp"
