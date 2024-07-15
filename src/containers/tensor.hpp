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
#include <vector>


namespace nw {

    /// Iterator class used to iterate over all of the items in a Tensor
    struct FlatIterator {
    public:
        FlatIterator();
        FlatIterator(float* pBegin, float* pEnd);
        FlatIterator(std::vector<float> vec);
        float* begin();

        float* end();

        size_t size();

        float& operator[](size_t ind);
    private:
        float *_begin, *_end;

        friend std::ostream &operator<<(std::ostream &os, FlatIterator iter);
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
        void assign(InputIter iter);

        /// Sets all of the values randomly according to a normal distribution
        void randomizeNormal();

        void operator+=(float scalar);

        void operator+=(Tensor<RANK> &tensor);

        void operator*=(float scalar);

        float dot(const Tensor<RANK>& other);

    private:
        std::unique_ptr<float[]> _data;
        size_t _dimensions[RANK]{}, _size;
        FlatIterator _iterator;
    };


    namespace operators {
        void add(const float *a, const float *b, float *result, size_t size);

        void add(const float *a, float b, float *result, size_t size);

        void mul(const float *a, float b, float *result, size_t size);

        float dot(const float *a, const float *b, size_t size);

        ///  Hadamard (inner) product of two tensors.
        void hadamard(FlatIterator a, FlatIterator b, FlatIterator result);

        /// The tensor product with two vectors.
        /// Later implementations of the full tensor product yet to come
        void vecTensorProduct(FlatIterator u, FlatIterator v, Tensor<2>& result);

        /// multiplies a matrix by a vector.
        /// \param m matrix pointer
        /// \param v vector pointer
        /// \param result result vector pointer
        /// \param matWidth number of columns in m, size of v
        /// \param matHeight number of rows in m, size result
        void vecMatMul(float* m, float *v, float *result, size_t matWidth, size_t matHeight);

        /// multiplies a matrix by a vector.
        /// \param m matrix iterator
        /// \param v vector iterator
        /// \param result result vector iterator
        void vecMatMul(FlatIterator m, FlatIterator v, FlatIterator result);

        /// Same as vecMatMul but with the matrix transposed
        void vecMatTransposeMul(float* m, float *v, float *result, size_t matWidth, size_t matHeight);

        /// Same as vecMatMul but with the matrix transposed
        void vecMatTransposeMul(FlatIterator m, FlatIterator v, FlatIterator result);
    }
}


#include "tensor.tpp"
