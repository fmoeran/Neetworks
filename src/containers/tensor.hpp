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

    /// Flat Iterators exists to allow the user to traverse through a Tensor's values,
    /// pass a tensor's values between two functions or scopes without templates being an issue,
    /// or to allow another scope to edit the values within a tensor easily.
    /// FlatIterators should NEVER exist on their own (without their data being owned by a Tensor)
    /// and should only be treated as a temporary "window" to view a tensor's data.
    /// If a FlatIterator is given to another class, that class should avoid saving the iterator for later use as the
    /// data it points to may be freed. Instead, prefer to copy the data into a Tensor using Tensor.assign(iterator).
    struct FlatIterator {
    public:
        FlatIterator();
        FlatIterator(float* pBegin, float* pEnd);
        FlatIterator(std::vector<float> vec);
        float* begin();

        float* end();

        size_t size();

        size_t maxIndex();

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

        /// deep copies other
        Tensor(const Tensor<RANK>& other);

        Tensor operator=(Tensor& other);

        /// Returns the number of floats stored by the tensor
        [[nodiscard]] size_t size() const;

        /// Returns the size of each dimension as given in the constructor
        [[nodiscard]] std::array<size_t, RANK> dimensions() const;

        float& get(std::initializer_list<size_t> pos);

        FlatIterator getFlatIterator() const;

        /// copies the values from iter into the tensor.
        template<typename InputIter>
        void assign(InputIter iter);

        /// Sets all of the values randomly according to a normal distribution
        void randomizeNormal();

        void fill(float val);

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
        /// \param m matrix iterator
        /// \param v vector iterator
        /// \param result result vector iterator
        void vecMatMul(FlatIterator m, FlatIterator v, FlatIterator result);

        /// Multiplies the transpose of matrix with the vector
        /// \param m matrix iterator
        /// \param v vector iterator
        /// \param result result vector iterator
        void vecMatTransposeMul(FlatIterator m, FlatIterator v, FlatIterator result);
    }
}


#include "tensor.tpp"
