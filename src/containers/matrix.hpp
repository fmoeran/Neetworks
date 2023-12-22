//
// Created by Felix Moeran on 20/12/2023.
//

#pragma once

#include "tensor.hpp"
#include "vector.hpp"

namespace nw {
// Wrapper for a rank 2 tensor
    using Matrix = Tensor<2>;


    namespace operators {
        void vecMatMul(const Matrix &a, const Vector &b, Vector &result);

        void vecMatMul(const Vector &a, const Matrix &b, Vector &result);
    }


    template<>
    struct Tensor<2> {
    public:
        Tensor(size_t rows, size_t cols);

        // returns the number of floats stored by the Vector
        [[nodiscard]] size_t size() const;

        [[nodiscard]] std::array<size_t, 2> dimensions() const;

        [[nodiscard]] size_t rows() const;

        [[nodiscard]] size_t cols() const;

        // returns the array pointer to the first item in the vector
        [[nodiscard]] float *begin() const;

        [[nodiscard]] float *end() const;

        template<typename InputIter>
        void assign(InputIter begin, InputIter end);

        [[nodiscard]] static size_t rank();

        void operator+=(float scalar);

        void operator+=(const Matrix &matrix);

        void operator*=(float scalar);

        float *operator[](size_t row) const;

    private:
        size_t _size, _rows, _cols;
        std::unique_ptr<float[]> _data;
    };

    template<typename InputIter>
    void Matrix::assign(InputIter begin, InputIter end) {
        assert(std::distance(begin, end) == size());
        std::copy(begin, end, _data.get());
    }

}
