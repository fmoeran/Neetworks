//
// Created by Felix Moeran on 20/12/2023.
//

#include "matrix.hpp"

namespace operators
{
    void vecMatMul(const Vector &a, const Matrix &b, Vector &result) {
        vecMatMul(b, a, result);
    }

    void vecMatMul(const Matrix &a, const Vector &b, Vector &result) {
        assert(&b != &result);
        assert(a.cols() == b.size() && a.rows() == result.size());
        std::memset(result.begin(), 0, result.size() * sizeof(float));

        for (int row = 0; row < a.rows(); row++) {
            for (int col = 0; col < b.size(); col++) {
                result.begin()[row] += a[row][col] * b.begin()[col];
            }
        }
    }
}

Tensor<2>::Tensor(size_t rows, size_t cols): _size(rows*cols), _rows(rows), _cols(cols){
    _data = std::make_unique<float[]>(_size);
    std::memset(begin(), 0, size() * sizeof(float));
}

float *Matrix::begin() const {
    return _data.get();
}

float *Matrix::end() const {
    return begin() + size();
}

size_t Matrix::size() const {
    return _size;
}

std::array<size_t, 2> Matrix::dimensions() const {
    return std::array<size_t, 2>({_rows, _cols});
}

size_t Matrix::rows() const {
    return _rows;
}

size_t Matrix::cols() const {
    return _cols;
}

size_t Matrix::rank() {
    return 2;
}

void Matrix::operator+=(float scalar) {
    operators::add(begin(), scalar, begin(), size());
}

void Matrix::operator+=(const Matrix& matrix) {
    assert(dimensions() == matrix.dimensions());
    operators::add(begin(), matrix.begin(), begin(), size());
}

void Matrix::operator*=(float scalar) {
    operators::mul(begin(), scalar, begin(), size());
}

float *Matrix::operator[](size_t row) const{
    size_t ind = row * _cols;
    assert(ind < size());
    return begin() + ind;

}



