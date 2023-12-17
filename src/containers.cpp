//
// Created by Felix Moeran on 19/07/2023.
//
#include "containers.hpp"
#include <string>
#include <new>
#include <cassert> // assert
#include <cstring> // memset
#include <iostream>

namespace operators {
    void add(float *a, float *b, float *result, size_t size) {
        while (size--) {
            *result = (*a) + (*b);
            a++;
            b++;
            result++;
        }
    }

    void add(float *a, float b, float *result, size_t size) {
        while (size--) {
            *result = (*a) + b;
            a++; result++;
        }
    }


    void mul(float *a, float b, float *result, size_t size) {
        while (size--) {
            *result = (*a) * b;
            a++;
            result++;
        }
    }

    void vecMatMul(const Vector &a, const Matrix &b, Vector &result) {
        vecMatMul(b, a, result);
    }

    void vecMatMul(const Matrix &a, const Vector &b, Vector &result) {
        assert(a.cols() == b.size() && a.rows() == result.size());
        std::memset(result.dataPtr(), 0, result.size() * sizeof(float));

        for (int row = 0; row < a.rows(); row++) {
            for (int col = 0; col < b.size(); col++) {
                result.dataPtr()[row] += a.at(row, col) * b.dataPtr()[col];
            }
        }
    }

    float dot(const Vector &a, const Vector &b) {
        assert(a.size() == b.size());
        float out = 0.0f;
        for (int i = 0; i < a.size(); i++) {
            out += a.dataPtr()[i] * b.dataPtr()[i];
        }
        return out;
    }
}


Vector::Vector(int rows): _rows(rows) {
    _data = std::make_unique<float[]>(rows);
    std::memset(dataPtr(), 0, rows*sizeof(float));
}

void Vector::operator+=(float scalar) {
    operators::add(dataPtr(), scalar, dataPtr(), _rows);
}

void Vector::operator+=(const Vector &vector) {
    assert(size() == vector.size());
    operators::add(dataPtr(), vector.dataPtr(), dataPtr(), size());
}

void Vector::operator*=(float scalar) {
    operators::mul(dataPtr(), scalar, dataPtr(), size());
}

float &Vector::operator[](size_t ind) const {
    return _data[ind];
}

void Vector::operator=(std::vector<float> vec) {
    assert(size() == vec.size());

    std::memcpy(dataPtr(), vec.begin().base(), size() * sizeof(float));
}

std::ostream &operator<<(std::ostream &os, const Vector &v) {
    for (int r=0; r<v.size(); r++) {
        os << v.dataPtr()[r];
        os << std::string(" ");
    }
    return os;
}

size_t Vector::size() const{
    return _rows;
}

float *Vector::dataPtr() const{
    return _data.get();
}

Matrix::Matrix(int rows, int cols) : _rows(rows), _cols(cols){
    _data = std::make_unique<float[]>(size());
    std::memset(dataPtr(), 0, rows*cols*sizeof(float));
}

float Matrix::at(int row, int col) const {
    assert(row < _rows && col < _cols);
    return _data[row * _cols + col];
}

float* Matrix::operator[](size_t row) const{
    assert(row<_rows);
    return dataPtr() + row*_cols;
}

void Matrix::operator+=(float scalar) {
    operators::add(dataPtr(), scalar, dataPtr(), size());
}
void Matrix::operator+=(const Matrix &matrix) {
    operators::add(dataPtr(), matrix.dataPtr(), dataPtr(), size());
}
void Matrix::operator*=(float scalar) {
    operators::mul(dataPtr(), scalar, dataPtr(), size());
}

std::ostream &operator<<(std::ostream &os, Matrix const &m) {
    for (int r=0; r<m.rows(); r++) {
        for (int c=0; c<m.cols(); c++) {
            os << std::to_string(m.at(r, c));
            os << std::string(" ");
        }
        os << std::string("\n");
    }
    return os;
}

size_t Matrix::rows() const {
    return _rows;
}

size_t Matrix::cols() const {
    return _cols;
}

size_t Matrix::size() const {
    return _rows*_cols;
}

float * Matrix::dataPtr() const {
    return _data.get();
}





