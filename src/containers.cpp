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
        assert(a.cols == b.rows && a.rows == result.rows);
        std::memset(result.data.get(), 0, result.rows * sizeof(float));

        for (int row = 0; row < a.rows; row++) {
            for (int col = 0; col < b.rows; col++) {
                result.data[row] += a.at(row, col) * b.data[col];
            }
        }
    }

    float dot(const Vector &a, const Vector &b) {
        assert(a.rows == b.rows);
        float out = 0.0f;
        for (int i = 0; i < a.rows; i++) {
            out += a.data[i] * b.data[i];
        }
        return out;
    }
}


Vector::Vector(int size): rows(size) {
    data = std::make_unique<float[]>(size);
    std::memset(data.get(), 0, rows*sizeof(float));
}

void Vector::operator+=(float scalar) {
    operators::add(data.get(), scalar, data.get(), rows);
}

void Vector::operator+=(const Vector &vector) {
    assert(rows == vector.rows);
    operators::add(data.get(), vector.data.get(), data.get(), rows);
}

void Vector::operator*=(float scalar) {
    operators::mul(data.get(), scalar, data.get(), rows);
}

float &Vector::operator[](size_t ind) {
    return data[ind];
}

void Vector::operator=(std::vector<float> vec) {
    assert(rows == vec.size());

    std::memcpy(data.get(), vec.begin().base(), rows * sizeof(float));
}

std::ostream &operator<<(std::ostream &os, const Vector &v) {
    for (int r=0; r<v.rows; r++) {
        os << v.data[r];
        os << std::string(" ");
    }
    return os;
}

Matrix::Matrix(int rows, int cols) : rows(rows), cols(cols), size(rows * cols){
    data = std::unique_ptr<float[]>(new(std::align_val_t(32)) float[size]);
    std::memset(data.get(), 0, rows*cols*sizeof(float));
}

float& Matrix::at(int row, int col) const {
    //assert(row < rows && col < cols);
    return  data[row * cols + col];
}

void Matrix::operator+=(float scalar) {
    operators::add(data.get(), scalar, data.get(), size);
}
void Matrix::operator+=(const Matrix &matrix) {
    operators::add(data.get(), matrix.data.get(), data.get(), size);
}
void Matrix::operator*=(float scalar) {
    operators::mul(data.get(), scalar, data.get(), size);
}

std::ostream &operator<<(std::ostream &os, Matrix const &m) {
    for (int r=0; r<m.rows; r++) {
        for (int c=0; c<m.cols; c++) {
            os << std::to_string(m.at(r, c));
            os << std::string(" ");
        }
        os << std::string("\n");
    }
    return os;
}



