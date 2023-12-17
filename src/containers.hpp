//
// Created by Felix Moeran on 19/07/2023.
//

#pragma once

#include <vector>
#include <memory>

struct Vector;
struct Matrix;

namespace operators
{
    void add(float* a, float* b, float* result, size_t size);
    void add(float* a, float b, float* result, size_t size);
    void mul(float* a, float b, float* result, size_t size);

    void vecMatMul(const Matrix&a, const Vector& b, Vector& result);
    void vecMatMul(const Vector& a, const Matrix& b, Vector& result);
    float dot(const Vector& a, const Vector& b);
}


struct Vector{
public:
    explicit Vector(int rows);

    // returns the size (number of rows) in the vector
    [[nodiscard]] size_t size() const;

    // returns the array pointer to the first item in the vector
    [[nodiscard]] float* dataPtr() const;

    void operator+=(float scalar);
    void operator+=(const Vector& vector);

    void operator*=(float scalar);

    float& operator[](size_t ind) const;

    void operator=(std::vector<float> vec);

    friend std::ostream &operator<<(std::ostream &os, Vector const &v);
private:
    size_t _rows;
    std::unique_ptr<float[]> _data;
};



struct Matrix {
public:
    Matrix(int rows, int cols);

    [[nodiscard]] size_t rows() const;
    [[nodiscard]] size_t cols() const;
    [[nodiscard]] size_t size() const;

    // returns the array pointer to the first item in the contiguously stored data
    [[nodiscard]] float * dataPtr() const;

    [[nodiscard]] float at(int row, int col) const;

    float* operator[](size_t row) const;

    void operator+=(float scalar);
    void operator+=(const Matrix& matrix);

    void operator*=(float scalar);

    friend std::ostream &operator<<(std::ostream &os, Matrix const &m);

private:
    size_t _rows, _cols;
    std::unique_ptr<float[]> _data;
};



