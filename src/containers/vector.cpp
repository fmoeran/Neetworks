//
// Created by Felix Moeran on 20/12/2023.
//
#include "vector.hpp"
#include "tensor.hpp"
#include "matrix.hpp"


namespace operators
{
    float dot(const float* a, const float* b, size_t size) {
        float out = 0;
        for (int i=0; i<size; i++) {
            out += a[i] * b[i];
        }
        return out;
    }
}


Tensor<1>::Tensor(size_t size): _size(size) {
    _data = std::make_unique<float[]>(size);
    std::memset(begin(), 0, size * sizeof(float));

}

float *Vector::begin() const {
    return _data.get();
}

float *Vector::end() const {
    return begin() + size();
}

size_t Vector::size() const {
    return _size;
}

std::array<size_t, 1> Vector::dimensions() const {
    return std::array<size_t, 1>({size()});
}

size_t Vector::rank() {
    return 1;
}

void Vector::operator+=(float scalar) {
    operators::add(begin(), scalar, begin(), size());
}

void Vector::operator+=(const Vector& vector) {
    assert(size() == vector.size());
    operators::add(begin(), vector.begin(), begin(), size());
}

void Vector::operator*=(float scalar) {
    operators::mul(begin(), scalar, begin(), size());
}

float Vector::dot(const Vector &vector) {
    assert(size() == vector.size());
    return operators::dot(begin(), vector.begin(), size());
}

float &Vector::operator[](size_t ind) const{
    assert(ind < size());
    return *(begin() + ind);
}


