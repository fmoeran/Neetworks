//
// Created by Felix Moeran on 20/12/2023.
//

#pragma once

#include "tensor.hpp"

// Wrapper for a rank 1 tensor
using Vector = Tensor<1>;

namespace operators
{
    float dot(const float* a, const float* b, size_t size);
}

template<>
struct Tensor<1> {
public:
    explicit Tensor(size_t size);

    // returns the number of floats stored by the Vector
    [[nodiscard]] size_t size() const;

    [[nodiscard]] std::array<size_t, 1> dimensions() const;

    // returns the array pointer to the first item in the vector
    [[nodiscard]] float* begin() const;
    [[nodiscard]] float* end() const;

    template<typename InputIter>
    void assign(InputIter begin, InputIter end);

    [[nodiscard]] static size_t rank();

    void operator+=(float scalar);
    void operator+=(const Vector& vector);

    void operator*=(float scalar);

    float dot(const Vector& vector);

    float& operator[](size_t ind) const;

private:
    size_t _size;
    std::unique_ptr<float[]> _data;

};


// This has to be defined here due to templates being instanced only when they are needed
// and it is the only template defined function


template<typename InputIter>
void Vector::assign(InputIter begin, InputIter end) {
    assert(std::distance(begin, end) == size());
    std::copy(begin, end, _data.get());
}









