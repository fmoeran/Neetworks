//
// Created by Felix Moeran on 04/04/2024.
//
#include "tensor.hpp"

namespace nw
{


    namespace operators {
        void add(const float *a, const float *b, float *result, size_t size) {
            for (int i = 0; i < size; i++) {
                result[i] = a[i] + b[i];
            }
        }

        void add(const float *a, float b, float *result, size_t size) {
            for (int i = 0; i < size; i++) {
                result[i] = a[i] + b;
            }
        }


        void mul(const float *a, float b, float *result, size_t size) {
            for (int i = 0; i < size; i++) {
                result[i] = a[i] * b;
            }
        }

        float dot(const float *a, const float *b, size_t size) {
            float out = 0;
            for (int i = 0; i < size; i++) {
                out += a[i] * b[i];
            }
            return out;
        }

        void vecMatMul(float *m, float *v, float *result, size_t matWidth, size_t matHeight) {
            assert(v != result && m != result);
            std::memset((void *) result, 0, matHeight * sizeof(float));

            for (size_t row = 0; row < matHeight; row++) {
                for (size_t col = 0; col < matWidth; col++) {
                    result[row] += m[row * matWidth + col] * v[col];
                }
            }
        }


        void vecMatMul(FlatIterator m, FlatIterator v, FlatIterator result) {
            assert(m.size() == v.size() * result.size());
            size_t matWidth = v.size();
            size_t matHeight = result.size();
            vecMatMul(m.begin(), v.begin(), result.begin(), matWidth, matHeight);
        }
    }

    FlatIterator::FlatIterator() {
        _begin = nullptr;
        _end = nullptr;
    }

    FlatIterator::FlatIterator(float* pBegin, float* pEnd) {
        _begin = pBegin;
        _end = pEnd;
    }

    float *FlatIterator::begin() {
        return _begin;
    }


    float *FlatIterator::end() {
        return _end;
    }

    size_t FlatIterator::size() {
        return std::distance(_begin, _end);
    }

    float& FlatIterator::operator[](size_t ind) {
        assert(ind < size());
        return begin()[ind];
    }

    }
