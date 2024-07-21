//
// Created by Felix Moeran on 04/04/2024.
//

#include "tensor.hpp"
#include <algorithm>
#include <cstring>
#include <ranges>
#include <execution>
#include <omp.h>

namespace nw
{


    namespace operators {
        void add(const float *a, const float *b, float *result, size_t size) {
            for (size_t i = 0; i < size; i++) {
                result[i] = a[i] + b[i];
            }
        }

        void add(const float *a, float b, float *result, size_t size) {
            for (size_t i = 0; i < size; i++) {
                result[i] = a[i] + b;
            }
        }

        void mul(const float *a, float b, float *result, size_t size) {
            for (size_t i = 0; i < size; i++) {
                result[i] = a[i] * b;
            }
        }

        float dot(const float *a, const float *b, size_t size) {
            float out = 0;
            for (size_t i = 0; i < size; i++) {
                out += a[i] * b[i];
            }
            return out;
        }

        void hadamard(FlatIterator a, FlatIterator b, FlatIterator result) {
            assert(a.size() == b.size() && b.size() == result.size());
            for (int i=0; i<a.size(); i++) {
                result[i] = a[i]*b[i];
            }
        }

        void vecTensorProduct(FlatIterator u, FlatIterator v, Tensor<2>& result) {
            assert(u.size()*v.size() == result.size());
            FlatIterator r = result.getFlatIterator();

            size_t height = u.size(), width = v.size();

            for (size_t i=0; i<height; i++) {
                for (size_t j=0; j<width; j++) {
                    r[i*width + j] = u.begin()[i]*v.begin()[j];
                }
            }
        }

        void vecMatMul(FlatIterator m, FlatIterator v, FlatIterator result) {
            assert(m.size() == v.size() * result.size());
            assert(v.begin() != result.begin() && m.begin() != result.begin());

            size_t matWidth = v.size();
            size_t matHeight = result.size();

            std::fill(result.begin(), result.end(), 0);
//#           pragma omp parallel for num_threads(4)
            for (size_t row = 0; row < matHeight; row++) {
                for (size_t col = 0; col < matWidth; col++) {
                    result[row] += m[row * matWidth + col] * v[col];
                }
            }

//            auto func = [&] (size_t row) {
//                for (size_t col=0, mIndex=row*matWidth; col<matWidth; col++, mIndex++) {
//                    result[row] += m[mIndex] * v[col];
//                }
//            };
//
//            auto rows = std::views::iota(0ull, matHeight);
//
//            std::for_each(std::execution::par, rows.begin(), rows.end(), func);
        }

        void vecMatTransposeMul(FlatIterator m, FlatIterator v, FlatIterator result){
            assert(m.size() == v.size() * result.size());
            size_t matWidth = result.size();
            size_t matHeight = v.size();

            std::fill(result.begin(), result.end(), 0);

            for (size_t col=0; col < matWidth; col++) {
                for (size_t row=0; row < matHeight; row++) {
                    result[col] += m[row*matWidth + col] * v[row];
                }
            }
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

    FlatIterator::FlatIterator(std::vector<float> vec) {
        _begin = vec.begin().base();
        _end = vec.end().base();
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

    std::ostream &operator<<(std::ostream &os, FlatIterator iter) {
        for (float val : iter) {
            os << val << ' ';
        }
        return os;
    }

    size_t FlatIterator::maxIndex() {
        return std::distance(begin(), std::max_element(begin(), end()));
    }
}
