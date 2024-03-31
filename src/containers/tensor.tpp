#pragma once

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

        void vecMatMul(Tensor<2> &m, Tensor<1> &v, Tensor<1> &result) {
            assert(&v != &result);
            assert(m.dimensions()[1] == v.size() && m.dimensions()[0] == result.size());
            std::memset(result.getFlatIterator().begin(), 0, result.size() * sizeof(float));

            for (size_t row = 0; row < result.size(); row++) {
                for (size_t col = 0; col < v.size(); col++) {
                    result.get({row}) += m.get({row, col}) * v.get({col});
                }
            }

        }
    }
    template<size_t RANK>
    Tensor<RANK>::Tensor(std::initializer_list<size_t> dimensions){
        assert(dimensions.size() == RANK);

        _size = std::accumulate(dimensions.begin(), dimensions.end(), 1, std::multiplies<>());
        _data = std::make_unique<float[]>(size());
        _iterator = FlatIterator(_data.get(), _data.get() + _size);

        memcpy(_dimensions, dimensions.begin(), RANK * sizeof(size_t));
    }

    template<size_t RANK>
    std::array<size_t, RANK> Tensor<RANK>::dimensions() const {
        std::array<size_t, RANK> out;
        memcpy(out.begin(), _dimensions, RANK * sizeof(size_t));
        return out;
    }

    template<size_t RANK>
    size_t Tensor<RANK>::size() const {
        return _size;
    }


    template<size_t RANK>
    float &Tensor<RANK>::get(std::initializer_list<size_t> pos) {
        assert(pos.size() == RANK);
        size_t index = 0;
        for (int i=RANK-1, step=1; step+=dimensions()[i], i--; i>=0){
            index += step * pos.begin()[i];
        }
        return _data[index];
    }

    template<size_t RANK>
    template<typename InputIter>
    void Tensor<RANK>::assign(InputIter begin, InputIter end) {
        assert(std::distance(begin, end) == size());
        std::copy(begin, end, _data.get());
    }

    template<size_t RANK>
    void Tensor<RANK>::operator+=(float scalar) {
        operators::add(_iterator.begin(), scalar, _iterator.begin(), size());
    }

    template<size_t RANK>
    void Tensor<RANK>::operator+=(const Tensor <RANK> &tensor) {
        assert(dimensions() == tensor.dimensions());
        operators::add(_iterator.begin(), tensor.begin(), _iterator.begin(), size());
    }

    template<size_t RANK>
    void Tensor<RANK>::operator*=(float scalar) {
        operators::mul(_iterator.begin(), scalar, _iterator.begin(), size());
    }


    template<size_t RANK>
    std::ostream &operator<<(std::ostream &os, Tensor <RANK> &t) {
        if constexpr (RANK == 1) { // VECTOR

            for (int r = 0; r < t.size(); r++) {
                os << std::to_string(t.getFlatIterator().begin()[r]);
                os << std::string(" ");
            }
            return os;
        } else if constexpr (RANK == 2) { // MATRIX
            for (int r = 0; r < t.dimensions()[0]; r++) {
                for (int c = 0; c < t.dimensions()[1]; c++) {
                    os << std::to_string(*(t.getFlatIterator().begin() + r * t.dimensions()[1] + c));
                    os << std::string(" ");
                }
                os << std::string("\n");
            }
            return os;
        } else { // 3 onwards Tensor
            os << std::string("Tensor<") << std::to_string(t.rank()) << std::string(">");
            os << std::string("{");
            for (int d = 0; d < t.rank(); d++) {
                os << std::to_string(t.dimensions()[d]);
                if (d != t.rank() - 1) {
                    os << std::string(", ");
                }
            }
            os << std::string("}");
            return os;
        }
    }

    template<size_t RANK>
    FlatIterator Tensor<RANK>::getFlatIterator() {
        return _iterator;
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

    template<size_t RANK>
    float Tensor<RANK>::dot(const Tensor<RANK>& other){
        assert(size() == other.size());
        return operators::dot(_iterator.begin(), _iterator.end(), size());

    }


}


