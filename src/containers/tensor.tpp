#pragma once




template<size_t RANK>
Tensor<RANK>::Tensor(std::initializer_list<size_t> dimensions) {
    assert(dimensions.size() == RANK);
    memcpy(_dimensions, dimensions.begin(), RANK * sizeof(size_t));
    _size = std::accumulate(dimensions.begin(), dimensions.end(), 1, std::multiplies<>());
    _data = std::make_unique<float[]>(size());
}

template<size_t RANK>
std::array<size_t, RANK> Tensor<RANK>::dimensions() const{
    std::array<size_t, RANK> out;
    memcpy(out.begin(), _dimensions, RANK * sizeof(size_t));
    return out;
}

template<size_t RANK>
size_t Tensor<RANK>::size() const {
    return _size;
}

template<size_t RANK>
float *Tensor<RANK>::begin() const {
    return _data.get();
}

template<size_t RANK>
float *Tensor<RANK>::end() const {
    return _data.get() + size();
}

template<size_t RANK>
size_t Tensor<RANK>::rank() const {
    return RANK;
}

template<size_t RANK>
template<typename InputIter>
void Tensor<RANK>::assign(InputIter begin, InputIter end) {
    assert(std::distance(begin, end) == size());
    std::copy(begin, end, _data.get());
}

template<size_t RANK>
void Tensor<RANK>::operator+=(float scalar) {
    operators::add(begin(), scalar, begin(), size());
}

template<size_t RANK>
void Tensor<RANK>::operator+=(const Tensor<RANK>& tensor) {
    assert(dimensions() == tensor.dimensions());
    operators::add(begin(), tensor.begin(), begin(), size());
}

template<size_t RANK>
void Tensor<RANK>::operator*=(float scalar) {
    operators::mul(begin(), scalar, begin(), size());
}


template<size_t RANK>
std::ostream &operator<<(std::ostream &os, const Tensor<RANK> &t) {
    if constexpr(RANK == 1) {
        for (int r=0; r<t.size(); r++) {
            os << std::to_string(t.begin()[r]);
            os << std::string(" ");
        }
        return os;
    }else {
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



