#include <iostream>
#include <vector>
#include "containers/vector.hpp"
#include "containers/tensor.hpp"
#include "containers/matrix.hpp"


int main() {
    using namespace nw;
    Matrix m1(2, 2);
    m1 += 2;

    std::vector<float> vec = {1, 0, 0, 1};

    Matrix m2(2, 2);
    m2.assign(vec.begin(), vec.end());

    Vector v1(2);
    v1[0] = 1;

    Vector v2(2);

    //std::cout << m1;

    operators::vecMatMul(m2, v1, v2);

    std::cout << v2 << std::endl;


    return 0;
}