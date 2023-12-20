#include <iostream>
#include <vector>
#include "containers/tensor.hpp"




int main() {
    Tensor<2> t({100, 3});
    std::vector<float> v(300, 3);
    t.assign(v.begin(), v.end());
    for (auto val : t) {
        std::cout <<val << ' ';
    }
    std::cout << std::endl;
    return 0;
}