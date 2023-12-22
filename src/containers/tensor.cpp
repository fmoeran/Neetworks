//
// Created by Felix Moeran on 20/12/2023.
//

#include <string>

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
    }
}