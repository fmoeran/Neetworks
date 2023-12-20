//
// Created by Felix Moeran on 17/12/2023.
//

#pragma once
#include"containers/containers.hpp"

class __Activation{
public:
    // applies the function to an entire vector, placing the result in res
    virtual void apply(const Vector& vec, Vector& res) {};
    // returns the result of the activation function on z
    virtual float apply(float z) {return 0;};
    // applies the derivative to an entire vector, placing the result in res
    virtual void applyDeriv(const Vector& vec, Vector& res) {};
    // returns the result of the derivative on z
    virtual float applyDeriv(float z) {return 0;};
};

class Sigmoid : public __Activation {
public:
    void apply(const Vector& vec, Vector& res) override;
    float apply(float z) override;
    void applyDeriv(const Vector& vec, Vector& res) override;
    float applyDeriv(float z) override;
};
