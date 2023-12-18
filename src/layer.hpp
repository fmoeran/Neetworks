//
// Created by Felix Moeran on 17/12/2023.
//

#pragma once
#include "activation.hpp"
#include "containers.hpp"



struct __Layer {
public:
    // Uses the layer's weights, biases, etc. to update the layer's values (found in getOutputs)
    virtual void propagate() {};
    // Retrieves a Vector of values that the layer has produced AFTER a propagate call
    virtual const Vector& getOutputs() {return emptyVec;};
    // Returns the size of the output vector of the layer
    virtual size_t size() const {return 0;};

    //virtual void backpropagate() {};

protected:
    // non-linear activation function
    __Activation* _activation;
    // previous layer in the network (set to nullptr with an input layer)
    __Layer* _previous;
};

// A layer that will always be at the start of any network
struct InputLayer: public __Layer {
public:
    explicit InputLayer(size_t size);

    void propagate() override;
    const Vector& getOutputs() override;

    [[nodiscard]] size_t size() const override;
private:
    size_t _size;
    Vector _values;
};



// The basic NN layer, edges between every node and every previous layer node, with weights biases and an activation function
struct DenseLayer : public __Layer {
public:
    explicit DenseLayer(size_t size, __Layer* prev, __Activation* activation);

    void propagate() override;
    const Vector& getOutputs() override;

    size_t size() const override;
private:
    size_t _size;
    Vector _biases, _values, _activatedValues;
    Matrix _weights;
};


