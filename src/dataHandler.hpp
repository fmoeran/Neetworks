//
// Created by Felix Moeran on 17/07/2023.
//

#pragma once

#include <vector>
#include <string>
#include <fstream>

// a container for training/test data for a neural network
struct Data {
public:
    std::vector<std::vector<float>> inputs;
    std::vector<std::vector<float>> targets;
    Data();
    Data(std::vector<std::vector<float>>& inp, std::vector<std::vector<float>>& targ);

    size_t size();

    void shuffle();

};



Data getMnistTrainingData();
Data getMnistTestData();