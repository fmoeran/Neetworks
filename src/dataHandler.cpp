//
// Created by Felix Moeran on 18/07/2023.
//
#include "dataHandler.hpp"

#include <string>
#include <fstream>
#include <random>
#include <cassert>
#include <algorithm>
#include <exception>

std::default_random_engine rnd;


Data::Data(std::vector<std::vector<float>>& inp, std::vector<std::vector<float>>& targ) {
    if (inp.size() != targ.size()) {
        throw std::invalid_argument("Input and target vectors must have the same size");
    }
    inputs = inp;
    targets = targ;
}

Data::Data() {
    inputs = std::vector<std::vector<float>>(0);
    targets = std::vector<std::vector<float>>(0);
}

void Data::shuffle() {
    std::vector<size_t> inds(size());
    std::iota(inds.begin(), inds.end(), 0);
    std::shuffle(inds.begin(), inds.end(), rnd);
    std::vector<std::vector<float>> oldInputs = inputs;
    std::vector<std::vector<float>> oldTargets = targets;
    for (int i=0; i<size(); i++) {
        inputs[i] = oldInputs[inds[i]];
        targets[i] = oldTargets[inds[i]];
    }
}

size_t Data::size() {
    return inputs.size();
}

namespace mnist
{
    std::vector<unsigned char> readLabels(std::string fileLocation) {
        std::ifstream file;
        file.open(fileLocation, std::ios::in | std::ios::binary);
        assert(file.is_open());
        int magic, numItems;
        unsigned char m[4], n[4];

        file.read((char*)m, sizeof(m));
        file.read((char*)n, sizeof(n));

        magic = ((int)m[0] << 24) | ((int)m[1] << 16) | ((int)m[2] << 8) | ((int)m[3]);
        numItems = ((int)n[0] << 24) | ((int)n[1] << 16) | ((int)n[2] << 8) | ((int)n[3]);

        assert(magic == 2049);

        std::vector<unsigned char> out(numItems);
        file.read((char*)out.begin().base(), numItems);
        file.close();

        return out;
    }

    std::vector<unsigned char> readImages(std::string fileLocation, int& rows, int& cols) {
        std::ifstream file;
        file.open(fileLocation, std::ios::in | std::ios::binary);
        assert(file.is_open());
        int magic, numItems;
        unsigned char m[4], n[4], r[4], c[4];

        file.read((char*)m, sizeof(m));
        file.read((char*)n, sizeof(n));
        file.read((char*)r, sizeof(r));
        file.read((char*)c, sizeof(c));

        magic =    ((int)m[0] << 24) | ((int)m[1] << 16) | ((int)m[2] << 8) | ((int)m[3]);
        numItems = ((int)n[0] << 24) | ((int)n[1] << 16) | ((int)n[2] << 8) | ((int)n[3]);
        rows =     ((int)r[0] << 24) | ((int)r[1] << 16) | ((int)r[2] << 8) | ((int)r[3]);
        cols =     ((int)c[0] << 24) | ((int)c[1] << 16) | ((int)c[2] << 8) | ((int)c[3]);

        assert(magic == 2051);

        std::vector<unsigned char> out(numItems * rows * cols);
        file.read((char*)out.begin().base(), numItems * rows * cols);

        file.close();

        return out;
    }

    Data getMnistData(std::string specifier) {
        int rows, cols;
        std::vector<unsigned char> labels = readLabels("../data/mnist/"+specifier+"-labels.idx1-ubyte");
        std::vector<unsigned char> images = readImages("../data/mnist/"+specifier+"-images.idx3-ubyte", rows, cols);

        std::vector<std::vector<float>> inputs(images.size()/(rows*cols), std::vector<float>(rows*cols));
        for (int i=0; i<inputs.size(); i++) {
            for (int j=0; j<rows*cols; j++) {
                inputs[i][j] = ((float)(int)images[i*rows*cols + j])/255.0f;
            }
        }

        std::vector<std::vector<float>> targets(labels.size(), std::vector<float>(10, 0.0f));
        for (int i=0; i<labels.size(); i++) {
            targets[i][labels[i]] = 1.0f;
        }
        return {inputs, targets };
    }
}


Data getMnistTrainingData() {
    return mnist::getMnistData("train");
}

Data getMnistTestData() {
    return mnist::getMnistData("t10k");
}



// creates a .ppm image from a greyscale set of integers
void uploadPPM(std::string& fileLocation, int width, int height, int maxCol, const char* ptr) {
    std::ofstream file;
    file.open(fileLocation, std::ios::out);
    assert(file.is_open());

    file << "P6" << '\n'
         << width << '\n'
         << height << '\n'
         << maxCol << '\n';
    for(int i=0; i<width*height; i++) {
        file << *ptr << *ptr<< *ptr;
        ptr++;
    }
    file.close();
}