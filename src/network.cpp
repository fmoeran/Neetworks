//
// Created by Felix Moeran on 19/12/2023.
//

#include "network.hpp"
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <string>

const int LOADING_BAR_WIDTH = 30;

namespace nw
{

    Network::Network(size_t inputSize) {
        _inputLayerPtr = new InputLayer(inputSize);
        _layers.push_back(_inputLayerPtr);

        _epochTrainingCorrect = 0;
        _epochTestingCorrect  = 0;
        _epochTestingCost     = 0;
        _cost                 = nullptr;
        _optimizer            = nullptr;

    }

    void Network::addLayer(__Layer *layer) {
        _layers.push_back(layer);
    }

    void Network::feedForward(FlatIterator inputIterator) {
        if (inputLayer().size() != inputIterator.size()) {
            throw std::invalid_argument("Input is not the same size as the network's input.");
        }
        _inputLayerPtr->loadInputs(inputIterator);
        for (__Layer* layer : _layers) {
            layer->propagate();
        }
    }

    __Layer& Network::getOutputLayer() {
        return *_layers.back();
    }
    FlatIterator Network::getOutput() {
        return getOutputLayer().getOutputs();
    }

    FlatIterator Network::getInput() {
        return _inputLayerPtr->getOutputs();
    }

    InputLayer& Network::inputLayer() {
        return *_inputLayerPtr;
    }

    std::ostream &operator<<(std::ostream &os, Network &n) {
        for (auto layer : n._layers) {
            os << layer->getOutputs() << '\n';
        }
        return os;
    }

    float Network::getCost(FlatIterator target) {
        return _cost->apply(target, getOutput());
    }


    void Network::train(Data trainingData, int epochs, int batchSize, Data testData) {
        // Run the epochs
        for (int currentEpoch = 0; currentEpoch < epochs; currentEpoch++) {
            trainingData.shuffle();
            _epochTrainingCorrect = 0;
            _trainEpoch(trainingData, batchSize);
            _printEpochInfo(currentEpoch, trainingData.size(), testData);
        }
    }

    void Network::train(Data trainingData, int epochs, int batchSize) {
        Data testData;
        train(trainingData, epochs, batchSize, testData);
    }

    void Network::_trainEpoch(Data trainingData, int batchSize) {
        if (trainingData.size() % batchSize != 0) {
            throw std::invalid_argument("The trainingData.size() must be divisible by batchSize");
        }

        int batchCount = trainingData.size() / batchSize;
        for (int currentBatch=0; currentBatch<batchCount; currentBatch++) {
            _printEpochProgressBar((float)currentBatch/(float)batchCount);

            int batchStart = currentBatch*batchSize;
            int batchEnd = batchStart + batchSize;

            for (int dataIndex=batchStart; dataIndex<batchEnd; dataIndex++) {
                _trainSingle(trainingData.inputs[dataIndex], trainingData.targets[dataIndex]);
            }
            _optimizer->updateLayers(batchSize);
            _resetGradients();
        }
    }

    void Network::_trainSingle(FlatIterator input, FlatIterator target) {
        feedForward(input);

        if (getSingleOutput() == target.maxIndex()) {
            _epochTrainingCorrect++;
        }

        _backPropagate(target);
    }

    void Network::_backPropagate(FlatIterator target) {
        if (getOutputLayer().size() != target.size()) {
            throw std::invalid_argument("Input is not the same size as the network's input.");
        }

        // Get derivative of last layer
        Tensor<1> outputLayerDerivatives({getOutputLayer().size()});
        _cost->applyDeriv(target, getOutput(), outputLayerDerivatives.getFlatIterator());
        FlatIterator derivativeIterator = outputLayerDerivatives.getFlatIterator();

        // Iterate backwards through layers
        for (size_t layerIndex= _layers.size() - 1; layerIndex > 0; layerIndex--) {
            derivativeIterator = _layers[layerIndex]->backPropagate(derivativeIterator);
        }
    }

    void Network::_resetGradients() {
        for (__Layer* layer : _layers) {
            layer->resetGradients();
        }
    }

    void Network::compile(__Cost *cost, __Optimizer* optimizer) {
        _cost = cost;
        _optimizer = optimizer;
        _optimizer->registerLayers(_layers);
    }

    int Network::getSingleOutput() {
        return getOutput().maxIndex();
    }

    void Network::_printEpochProgressBar(float progress) {
        _epochProgressBar = "";
        _epochProgressBar += "[";
        int pos = (int) (LOADING_BAR_WIDTH * progress);
        for (int i=0; i<LOADING_BAR_WIDTH; i++) {
            if (i <= pos) _epochProgressBar += "=";
            else _epochProgressBar += " ";
        }

        _epochProgressBar += "]" + std::to_string(int(progress * 100)) + "%\r";
        std::cout << _epochProgressBar;
        std::cout.flush();
    }

    void Network::_printEpochInfo(int currentEpoch, int trainingSize, Data testData) {
        _runTest(testData);

        // clear loading bar
        for (size_t i=0; i<_epochProgressBar.size(); i++) std::cout << ' ';
        std::cout << '\r';

        std::cout << "Epoch " << currentEpoch+1 << ": ";
        std::cout << "Training = " << _epochTrainingCorrect << "/" << trainingSize << " ";
        if (testData.size() > 0) {
            std::cout << "Testing = " << _epochTestingCorrect << "/" << testData.size() << " ";
            std::cout << "Cost = " << _epochTestingCost << " ";
        }
        std::cout << std::endl;
    }

    void Network::_runTest(Data testData) {
        _epochTestingCorrect = 0;
        _epochTestingCost    = 0;
        for (size_t testIndex=0; testIndex < testData.size(); testIndex++) {
            FlatIterator input(testData.inputs[testIndex]), target = testData.targets[testIndex];
            feedForward(input);
            _epochTestingCorrect += (getSingleOutput() == target.maxIndex());
            _epochTestingCost    += getCost(target);
        }
        _epochTestingCost /= (float)testData.size();
    }
}