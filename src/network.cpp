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
    }

    void Network::addLayer(__Layer *layer) {
        _layers.push_back(layer);
    }

    void Network::feedForward(FlatIterator inputIterator) {
        if (inputLayer().size() != inputIterator.size()) {
            throw std::invalid_argument("Input is not the same size as the network's input.");
        }
        assert(_inputLayerPtr->size() == inputIterator.size());
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

    void Network::train(Data trainingData, int epochs, int batchSize, bool track) {
        // Run the epochs
        for (int currentEpoch = 0; currentEpoch < epochs; currentEpoch++) {
            trainingData.shuffle();
            _currentEpochCorrect = 0;
            _trainEpoch(trainingData, batchSize);
            _printEpochInfo(currentEpoch, trainingData.count);
        }

    }

    void Network::_trainEpoch(Data trainingData, int batchSize) {
        assert(trainingData.count % batchSize == 0);
        int batchCount = trainingData.count / batchSize;
        for (int currentBatch=0; currentBatch<batchCount; currentBatch++) {
            _printEpochProgressBar((float)currentBatch/(float)batchCount);

            int batchStart = currentBatch*batchSize;
            int batchEnd = batchStart + batchSize;

            for (int dataIndex=batchStart; dataIndex<batchEnd; dataIndex++) {
                _trainSingle(trainingData.inputs[dataIndex], trainingData.targets[dataIndex]);
            }

            _optimizer->updateLayers(batchSize);
        }
    }

    void Network::_trainSingle(FlatIterator input, FlatIterator target) {
        feedForward(input);

        if (getSingleOutput() == std::distance(target.begin(), std::max_element(target.begin(), target.end()))) {
            _currentEpochCorrect++;
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

    void Network::compile(__Cost *cost, __Optimizer* optimizer) {
        _cost = cost;
        _optimizer = optimizer;
        _optimizer->registerLayers(_layers);
    }

    int Network::getSingleOutput() {
        FlatIterator it = getOutput();
        int max = 0;
        for (size_t i=0; i<it.size(); i++) {
            if (it[i] > it[max]) max = i;
        }
        return max;
    }

    void Network::_printEpochProgressBar(float progress) {
        std::string out = "";
        out += "[";
        int pos = (int) (LOADING_BAR_WIDTH * progress);
        for (int i=0; i<LOADING_BAR_WIDTH; i++) {
            if (i <= pos) out += "=";
            else out += " ";
        }
        // TODO: fix
        out += "]" + std::to_string(int(progress * 100)) + "%\r";
        std::cout << out;
        std::cout.flush();
    }

    void Network::_printEpochInfo(int currentEpoch, int trainingSize) {
        // clear loading bar
        for (int i=0; i<LOADING_BAR_WIDTH+10; i++) std::cout << ' ';
        std::cout << '\r';
        std::cout << "Epoch " << currentEpoch+1 << ": Training = " << _currentEpochCorrect << "/" << trainingSize;
        std::cout << std::endl;
    }
}