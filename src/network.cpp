//
// Created by Felix Moeran on 19/12/2023.
//

#include "network.hpp"
#include <iostream>
#include <stdexcept>

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
            _trainEpoch(trainingData, batchSize);
        }

    }

    void Network::_trainEpoch(Data trainingData, int batchSize) {
        assert(trainingData.count % batchSize == 0);
        int batchCount = trainingData.count / batchSize;
        for (int currentBatch=0; currentBatch<batchCount; currentBatch++) {
            std::cout << currentBatch << std::endl;
            int batchStart = currentBatch*batchSize;
            int batchEnd = batchStart + batchSize;

            for (int dataIndex=batchStart; dataIndex<batchEnd; dataIndex++) {
                _trainSingle(trainingData.inputs[dataIndex], trainingData.targets[dataIndex]);
            }
        }
    }

    void Network::_trainSingle(FlatIterator input, FlatIterator target) {
        feedForward(input);
    }

    void Network::_backPropagate(FlatIterator target) {
        if (getOutputLayer().size() != target.size()) {
            throw std::invalid_argument("Input is not the same size as the network's input.");
        }

        


    }
}