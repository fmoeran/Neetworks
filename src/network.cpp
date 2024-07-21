//
// Created by Felix Moeran on 19/12/2023.
//

#include "network.hpp"
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <string>
#include <fstream>

const int LOADING_BAR_WIDTH = 30;

namespace nw
{

    Network::Network(size_t inputSize): _recentOutputs({1}) {
        _inputLayerPtr = new InputLayer(inputSize);
        _layers.push_back(_inputLayerPtr);

        _currentEpoch         = 0;
        _epochTrainingCorrect = 0;
        _epochTestingCorrect  = 0;
        _epochTestingCost     = 0;
        _cost                 = nullptr;
        _optimizer            = nullptr;
    }

    void Network::addLayer(__Layer *layer) {
        _layers.push_back(layer);
        Tensor<1> t({layer->size()});
        _recentOutputs = t;
    }

    void Network::feedForward(FlatIterator inputIterator) {
        if (inputLayer().size() != inputIterator.size()) {
            throw std::invalid_argument("Input is not the same size as the network's input.");
        }
        FlatIterator layerOutput = inputIterator;
        for (__Layer* layer : _layers) {
            layerOutput = layer->propagate(layerOutput);
        }
        _recentOutputs.assign(layerOutput);
    }

    __Layer& Network::getOutputLayer() {
        return *_layers.back();
    }
    FlatIterator Network::getOutput() {
        return _recentOutputs.getFlatIterator();
    }

    InputLayer& Network::inputLayer() {
        return *_inputLayerPtr;
    }

    std::ostream &operator<<(std::ostream &os, Network &n) {
        for (auto layer : n._layers) {
            os << layer << '\n';
        }
        return os;
    }

    float Network::getCost(FlatIterator target) {
        return _cost->apply(target, getOutput());
    }

    void Network::train(Data trainingData, int epochs, int batchSize, Data testData) {
        if (trainingData.size() % batchSize != 0) {
            throw std::invalid_argument("The trainingData.size() must be divisible by batchSize");
        }
        // Run the epochs
        for (_currentEpoch = 0; _currentEpoch < epochs; _currentEpoch++) {
            _trainEpoch(trainingData, batchSize);
            _runTest(testData);
            _printEpochInfo(trainingData.size(), testData.size());
            _uploadEpochStats(trainingData.size(), testData.size());
        }
    }

    void Network::train(Data trainingData, int epochs, int batchSize) {
        Data testData;
        train(trainingData, epochs, batchSize, testData);
    }

    // TODO: store the total parameter derivatives somewhere and give them to _optimizer.
    // The optimizer is currently just taking the gradients from __Layer::getPatameterGradients() which only
    // gets the gradient of the very last piece of training data, not the whole batch.
    void Network::_trainEpoch(Data trainingData, int batchSize) {
        _epochTrainingCorrect = 0;
        _epochStartTime = std::chrono::high_resolution_clock::now();

        trainingData.shuffle();

        int batchCount = (int)trainingData.size() / batchSize;
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
        _cost->applyDerivative(target, getOutput(), outputLayerDerivatives.getFlatIterator());
        FlatIterator derivativeIterator = outputLayerDerivatives.getFlatIterator();

        // Iterate backwards through layers
        for (size_t layerIndex= _layers.size() - 1; layerIndex > 0; layerIndex--) {
            derivativeIterator = _layers[layerIndex]->backPropagate(derivativeIterator);
        }
    }

    void Network::_resetGradients() {
        for (__Layer* layer : _layers) {
            layer->resetParameterGradients();
        }
    }

    void Network::compile(__Cost *cost, __Optimizer* optimizer) {
        _cost = cost;
        _optimizer = optimizer;
        _optimizer->registerLayers(_layers);
    }

    int Network::getSingleOutput() {
        return (int)getOutput().maxIndex();
    }

    void Network::_printEpochProgressBar(float progress) {

        _epochProgressBar = "";
        _epochProgressBar += "[";
        int pos = (int) (LOADING_BAR_WIDTH * progress);
        for (int i=0; i<LOADING_BAR_WIDTH; i++) {
            if (i <= pos) _epochProgressBar += "=";
            else _epochProgressBar += " ";
        }

        _epochProgressBar += "]" + std::to_string(int(progress * 100)) + "% ";
        _epochProgressBar += std::to_string(_getEpochDuration()) + "s";
        _epochProgressBar += "\r";

        std::cout << _epochProgressBar;
        std::cout.flush();
    }

    void Network::_printEpochInfo(size_t trainingSize, size_t testSize) {

        // clear loading bar
        for (size_t i=0; i<_epochProgressBar.size(); i++) std::cout << ' ';
        std::cout << '\r';

        std::cout << "Epoch " << _currentEpoch+1;
        std::cout <<  " (" << _getEpochDuration() << "s): ";
        std::cout << "Training = " << _epochTrainingCorrect << "/" << trainingSize << " ";
        if (testSize > 0) {
            std::cout << "Testing = " << _epochTestingCorrect << "/" << testSize << " ";
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

    void Network::_openStatsCSV(std::ios_base::openmode mode) {
        _epochStatsCSV.open(_epochStatsLocation, mode);
        if (!_epochStatsCSV.is_open()) {
            throw std::runtime_error("Could not open csv file.");
        }
    }

    void Network::_closeStatsCSV() {
        _epochStatsCSV.close();
    }

    void Network::autoSaveStats(const std::string& csvLocation, std::ios_base::openmode mode) {
        _epochStatsLocation = csvLocation;
        _openStatsCSV(mode);

        _epochStatsCSV << "Epoch,Training Accuracy,Testing Accuracy,Cost\n";
        _closeStatsCSV();
    }

    void Network::_uploadEpochStats(size_t trainingSize, size_t testSize) {
        if (_epochStatsLocation.empty()) return;

        _openStatsCSV();
        _epochStatsCSV << _currentEpoch + 1 << ",";
        _epochStatsCSV << ((float)_epochTrainingCorrect / (float)trainingSize) << ",";
        _epochStatsCSV << ((float)_epochTestingCorrect / (float)testSize) << ",";
        _epochStatsCSV << _epochTestingCost;
        _epochStatsCSV << '\n';
        _closeStatsCSV();
    }

    long long Network::_getEpochDuration() {
        auto timeNow = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(timeNow - _epochStartTime);
        return duration.count();
    }

}