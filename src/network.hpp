//
// Created by Felix Moeran on 19/12/2023.
//

#pragma once

#include "layer.hpp"
#include "layers/inputLayer.hpp"
#include "dataHandler.hpp"
#include "cost.hpp"
#include "optimizer.hpp"
#include <vector>
#include <string>

namespace nw
{

    struct Network {
    public:
        Network(size_t inputSize);

        /// Adds a layer to the NN
        /// @param layer pointer to an object inheriting from the __Layer class
        void addLayer(__Layer *layer);

        /// Runs the NN, updating the values stored in each layer
        void feedForward(FlatIterator inputIterator);

        void compile(__Cost* cost, __Optimizer* optimizer);

        /// Trains the network on a given dataset.
        /// \param trainingData The dataset to train the network on.
        /// \param epochs The number of epochs (cycles of training on the training data) to be performed.
        /// \param batchSize The size of the mini batches within each epoch. The size of the trainingData must be divisible by this.
        /// \param testData The dataset to test the network on after each epoch.
        void train(Data trainingData, int epochs, int batchSize, Data testData);

        /// Trains the network on a given dataset without testing.
        /// \param trainingData The dataset to train the network on.
        /// \param epochs The number of epochs (cycles of training on the training data) to be performed.
        /// \param batchSize The size of the mini batches within each epoch. The size of the trainingData must be divisible by this.
        void train(Data trainingData, int epochs, int batchSize);

        /// Returns the output that the network most favoured in the most recent pass
        int getSingleOutput();

        /// Returns the cost of the most recent feedForward call.
        /// \param target The desired output to be measured against.
        float getCost(FlatIterator target);

        /// Pointer to the final layer in the network
        /// Often used to assign a new layer after this one
        __Layer& getOutputLayer();

        /// Pointer to the input layer, will
        InputLayer& inputLayer();

        /// Returns the output tensor from the final layer in the network
        FlatIterator getOutput();

        /// Returns the input tensor most recently used by the network
        FlatIterator getInput();

    private:
        /// Network parameters
        std::vector<__Layer *> _layers;
        InputLayer *_inputLayerPtr;
        __Cost* _cost;
        __Optimizer* _optimizer;

        /// Current Epoch info
        std::string _epochProgressBar;
        int _epochTrainingCorrect, _epochTestingCorrect;
        float _epochTestingCost; // average

        void _trainEpoch(Data trainingData, int batchSize);

        // Forward and Back propagates the data on the input, updating the gradients of layers
        void _trainSingle(FlatIterator input, FlatIterator target);

        void _backPropagate(FlatIterator target);

        void _resetGradients();

        void _printEpochProgressBar(float progress);

        void _printEpochInfo(int currentEpoch, int trainingSize, Data testData);

        /// Runs the network on the testData, updating _epochTestingCorrect and _epochTestingCost
        void _runTest(Data testData);

        friend std::ostream &operator<<(std::ostream &os, Network &n);
    };

}





