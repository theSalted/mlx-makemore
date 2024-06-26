
//
//  MLP.swift
//
//
//  Created by Yuhao Chen on 6/23/24.
//

import Foundation
import MLX
import MLXNN
import MLXRandom
import MLXOptimizers
import MLXLinalg

import Charts
import SwiftUI

/// Multi-Layer Perceptron (MLP) Neural Network
///
/// This class implements a simple MLP with one hidden layer and cross-entropy loss.
/// Reference: https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf
class MLP: Module, UnaryLayer, MMNeuralNetwork {
    let hiddenLayerSize: Int
    let embeddingDimension: Int
    let outputSize: Int
    
    /// i.e. embeddings
    var C: MLXArray
    var weights1: MLXArray
    var biases1: MLXArray
    var weights2: MLXArray
    var biases2: MLXArray
    
    /// Array to store loss values during training
    var lossValues: [Float] = []
    
    /// Initializes the MLP with given dimensions
    ///
    /// - Parameters:
    ///   - trainingDataSize: Number of training samples
    ///   - hiddenLayerSize: Number of neurons in the hidden layer
    ///   - embeddingDimension: Dimension of the embedding vectors
    ///   - outputSize: Number of output classes
    init(trainingDataSize: Int, hiddenLayerSize: Int = 200, embeddingDimension: Int = 2, blockSize: Int = 3, outputSize: Int = 27) {
        self.hiddenLayerSize = hiddenLayerSize
        self.embeddingDimension = embeddingDimension
        self.outputSize = outputSize
        
        self.C = MLXRandom.normal([trainingDataSize, embeddingDimension])
        let w1Kaiming: Float = sqrt((5/3)/(Float(embeddingDimension) * Float(blockSize)))
        self.weights1 = MLXRandom.normal([embeddingDimension * blockSize, hiddenLayerSize]) * w1Kaiming
        self.biases1 = MLXRandom.normal([hiddenLayerSize]) * 0.01
        self.weights2 = MLXRandom.normal([hiddenLayerSize, outputSize]) * 0.01
        self.biases2 = MLXRandom.normal([outputSize]) * 0.01
    }
    
    /// Forward pass through the network
    ///
    /// - Parameter x: Input data
    /// - Returns: Logits
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let embedding = C[x]
        let embeddingConcatenate = embedding.reshaped(embedding.shape[0], embeddingDimension * 3)
        let hiddenLayerPreActivation = matmul(embeddingConcatenate, weights1) + biases1
        let hiddenLayerActivation = tanh(hiddenLayerPreActivation)
        let logits = matmul(hiddenLayerActivation, weights2) + biases2
        return logits
    }
    
    /// Computes the loss for the given model and data
    ///
    /// - Parameters:
    ///   - model: The MLP model
    ///   - x: Input data
    ///   - y: Target data
    /// - Returns: Cross-entropy loss
    func loss(model: MLP, x: MLXArray, y: MLXArray) -> MLXArray {
        return crossEntropy(logits: model(x), targets: y, reduction: .mean)
    }
    
    /// Trains the MLP model
    ///
    /// - Parameters:
    ///   - inputs: Input data
    ///   - outputs: Target data
    ///   - initialLearningRate: Initial learning rate
    ///   - decayedLearningRate: Decayed learning rate after 100,000 iterations
    ///   - epochSize: Number of epochs
    ///   - batchSize: Batch size
    ///   - debugPrint: Flag to enable/disable debug prints
    /// - Returns: Final loss value
    @discardableResult
    func train(
        inputs x: MLXArray,
        outputs y: MLXArray,
        initialLearningRate lr: Float = 1e-1,
        decayedLearningRate decayedLR: Float = 1e-2,
        epochSize: Int = 20,
        batchSize: Int = 32,
        debugPrint: Bool = true
    ) -> Float {
        if debugPrint {
            print("MLP training...")
        }
        
        eval(x, y)
        let inputSize = x.shape[0]
        // Print initial loss
        let initialLoss = loss(model: self, x: x, y: y)
        print("Initial Loss: \(initialLoss.asArray(Float.self)[0])")
        
        for epoch in 0..<epochSize {
            let indexes = MLXRandom.randInt(0..<inputSize, [batchSize])
            // Decay learning rate after 100,000 epochs
            let learningRate = epoch < 100000 ? lr : decayedLR
            let optimizer = SGD(learningRate: learningRate)
            
            let lossAndGradients = valueAndGrad(model: self, loss)
            
            let (loss, grads) = lossAndGradients(self, x[indexes], y[indexes])
            
            
            optimizer.update(model: self, gradients: grads)
            
            eval(self, optimizer)
            
            if debugPrint {
                print("MLP Epoch \(epoch + 1)/\(epochSize) Loss: \(loss.asArray(Float.self)[0]), Learning Rare: \(learningRate)")
            }
            
            // Track loss values
            lossValues.append(loss.asArray(Float.self)[0])
        }
        
        let totalLoss = loss(model: self, x: x, y: y)
        if debugPrint {
            print("MLP Total Loss: \(totalLoss)")
        }
        return totalLoss.item()
    }
    
    /// Evaluates the model on the given data
    ///
    /// - Parameters:
    ///   - name: Name of the evaluation
    ///   - inputs: Input data
    ///   - outputs: Target data
    /// - Returns: Evaluation loss value
    @discardableResult
    func evaluate(
        name: String,
        inputs x: MLXArray,
        outputs y: MLXArray
    ) -> Float {
        let evaluationLoss: Float = loss(model: self, x: x, y: y).item()
        print("MLP \(name) evaluation loss \(evaluationLoss)")
        return evaluationLoss
    }
    
    /// Generates samples using the trained MLP model
    ///
    /// This method uses the trained MLP model to generate samples of text or other
    /// sequence data. It starts from a given index (usually 0 for the beginning
    /// of a sequence) and generates samples by predicting the next token until
    /// it encounters the closing token.
    ///
    /// - Parameters:
    ///   - count: The number of samples to generate.
    ///   - indexer: An Indexer instance used to map indices to tokens.
    /// - Returns: An array of generated samples as strings.
    @discardableResult
    func sample(
        _ count: Int = 20,
        blockSize: Int,
        indexer: Indexer
    ) -> [String] {
        print("MLP predicting...")
        var results = [String]()
        for _ in 0...count {
            
            var context = Array(repeating: 0, count: blockSize)
            var out: [Int] = []
            
            while true {
                let reshapedContext = MLXArray(context, [1, blockSize])
                let embedding = C[reshapedContext]
                let hiddenLayerActivation = tanh(matmul(embedding.reshaped(1, -1), weights1) + biases1)
                let logits = matmul(hiddenLayerActivation, weights2) + biases2
                
                let probability = softmax(logits, axis: 1)
                let index: Int = multinomial(probability: probability, numberOfSamples: 1).item()
                
                context = context[1...] + [index]
                
                out.append(index)
                
                guard let predictedCharacter = indexer.indexToTokenLookup[index] else {
                    print("Couldn't find character from indexer")
                    break
                }
                if predictedCharacter == indexer.closingToken {
                    break
                }
            }
            let result = out.map({indexer.indexToTokenLookup[$0]!}).joined()
            print("MLP Sample: ", result)
            results.append(result)
        }
        return results
    }
}

extension MLP {
    /// Plots the log10 of the loss values tracked during training
    ///
    /// - Parameters:
    ///   - losses: Array of loss values to plot
    @available(macOS 15.0, *)
    @MainActor
    func plotLosses() {
        let log10Losses = lossValues.map { log10($0) }
        
        let chart = Chart(log10Losses.indices, id: \.self) { index in
            LineMark(x: .value("Epoch", index), y: .value("Loss", log10Losses[index]))
        }.background(.white).padding().frame(width: 1000, height: 1000)
        
        plot(chart, name: "MLP Log10 Loss Values")
    }
    
    /// Plots the learning rate vs. loss curve
    ///
    /// - Parameters:
    ///   - inputs: Input data
    ///   - outputs: Target data
    ///   - start: Starting value of the learning rate exponent
    ///   - stop: Ending value of the learning rate exponent
    ///   - count: Number of learning rate values to test
    @available(macOS 15.0, *)
    @MainActor
    func plotLearningRates(inputs x: MLXArray,
                           outputs y: MLXArray,
                           start: Float = -3,
                           stop: Float = 0,
                           count: Int = 1000) {
        print("MLP learning rate finding…")
        let lre = linspace(start, stop, count: count)
        let lrs = exp(lre)
        
        var losses: [LossRecord] = []
        for i in 0..<count {
            let lr: Float = lrs[i].item()
            let loss = train(inputs: x, outputs: y, initialLearningRate: lr, epochSize: 1, debugPrint: false)
            print("MLP lr: \(lr), loss: \(loss)")
            let record = LossRecord(rate: lr, loss: loss)
            losses.append(record)
        }
        
        let chart = Chart {
            LinePlot(losses, x: .value("Learning Rate", \.rate), y: .value("Loss", \.loss))
        }.background(.white).padding().frame(width: 1000, height: 1000)
        
        plot(chart, name: "MLP Learning Rates")
    }
    
    /// Plots the word embeddings in 2D space
    ///
    /// - Parameters:
    ///   - indexer: An Indexer instance to look up token names
    ///   - dimensionX: The dimension to use for the x-axis
    ///   - dimensionY: The dimension to use for the y-axis
    @MainActor
    func plotWordEmbedding(indexToTokenLookup: [Int: String], dimensionX: Int = 0, dimensionY: Int = 1) {
        let chart = Chart {
            ForEach(0..<C.shape[0], id: \.self) { [self] id in
                let x: Float = self.C[id][dimensionX].item()
                let y: Float = C[id][dimensionY].item()
                PointMark(x: .value("x", x),
                          y: .value("y", y))
                .symbolSize(.init(width: 20, height: 20))
                .annotation(position: .overlay) {
                    Text(indexToTokenLookup[id] ?? "<?>").foregroundStyle(Color.white)
                }
            }
        }.background(.white).padding().frame(width: 1000, height: 1000, alignment: .center)
        
        plot(chart, name: "MLP Word Feature Embedding")
    }
    
    /// Performs PCA on the embeddings to reduce dimensions to 2
    ///
    /// - Warning: Not implemented
    /// - Returns: Indices of the two principal components to plot
    func performPCA() {
        // Center the embeddings by subtracting the mean
        let meanCenteredEmbeddings = C - C.mean(axis: 0)
        
        // Compute the SVD
        let (_, _, Vt) = svd(meanCenteredEmbeddings)
        
        // The principal components are the first two columns of Vt.T (or rows of Vt)
        // The singular values S are already sorted in descending order
        let principalComponent1 = Vt[0]
        let principalComponent2 = Vt[1]
        
        print("MLP PC1: \(principalComponent1), PC2: \(principalComponent2)")
    }
    
    /// Struct to record loss at different learning rates
    struct LossRecord {
        let rate: Float
        let loss: Float
    }
}
