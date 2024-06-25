//
//  MLP.swift
//
//
//  Created by Yuhao Chen on 6/17/24.
//

import Foundation
import MLX
import MLXNN
import MLXRandom
import MLXOptimizers

/// A simple MLP with single hidden layer and process
class BigramNeural: Module, UnaryLayer, MMNeuralNetwork {
    var weights: MLXArray
    let dimension: Int
    
    init(dimension: Int) {
        self.weights = MLXRandom.normal([dimension, dimension])
        self.dimension = dimension
    }
    
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        /* The forward layers  (only 1 hidden layer) */
        
        // Input to the network - one-hot encoding
        let xEncoded = oneHot(x, numberOfClasses: dimension).asType(Float32.self)
        
        /// Logits, i.e. the output of hidden layer before activation function is applied
        let logits = matmul(xEncoded, weights)
        
        // Softmax
        let counts = logits.exp()
        let probabilities = counts / counts.sum(axis: 1, keepDims: true)
        
        return probabilities
    }
    
    func loss(model: BigramNeural, x: MLXArray, y: MLXArray) -> MLXArray {
        /* Usually people consider to the loss part of forward layer too */
        let probabilities = model(x)
        return -probabilities[MLXArray(0..<x.shape[0]), y].log().mean() + 0.01 * (weights ** 2).mean() // Cross entropy
    }

    
    func train(
        inputs x: MLXArray,
        outputs y: MLXArray,
        learningRate lr: Float = 1e-1,
        epochSize: Int = 20,
        debugPrint: Bool = true
    ) {
        if debugPrint {
            print("Bigram Neural training...")
        }
        /* We use stochastic gradient descent as our backprop method */
        let optimizer = SGD(learningRate: lr)
        
        eval(x, y, self.parameters())
        
        for epoch in 0..<epochSize {
            
            let lossAndGradients = valueAndGrad(model: self, loss)
            
            let (loss, grads) = lossAndGradients(self, x, y)
            
            optimizer.update(model: self, gradients: grads)
            
            eval(self, optimizer) // Prevent graphs from getting too large by eval it every epoch. Reference: MLX - Lazy Evaluation
            if debugPrint {
                print("Bigram Neural Epoch \(epoch + 1)/\(epochSize) Loss: \(loss.asArray(Float.self)[0])")
            }
            
        }
    }
    
    func sample(
        _ count: Int = 0,
        indexer: Indexer
    ) -> [String] {
        var results: [String] = []
        
        print("Bigram Neural predicting...")
        for _ in 0...count {
            var index = 0
            var result = ""
            
            while true {
                let probability = callAsFunction(MLXArray([index]))
                index = multinomial(probability: probability, numberOfSamples: 1).asArray(Int.self)[0]
                guard let predictedCharacter = indexer.indexToTokenLookup[index] else {
                    print("Couldn't find character from indexer")
                    break
                }
                if predictedCharacter == indexer.closingToken {
                    break
                }
                result += predictedCharacter
            }
            
            results.append(result)
        }
        return results
    }
    
}
