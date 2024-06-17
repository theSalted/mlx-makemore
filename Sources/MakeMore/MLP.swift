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

/// A simple MLP with single hidden layer
class SimpleMLP: Module, UnaryLayer, MMNeuralNetwork {
    var weights: MLXArray
    let dimension: Int
    init(dimension: Int = 27, key: MLXArray? = nil) {
        self.weights = MLXRandom.normal([dimension, dimension], key: key)
        self.dimension = dimension
    }
    
    func callAsFunction(_ x: MLX.MLXArray) -> MLX.MLXArray {
        /* The forward layers  (only 1 hidden layer) */
        
        // Input to the network - one-hot encoding
        let xEncoded = oneHot(x, numberOfClasses: dimension).asType(Float32.self)
        
        // Logits, i.e. the hidden layer
        let logits = matmul(xEncoded, weights)
        
        // Softmax
        let counts = logits.exp()
        let probability = counts / counts.sum(axis: 1, keepDims: true)
        
        return probability
    }
    
    func loss(model: SimpleMLP, x: MLXArray, y: MLXArray) -> MLXArray {
        /* Usually people consider to the loss part of forward layer too */
        let probability = model(x)
        return -probability[MLXArray(x.ndim), y].log().mean() + 0.01 * (weights ** 2).mean()
    }

    
    func train(
        inputs x: MLXArray,
        outputs y: MLXArray,
        learningRate lr: Float = 1e-1,
        epochSize: Int = 20
    ) {
        print("SimMLP training...")
        /* We use stochastic gradient descent as our backprop method */
        let optimizer = SGD(learningRate: lr)
        
        for epoch in 0...epochSize-1 {
            eval(x, y, self.parameters())
            
            let lg = valueAndGrad(model: self, loss)
            
            let (loss, grads) = lg(self, x, y)
            
            optimizer.update(model: self, gradients: grads)
            
            eval(self, optimizer)
            
            print("SimMLP Epoch \(epoch)/\(epochSize - 1): loss at \(loss.asArray(Float.self)[0])")
        }
    }
    
    func sample(_ count: Int = 0, indexer: Indexer, key: MLXArray? = nil) -> [String] {
        var results: [String] = []
        
        print("SimMLP predicting...")
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
