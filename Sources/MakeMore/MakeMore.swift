// The Swift Programming Language
// https://docs.swift.org/swift-book
// The swift-mlx Documentation
// https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx

import MLX
import MLXRandom
import MLXNN
import Foundation
import Charts
import SwiftUI

@available(macOS 15.0, *)
@main
struct MakeMore {
    static func main() {
        // A 3x3 Matrix
        guard
            let namesUrl = Bundle.module.url(forResource: "names", withExtension: "txt"),
            let namesString = try? String(contentsOf: namesUrl, encoding: .utf8)
        else {
            fatalError("Can't extract contents of names.txt")
        }
        
        
        let names = namesString.split(separator: "\n").map{String($0)}
        let wrapperToken = "."
        let indexer = Indexer.create(from: names, wrapperToken: wrapperToken)
        /*MLXRandom.seed(1234)*/
        
        // MARK: BIGRAM
        /*bigram(names: names, indexer: indexer)*/

        // MARK: BigramNeural (Single Layer Bigram MLP)
        /*bigramNeuralNetwork(names: names, indexer: indexer, wrapperToken: wrapperToken)*/
       
        // MARK: MLP
        mlp(names: names, indexer: indexer, wrapperToken: wrapperToken)
    }
    
    @MainActor
    static func bigram(names: [String], indexer: Indexer, plot: Bool = true, evaluate: Bool = true) {
        let model = BigramModel.train(on: names, indexer: indexer)
        let samples = model.predict(20)
        print(samples)
        model.plotFrequencies()
        model.evaluate(on: names)
    }
    
    static func bigramNeuralNetwork(names: [String], indexer: Indexer, wrapperToken: String) {
        let (x, y) = BigramNeural.formatInputsOutputs(from: names, indexer: indexer, wrapperToken: wrapperToken)
        let model = BigramNeural(dimension: 27)
        model.train(inputs: x, outputs: y, learningRate: 50, epochSize: 200)
        let samples = model.sample(20, indexer: indexer)
        print(samples)
    }
    
    @MainActor
    static func mlp(names: [String], indexer: Indexer, wrapperToken: String) {
        let names = names.shuffled()
        
        /// https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf
        let (x, y) = MLP.formatInputsOutputs(
            from: Array(names),
            indexer: indexer,
            wrapperToken: wrapperToken,
            blockSize: 3,
            printDebug: false
        )
        
        let n1 = Int(0.8 * Float(names.count))
        let n2 = Int(0.9 * Float(names.count))
        let xTrain = x[0..<n1]
        let yTrain = y[0..<n1]
        
        let xDev = x[n1..<n2]
        let yDev = y[n1..<n2]
        let xTest = x[n2...]
        let yTest = y[n2...]
        
        let model = MLP(trainingDataSize: 27, hiddenLayerSize: 200, embeddingDimension: 10)
        model.train(inputs: xTrain, outputs: yTrain, initialLearningRate: 0.1, epochSize: 200000)
        
        model.evaluate(name: "train", inputs: xTrain, outputs: yTrain)
        model.evaluate(name: "dev", inputs: xDev, outputs: yDev)
        model.evaluate(name: "test", inputs: xTest, outputs: yTest)
        model.plotLosses()
        /*model.plotLearningRates(inputs: xTrain, outputs: yTrain)*/
        model.plotWordEmbedding(indexToTokenLookup: indexer.indexToTokenLookup)
        let samples = model.sample(20, blockSize: 3, indexer: indexer)
        print(samples)
    }
}
