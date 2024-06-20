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
        let indexer = Indexer.create(from: names, wrapperToken: ".")
        /*MLXRandom.seed(1234)*/
        
        // MARK: BIGRAM
        /*bigram(names: names, indexer: indexer)*/

        // MARK: BigramNeural (Single Layer Bigram MLP)
        /*bigramNeuralNetwork(names: names, indexer: indexer, wrapperToken: wrapperToken)*/
        
        let paddingSize = 3
        let (x, y) = BigramNeural.createInputsOutputs(
            from: Array(names[0..<5]),
            indexer: indexer, 
            wrapperToken: wrapperToken,
            openingPaddingSize: paddingSize,
            printDebug: true
        )
        print(x.shape, x.dtype)
        print(y.shape, y.dtype)
    }
    
    @MainActor
    static func bigram(names: [String], indexer: Indexer, plot: Bool = true, evaluate: Bool = true) {
        let bigramModel = BigramModel.train(on: names, indexer: indexer)
        let bgResults = bigramModel.predict(20)
        print(bgResults)
        bigramModel.plotFrequencies()
        bigramModel.evaluate(on: names)
    }
    
    static func bigramNeuralNetwork(names: [String], indexer: Indexer, wrapperToken: String) {
        let (x, y) = BigramNeural.createInputsOutputs(from: names, indexer: indexer, wrapperToken: wrapperToken)
        let model = BigramNeural(dimension: 27)
        model.train(inputs: x, outputs: y, learningRate: 50, epochSize: 200)
        let mlpResults = model.sample(20, indexer: indexer)
        print(mlpResults)
    }
}
