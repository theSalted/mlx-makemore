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
        let bigramModel = BigramModel.train(on: names, indexer: indexer)
        let bgResults = bigramModel.predict(20)
        print(bgResults)
        /*
        bigramModel.plotFrequencies()
        bigramModel.evaluate(on: names)
        */

        // MARK: SimpleMLP (Single Layer Bigram MLP)
        // Training set (x, y)
        let (x, y) = SimpleMLP.createInputsOutputs(from: names, indexer: indexer, wrapperToken: wrapperToken)
        let model = SimpleMLP(dimension: 27)
        model.train(inputs: x, outputs: y, learningRate: 50, epochSize: 200)
        let mlpResults = model.sample(20, indexer: indexer)
        print(mlpResults)
    }
}
