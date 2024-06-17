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
        let key = MLXRandom.key(1234)
        
        let bigramModel = BigramModel.train(on: names, indexer: indexer)
//        bigramModel.plotFrequencies()
        let bgResults = bigramModel.predict(20, key: key)
        print(bgResults)
//        bigramModel.evaluate(on: names)

        // MLP
        // Training set (x, y)
        let (x, y) = SimpleMLP.createInputsOutputs(from: names, indexer: indexer, wrapperToken: wrapperToken)
        let model = SimpleMLP(key: key)
        model.train(inputs: x, outputs: y, learningRate: 10, epochSize: 200)
        let mlpResults = model.sample(20, indexer: indexer, key: key)
        print(mlpResults)
    }
}
