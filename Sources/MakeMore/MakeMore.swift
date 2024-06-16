// The Swift Programming Language
// https://docs.swift.org/swift-book
// The swift-mlx Documentation
// https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx

import MLX
import MLXRandom
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

        let bigramModel = BigramModel.train(on: names)
        let results = bigramModel.predict(20)
        print(results)
//        bigramModel.plotFrequencies()
        bigramModel.evaluateLoss(on: names)
    }
}


