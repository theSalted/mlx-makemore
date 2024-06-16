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

        let bigramModel = BigramModel.train(words: names)
//        bigramModel.plotFrequencies()
        let results = bigramModel.predict(20)
        print(results)
//        Bigram.plotFrequency(frequencyTensor, lookup: indexToCharacterLookup)
//                
//        var probability = bigramTensor[0].asType(Float.self)
//        
////        print(probability)
//        
//        let key = MLXRandom.key(2147482647)
////        var probability = MLXRandom.uniform(low: 0.0, high: 1.0, [3], key: key)
//        probability = probability / probability.sum()
//        print(probability)
//        let ix = multiNomial(probability: probability, numberOfSamples: 100, key: key)
//        let index = ix.asArray(Int.self)
//        let c = index.map { indexToCharacterLookup[$0]! }
//        print(c)
//        
//        // Calculate frequency
//        var frequencyDict: [String: Int] = [:]
//
//        for char in c {
//            frequencyDict[char, default: 0] += 1
//        }
//
//        // Print frequency
//        for (char, count) in frequencyDict {
//            print("\(char): \(count)")
//        }
//        
//        print("done")
    }
}


