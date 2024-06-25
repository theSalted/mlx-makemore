//
//  Bigram.swift
//
//
//  Created by Yuhao Chen on 6/17/24.
//

import Foundation
import MLX
import MLXRandom
import Charts
import SwiftUI
import OSLog

struct Bigram: Hashable {
    let lhs: String
    let rhs: String
    
    
    init(lhs: String, rhs: String) {
        self.lhs = lhs
        self.rhs = rhs
    }
    
    init(_ lhs: String, _ rhs: String) {
        self.lhs = lhs
        self.rhs = rhs
    }
}

struct BigramModel {
    let frequencies: MLXArray
    let indexer: Indexer
    
    private init(
        frequencies: MLXArray,
        indexer: Indexer
    ) {
        self.frequencies = frequencies
        self.indexer = indexer
    }
    
    static func train(
        on words: [String],
        indexer: Indexer
    ) -> BigramModel {
        print("Bigram training...")
        // Syntax like  `array[1, 1] += 1` currently are not working. Instead, we first map bigram frequency in a dict then mapped to tensor
        let tokenCount = indexer.tokens.count
        let frequencies = MLXArray.zeros([tokenCount, tokenCount], type: Int.self) //
        var frequencyLookup = [Bigram: Int]()
        
        for word in words {
            var characters = Array(word).map { String($0) } // Convert the string to an array of characters
            /*characters = [startCharacter] + characters + [endCharacter]*/ // Again uncomment for special token support
            characters = [indexer.openingToken] + characters + [indexer.closingToken]
            
            // Equivalent to python `for zip(w, w[1:]):` (python auto halt when zip lists of two different sizes.
            for (character1, character2) in zip(characters.dropLast(), characters.dropFirst()) {
                let bigram = Bigram(character1, character2)
                frequencyLookup[bigram, default: 0] += 1
            }
        }
        
        for (bigram, count) in frequencyLookup {
            guard let lhsIndex = indexer.tokenToIndexLookup[bigram.lhs],
                  let rhsIndex = indexer.tokenToIndexLookup[bigram.rhs] else {
                fatalError("Can't find characters in lookup")
            }
            
            frequencies[lhsIndex, rhsIndex] = MLXArray(count)
        }
        
        return BigramModel(
            frequencies: frequencies,
            indexer: indexer
        )
    }
    
    func predict(_ count: Int = 0) -> [String] {
        print("Bigram predicting...")
        var results: [String] = []
        
        let probability = frequencies.asType(Float.self)
        probability /= probability.sum(axis: 1, keepDims: true)
        
        for _ in 0...count {
            var index = 0
            var result = ""
            
            while true {
                index = multinomial(probability: probability[index], numberOfSamples: 1).asArray(Int.self)[0]
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
    
    func evaluate(on words: [String]) {
        print("Bigram evaluating...")
        let probability = frequencies.asType(Float.self) + 1
        probability /= probability.sum(axis: 1, keepDims: true)
        
        var averageNegativeLogLikelihood = MLXArray(1)
        
        var numberOfEval = 0
        
        for word in words {
            var characters = Array(word).map { String($0) } // Convert the string to an array of characters
            /*characters = [startCharacter] + characters + [endCharacter]*/ // Again uncomment for special token support
            characters = [indexer.openingToken] + characters + [indexer.closingToken]
            
            // Equivalent to python `for zip(w, w[1:]):` (python auto halt when zip lists of two different sizes.
            for (character1, character2) in zip(characters.dropLast(), characters.dropFirst()) {
                guard let index1 = indexer.tokenToIndexLookup[character1],
                      let index2 = indexer.tokenToIndexLookup[character2] else {
                    print("Could not find index to character while evaluating model")
                    return
                }
                var likelihood = probability[index1, index2]
                likelihood = log(likelihood)
                averageNegativeLogLikelihood += likelihood
                numberOfEval += 1
            }
            eval(averageNegativeLogLikelihood)
        }
        
        averageNegativeLogLikelihood = -averageNegativeLogLikelihood
        averageNegativeLogLikelihood /= MLXArray(numberOfEval)
        print("Loss (smoothed average negative log likelihood): \(averageNegativeLogLikelihood.asArray(Float.self)[0])")
    }
    
    
    @MainActor
    func plotFrequencies() {
        plotHeatMap(frequencies, name: "BigramFrequencies")
    }
    
    @MainActor
    private func plotHeatMap(_ tensor: MLXArray, name: String) {
        print("Bigram plotting...")
        let chart = Chart() {
            let length = indexer.tokenToIndexLookup.count - 1
            ForEach(0...length, id: \.self) { x in
                ForEach(0...length, id: \.self) { y in
                    let value = tensor[x, y].asArray(Int.self)[0]
                    let xCharacter = indexer.indexToTokenLookup[x] ?? "unknown"
                    let yCharacter = indexer.indexToTokenLookup[y] ?? "unknown"
                    RectangleMark(x: .value(yCharacter, y),
                                  y: .value(xCharacter, x),
                                  width: 60,
                                  height: 60
                    )
                    .offset(x: 30, y: -30)
                    .foregroundStyle(by: .value("Value", value))
                    .annotation(position: .overlay, alignment: .topLeading) {
                        VStack(alignment: .leading) {
                            Text("\(xCharacter)\(yCharacter)").bold().fontDesign(.monospaced)
                            Text("\(value)").fontDesign(.monospaced)
                        }
                        .padding(2)
                    }
                }
                
            }
        }
        plot(chart, name: name, minWidth: 2000, minHeight: 2000)
    }
}
