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
    let trainingWords: [String]
    let frequencies: MLXArray
    let characterToIndexLookup: [String: Int]
    let indexToCharacterLookup: [Int: String]
    
    private init(
        trainingWords: [String],
        frequencies: MLXArray,
        characterToIndexLookup: [String : Int],
        indexToCharacterLookup: [Int : String]
    ) {
        self.trainingWords = trainingWords
        self.frequencies = frequencies
        self.characterToIndexLookup = characterToIndexLookup
        self.indexToCharacterLookup = indexToCharacterLookup
    }
    
    static func train(words: [String]) -> BigramModel {
        print("training...")
        /// A list of all character in all words
        let allCharacters = Array(Set(words.joined())).sorted().map { String($0)}
        
        var characterToIndexLookup: [String: Int] = Dictionary(
            uniqueKeysWithValues: allCharacters.enumerated().map { ($0.element, $0.offset + 1 /* normally should be 0 */) })
        
        let specialToken = "." // This approach is more optimize for this use case
        characterToIndexLookup[specialToken] = 0
        
        // Uncomment for proper special token support. allCharacters should start 0 instead of one here
        /**let startCharacter = "<S>"
        characterToIndexLookup[startCharacter] = characterCount
        
        let endCharacter = "<E>"
        characterToIndexLookup[endCharacter] = characterCount + 1**/
        
        let indexToCharacterLookup: [Int: String] = Dictionary(
            uniqueKeysWithValues: characterToIndexLookup.map { ($0.value, $0.key) })
        var characterCount = allCharacters.count
        
        characterCount = characterToIndexLookup.count
        
        // Syntax like  `array[1, 1] += 1` currently are not working. Instead, we first map bigram frequency in a dict then mapped to tensor
        let frequencies = MLXArray.zeros([characterCount, characterCount], type: Int.self) //
        var bigramFrequency = [Bigram: Int]()
        
        for word in words {
            var characters = Array(word).map { String($0) } // Convert the string to an array of characters
            /*characters = [startCharacter] + characters + [endCharacter]*/ // Again uncomment for special token support
            characters = [specialToken] + characters + [specialToken]
            
            // Equivalent to python `for zip(w, w[1:]):` (python auto halt when zip lists of two different sizes.
            for (character1, character2) in zip(characters.dropLast(), characters.dropFirst()) {
                let bigram = Bigram(character1, character2)
                bigramFrequency[bigram, default: 0] += 1
            }
        }
        
        
        for (bigram, count) in bigramFrequency {
            guard let lhsIndex = characterToIndexLookup[bigram.lhs],
                  let rhsIndex = characterToIndexLookup[bigram.rhs] else {
                fatalError("Can't find characters in lookup")
            }
            
            frequencies[lhsIndex, rhsIndex] = MLXArray(count)
        }
        
        return BigramModel(
            trainingWords: words,
            frequencies: frequencies,
            characterToIndexLookup: characterToIndexLookup,
            indexToCharacterLookup: indexToCharacterLookup)
    }
    
    func predict(_ count: Int = 0, key: MLXArray? = nil) -> [String] {
        print("prepping...")
        var results: [String] = []
        
        let probability = frequencies.asType(Float.self)
        probability /= probability.sum(axis: 1, keepDims: true)
        
        print("predicting...")
        for _ in 0...count {
            var index = 0
            var result = ""
            
            while true {
                index = multinomial(probability: probability[index], numberOfSamples: 1).asArray(Int.self)[0]
                let predictedCharacter = indexToCharacterLookup[index]
                result += predictedCharacter ?? ""
                if index == 0 { break }
            }
            
            results.append(result)
        }
        
        return results
    }
    
    @MainActor
    func plotFrequencies() {
        plotHeatMap(frequencies, name: "BigramFrequencies")
    }
    
    @MainActor
    private func plotHeatMap(_ tensor: MLXArray, name: String) {
        print("plotting...")
        let chart = Chart() {
            let length = indexToCharacterLookup.count - 1
            ForEach(0...length, id: \.self) { x in
                ForEach(0...length, id: \.self) { y in
                    let value = tensor[x, y].asArray(Int.self)[0]
                    let xCharacter = indexToCharacterLookup[x] ?? "unknown"
                    let yCharacter = indexToCharacterLookup[y] ?? "unknown"
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
