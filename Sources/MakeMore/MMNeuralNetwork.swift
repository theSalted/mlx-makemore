//
//  File.swift
//  
//
//  Created by Yuhao Chen on 6/18/24.
//

import MLX

protocol MMNeuralNetwork {}

extension MMNeuralNetwork {
    static func createInputsOutputs(
        from words: [String],
        indexer: Indexer,
        wrapperToken: String
    ) -> (inputs: MLXArray, outputs: MLXArray) {
        return createInputsOutputs(from: words, indexer: indexer, openingToken: wrapperToken, closingToken: wrapperToken)
    }

    static func createInputsOutputs(
        from words: [String],
        indexer: Indexer,
        openingToken: String,
        closingToken: String
    ) -> (inputs: MLXArray, outputs: MLXArray) {
        var inputs: [Int] = []
        var outputs: [Int] = []
        
        for word in words {
            var characters = Array(word).map { String($0) }
            characters = [openingToken] + characters + [closingToken]
            
            for (character1, character2) in zip(characters.dropLast(), characters.dropFirst()) {
                guard let index1 = indexer.tokenToIndexLookup[character1],
                      let index2 = indexer.tokenToIndexLookup[character2] else {
                    fatalError("Could not find index to character while evaluating model")
                }
                inputs.append(index1)
                outputs.append(index2)
            }
        }
        
        return (MLXArray(inputs), MLXArray(outputs))
    }
}
