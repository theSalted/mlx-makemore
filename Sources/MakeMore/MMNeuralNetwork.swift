//
//  File.swift
//  
//
//  Created by Yuhao Chen on 6/18/24.
//

import MLX

protocol MMNeuralNetwork {}

extension MMNeuralNetwork {
    static func formatInputsOutputs(
        from words: [String],
        indexer: Indexer,
        wrapperToken: String,
        blockSize: Int = 1,
        printDebug: Bool = false
    ) -> (inputs: MLXArray, outputs: MLXArray) {
        return createInputsOutputs(
            from: words,
            indexer: indexer, 
            openingToken: wrapperToken,
            closingToken: wrapperToken,
            blockSize: blockSize,
            printDebug: printDebug
        )
    }

    static func createInputsOutputs(
        from words: [String],
        indexer: Indexer,
        openingToken: String,
        closingToken: String,
        blockSize: Int,
        printDebug: Bool = false
    ) -> (inputs: MLXArray, outputs: MLXArray) {
        var inputs: [Int] = []
        var outputs: [Int] = []
        for word in words {
            let characters = Array(word).map { String($0) }
            var context: [Int] = Array(repeating: 0, count: blockSize)
            if printDebug {
                print(word)
            }
            for character in characters + [closingToken] {
                guard let index = indexer.tokenToIndexLookup[character] else {
                    fatalError("Could not find index to character while evaluating model")
                }
                inputs.append(contentsOf: context)
                outputs.append(index)
                if printDebug {
                    print("\(context.map({ index in indexer.indexToTokenLookup[index]!}).joined()) -> \(indexer.indexToTokenLookup[index]!)")
                }
                context.removeFirst()
                context += [index]
            }
        }
        return (MLXArray(inputs, 
                         blockSize == 1 ?
                            nil : [inputs.count / blockSize, blockSize]),
                MLXArray(outputs))
    }
}
