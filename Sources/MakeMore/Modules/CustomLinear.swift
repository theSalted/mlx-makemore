//
//  Linear.swift
//  
//
//  Created by Yuhao Chen on 6/29/24.
//

import MLX
import MLXNN
import MLXRandom

class CustomLinear: Module, UnaryLayer {
    let weight: MLXArray
    let bias: MLXArray?
    
    init(_ inputDimensions: Int, _ outputDimensions: Int, bias: Bool = true) {
        self.weight = MLXRandom.uniform(low: 0.0, high: 1.0, [inputDimensions, outputDimensions])
        self.bias = bias ? MLXArray.zeros([outputDimensions]) : nil
    }
    
    init(_ inputDimensions: Int, _ outputDimensions: Int, bias: MLXArray) {
        self.weight = MLXRandom.uniform(low: 0.0, high: 1.0, [inputDimensions, outputDimensions])
        self.bias = bias
    }
    
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let output = matmul(x, weight)
        if let bias {
            output += bias
        }
        return output
    }
}

