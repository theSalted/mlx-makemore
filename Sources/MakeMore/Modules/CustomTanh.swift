//
//  CustomTanh.swift
//
//
//  Created by Yuhao Chen on 6/30/24.
//

import MLX
import MLXNN

class CustomTanh: Module, UnaryLayer {
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return tanh(x)
    }
}

