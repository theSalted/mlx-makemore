//
//  BatchNorm1d.swift
//  
//
//  Created by Yuhao Chen on 6/29/24.
//
import MLX
import MLXNN

class BatchNorm1d: Module, UnaryLayer {
    let gamma: MLXArray
    let beta: MLXArray
    
    var runningMean: MLXArray
    var runningVariance: MLXArray
    
    let eps: Float
    let momentum: Float
    
    init(_ featureCount: Int, eps: Float = 1e-5, momentum: Float = 0.1) {
        self.eps = eps
        self.momentum = momentum
        // parameters
        self.gamma = MLXArray.zeros([featureCount])
        self.beta = MLXArray.ones([featureCount])
        // buffers
        self.runningMean =  MLXArray.zeros([featureCount])
        self.runningVariance = MLXArray.ones([featureCount])
    }
    
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var xMean: MLXArray
        var xVariance: MLXArray
        if training {
            xMean = x.mean(axis: 0, keepDims: true)
            xVariance = x.variance(axis: 0, keepDims: true)
        } else {
            xMean = runningMean
            xVariance = runningVariance
        }
        
        let xHat = (x - xMean) / sqrt(xVariance + eps)
        let out = gamma * xHat + beta
        
        if training {
            runningMean = stopGradient((1 - momentum) * runningMean + momentum * xMean)
            runningVariance = stopGradient((1 - momentum) * runningVariance + momentum * xVariance)
        }
        
        return out
    } 
}
