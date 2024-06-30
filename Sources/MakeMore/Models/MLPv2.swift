//
//  MLPv2.swift
//  
//
//  Created by Yuhao Chen on 6/30/24.
//

/* Yes It's called V2, deal with it! Naming it V1 is way too confusing. Unless I rename ``MLP`` to ``MLPv0`` which I am just not gonna do. Technically the existing implementation of ``MLP`` received some optimization, which meant its slightly different from Bengio. So there can be a V0 in our heart.
 */

import MLX
import MLXNN

/// MLXNN Style Multi-Layer Perceptron (MLP) Neural Network (Version 2)
///
/// - Warning: V2 is currently not working as expected, it doesn't' really converge. You should still use the original one
class MLPv2: MLP {
    // TODO: This model not working and poorly implemented. Come back later and exam what's going on her, 
    @ModuleInfo var linear1: CustomLinear
    @ModuleInfo var linear2: CustomLinear
    @ModuleInfo var linear3: CustomLinear
    @ModuleInfo var linear4: CustomLinear
    @ModuleInfo var linear5: CustomLinear
    @ModuleInfo var linear6: CustomLinear
    
    @ModuleInfo var batchNorm1: BatchNorm1d
    @ModuleInfo var batchNorm2: BatchNorm1d
    @ModuleInfo var batchNorm3: BatchNorm1d
    @ModuleInfo var batchNorm4: BatchNorm1d
    @ModuleInfo var batchNorm5: BatchNorm1d
    @ModuleInfo var batchNorm6: BatchNorm1d
    
    @ModuleInfo var tanh1: CustomTanh
    @ModuleInfo var tanh2: CustomTanh
    @ModuleInfo var tanh3: CustomTanh
    @ModuleInfo var tanh4: CustomTanh
    @ModuleInfo var tanh5: CustomTanh
    
    
    override var name: String { "MLPv2" }
    
    override init(
        trainingDataSize: Int,
        hiddenLayerSize: Int = 200,
        embeddingDimension: Int = 2,
        blockSize: Int = 3,
        outputSize: Int = 27
    ) {
        linear1 = CustomLinear(embeddingDimension * blockSize, hiddenLayerSize)
        linear2 = CustomLinear(hiddenLayerSize, hiddenLayerSize)
        linear3 = CustomLinear(hiddenLayerSize, hiddenLayerSize)
        linear4 = CustomLinear(hiddenLayerSize, hiddenLayerSize)
        linear5 = CustomLinear(hiddenLayerSize, hiddenLayerSize)
        linear6 = CustomLinear(hiddenLayerSize, outputSize)
        
        batchNorm1 = BatchNorm1d(hiddenLayerSize)
        batchNorm2 = BatchNorm1d(hiddenLayerSize)
        batchNorm3 = BatchNorm1d(hiddenLayerSize)
        batchNorm4 = BatchNorm1d(hiddenLayerSize)
        batchNorm5 = BatchNorm1d(hiddenLayerSize)
        batchNorm6 = BatchNorm1d(outputSize)
        
        tanh1 = CustomTanh()
        tanh2 = CustomTanh()
        tanh3 = CustomTanh()
        tanh4 = CustomTanh()
        tanh5 = CustomTanh()
        
        super.init(
            trainingDataSize: trainingDataSize,
            hiddenLayerSize: hiddenLayerSize,
            embeddingDimension: embeddingDimension,
            blockSize: blockSize,
            outputSize: outputSize)
        for layer in [linear1, linear2, linear3, linear4, linear5, linear6] {
            layer.weight *= 5/3
        }
        batchNorm6.gamma *= 0.1
    }
    
    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let embedding = C[x]
        let embeddingConcatenate = embedding.reshaped(embedding.shape[0], embeddingDimension * 3)
        var x = embeddingConcatenate
        x = linear1(x)
        x = batchNorm1(x)
        x = tanh1(x)
        
        x = linear2(x)
        x = batchNorm2(x)
        x = tanh2(x)
        
        x = linear3(x)
        x = batchNorm3(x)
        x = tanh3(x)
        
        x = linear4(x)
        x = batchNorm4(x)
        x = tanh4(x)
        
        x = linear5(x)
        x = batchNorm5(x)
        x = tanh5(x)
        
        x = linear6(x)
        x = batchNorm6(x)
        return x
    }
}
