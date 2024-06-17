//
//  File.swift
//  
//
//  Created by Yuhao Chen on 6/17/24.
//

import Foundation
import MLX
import MLXRandom
import Charts
import SwiftUI

// Function to handle MLXArray input
func oneHot(_ tensor: MLXArray, numberOfClasses: Int? = nil) -> MLXArray {
    guard tensor.shape.count == 1 else {
            fatalError("Input tensor must be 1-dimensional")
    }
    let dtype = tensor.dtype
    let indices = tensor.asArray(Int.self)
    let inferredClasses = numberOfClasses ?? (indices.max() ?? -1) + 1
    return oneHot(indices, numberOfClasses: inferredClasses, dtype: dtype)
}

// Function to handle [Int] input and perform the actual one-hot encoding
func oneHot(_ indices: [Int], numberOfClasses: Int, dtype: DType) -> MLXArray {
    let numElements = indices.count
    var oneHotEncoded = [Float](repeating: 0, count: numElements * numberOfClasses)
    
    for (i, index) in indices.enumerated() {
        oneHotEncoded[i * numberOfClasses + index] = 1.0
    }
    
    return MLXArray(oneHotEncoded, [numElements, numberOfClasses]).asType(dtype)
}

func searchSorted(_ sortedSequence: MLXArray, values: MLXArray) -> MLXArray {
    let indices = MLXArray.zeros(like: values) // Ensure the size matches the `values` array
    
    for i in 0..<values.size { // Correct range for `values` array
        var found = false
        for j in 0..<sortedSequence.size { // Correct range for `sortedSequence` array
            if (values[i] .< sortedSequence[j]).all().item() {
                indices[i] = MLXArray(j)
                found = true
                break
            }
        }
        if !found {
            indices[i] = MLXArray(sortedSequence.size - 1)
        }
    }
    
    return indices
}

func multinomial(probability: MLXArray, numberOfSamples: Int, replacement: Bool = true, key: MLXArray? = nil) -> MLXArray {
    let probSum = probability.sum()
    let normalizedProbs = probability / probSum
    
    let cumulativeProbs = cumsum(normalizedProbs)
    
    let randomSamples = MLXRandom.uniform(low: 0.0, high: 1.0, [numberOfSamples], key: key)
    
    let samples = searchSorted(cumulativeProbs, values: randomSamples)
    
    return samples
}






@MainActor
func plot(_ chart: Chart<some ChartContent>,
          name: String,
          minWidth: CGFloat? = 2000,
          minHeight: CGFloat? = 2000) {
    plot(chart
        .chartYAxis {
            AxisMarks(position: .automatic)
        }
        .chartXAxis {
            AxisMarks(position: .automatic)
        }
        .padding(40)
        .frame(minWidth: 2000, minHeight: 2000)
        .background(Color.white), name: name)
}

@MainActor
func plot(_ content: some View, name: String) {
    let renderer = ImageRenderer(content: content)
    
    guard let chartImage = renderer.nsImage else {
        fatalError("Can't render to NSImage")
    }
    
    let fileURL = URL(filePath: "./\(name).png")
    
    guard let tiffData = chartImage.tiffRepresentation,
       let bitmap = NSBitmapImageRep(data: tiffData),
       let pngData = bitmap.representation(using: .png, properties: [:]) else {
        fatalError("Data conversion error")
    }
    
    do {
        try pngData.write(to: fileURL)
        print("Chart image saved successfully at \(fileURL.path)")
    } catch {
        fatalError("Failed to save chart image: \(error)")
    }
}
