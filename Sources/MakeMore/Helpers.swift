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
