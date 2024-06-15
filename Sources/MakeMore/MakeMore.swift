// The Swift Programming Language
// https://docs.swift.org/swift-book
// The swift-mlx Documentation
// https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx

import MLX
import Foundation
import Charts
import SwiftUI

@main
struct MakeMore {
    static func main() {
        // A 3x3 Matrix
        guard
            let namesUrl = Bundle.module.url(forResource: "names", withExtension: "txt"),
            let namesString = try? String(contentsOf: namesUrl, encoding: .utf8)
        else {
            fatalError("Can't extract contents of names.txt")
        }
        
        let names = namesString.split(separator: "\n").map{String($0)}

        Bigram.generateBigrams(words: names)
        
        print("done")
    }
    
    
}

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
    
    @MainActor 
    static func generateBigrams(words: [String]) {
        
        /// A list of all character in all words
        let allCharacters = Array(Set(words.joined())).sorted().map { String($0)}
        
        var characterToIndexLookup: [String: Int] = Dictionary(
            uniqueKeysWithValues: allCharacters.enumerated().map { ($0.element, $0.offset) })
        
        var indexToCharacterLookup: [Int: String] = Dictionary(
            uniqueKeysWithValues: allCharacters.enumerated().map { ($0.offset, $0.element) })
        var characterCount = allCharacters.count
        
        let startCharacter = "<S>"
        characterToIndexLookup[startCharacter] = characterCount
        indexToCharacterLookup[characterCount] = startCharacter
        
        let endCharacter = "<E>"
        characterToIndexLookup[endCharacter] = characterCount + 1
        indexToCharacterLookup[characterCount + 1] = endCharacter
        
        
        characterCount = characterToIndexLookup.count
        print(characterCount)
        
        // Syntax like  `array[1, 1] += 1` currently are not working. Instead, we first map bigram frequency in a dict then mapped to tensor
        let bigramTensor = MLXArray.zeros([28, 28], type: Int.self)
        var bigramFrequency = [Bigram: Int]()
        
        for word in words {
            var characters = Array(word).map { String($0) } // Convert the string to an array of characters
            characters = [startCharacter] + characters + [endCharacter]
            
            // Equivalent to python `for zip(w, w[1:]):` (python auto halt when zip lists of two different sizes.
            for (character1, character2) in zip(characters.dropLast(), characters.dropFirst()) {
                let bigram = Bigram(character1, character2)
                bigramFrequency[bigram] = (bigramFrequency[bigram] ?? 0) + 1
            }
        }
        
        
        for (bigram, count) in bigramFrequency {
            guard let lhsIndex = characterToIndexLookup[bigram.lhs],
                  let rhsIndex = characterToIndexLookup[bigram.rhs] else {
                fatalError("Can't find characters in lookup")
            }
            
            bigramTensor[lhsIndex, rhsIndex] = MLXArray(count)
        }
        
        let length = characterCount - 1
        let chart = Chart() {
            ForEach(0...length, id: \.self) { x in
                ForEach(0...length, id: \.self) { y in
                    let frequency = bigramTensor[x, y].asArray(Int.self)[0]
                    let xCharacter = indexToCharacterLookup[x] ?? "unknown"
                    let yCharacter = indexToCharacterLookup[y] ?? "unknown"
                    RectangleMark(x: .value(yCharacter, y),
                                  y: .value(xCharacter, x),
                                  width: 60,
                                  height: 60
                    )
                    .offset(x: 30, y: -30)
                    .foregroundStyle(by: .value("Frequency", frequency))
                    .annotation(position: .overlay, alignment: .topLeading) {
                        VStack(alignment: .leading) {
                            Text("\(xCharacter)\(yCharacter)").bold().fontDesign(.monospaced)
                            Text("\(frequency)").fontDesign(.monospaced)
                        }.font(.caption)
                        .padding(2)
                    }
                }
                
            }
        }
        
        
        plot(chart, name: "BigramMap")
        
    }
}


@MainActor 
func plot(_ chart: Chart<some ChartContent>, name: String) {
    
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
