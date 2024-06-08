// The Swift Programming Language
// https://docs.swift.org/swift-book
// The swift-mlx Documentation
// https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx

import MLX

@main
struct Template {
    static func main() {
        // A 3x3 Matrix
        let matrixA = MLXArray([
            1, 2, 3,
            4, 5, 6,
            7, 8, 9
        ], [3, 3])
        
        print(matrixA)
    }
}
