//
//  Indexer.swift
//  
//
//  Created by Yuhao Chen on 6/17/24.
//


/// A index based character level tokenizer
struct Indexer {
    let tokens: [String]
    let tokenToIndexLookup: [String: Int]
    let indexToTokenLookup: [Int: String]
    
    init(tokens: [String], tokenToIndexLookup: [String : Int]) {
        self.tokens = tokens
        self.tokenToIndexLookup = tokenToIndexLookup
        self.indexToTokenLookup =  Dictionary(
            uniqueKeysWithValues: tokenToIndexLookup.map { ($0.value, $0.key) })
    }
    
    static func create(
        from words: [String],
        wrapperToken: String = "."
    ) -> Indexer {
        let tokens = [wrapperToken] + Array(Set(words.joined())).sorted().map { String($0)}
        let tokenToIndexLookup: [String: Int] = Dictionary(
            uniqueKeysWithValues: tokens.enumerated().map { ($0.element, $0.offset) })
        return Indexer(tokens: tokens, tokenToIndexLookup: tokenToIndexLookup)
    }
    
    static func create(
        from words: [String],
        openingToken: String,
        closingToken: String
    ) -> Indexer {
        let tokens = [openingToken] + Array(Set(words.joined())).sorted().map { String($0)} + [closingToken]
        let tokenToIndexLookup: [String: Int] = Dictionary(
            uniqueKeysWithValues: tokens.enumerated().map { ($0.element, $0.offset) })
        
        return Indexer(tokens: tokens, tokenToIndexLookup: tokenToIndexLookup)
    }
    
    static func create(
        from words: [String]
    ) -> Indexer {
        let tokens = Array(Set(words.joined())).sorted().map { String($0)}
        let tokenToIndexLookup: [String: Int] = Dictionary(
            uniqueKeysWithValues: tokens.enumerated().map { ($0.element, $0.offset) })
        
        return Indexer(tokens: tokens, tokenToIndexLookup: tokenToIndexLookup)
    }
}
