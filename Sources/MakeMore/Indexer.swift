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
    
    let openingToken: String
    let closingToken: String
    
    init(tokens: [String], 
         tokenToIndexLookup: [String : Int],
         openingToken: String,
         closingToken: String) {
        self.tokens = tokens
        self.tokenToIndexLookup = tokenToIndexLookup
        self.indexToTokenLookup =  Dictionary(
            uniqueKeysWithValues: tokenToIndexLookup.map { ($0.value, $0.key) })
        self.openingToken = openingToken
        self.closingToken = closingToken
    }
    
    static func create(
        from words: [String],
        wrapperToken: String = "."
    ) -> Indexer {
        return create(from: words, openingToken: wrapperToken, closingToken: wrapperToken)
    }
    
    static func create(
        from words: [String],
        openingToken: String,
        closingToken: String
    ) -> Indexer {
        let tokens = openingToken == closingToken ?
            [openingToken] + Array(Set(words.joined())).sorted().map { String($0)} :
            [openingToken] + Array(Set(words.joined())).sorted().map { String($0)} + [closingToken]
        let tokenToIndexLookup: [String: Int] = Dictionary(
            uniqueKeysWithValues: tokens.enumerated().map { ($0.element, $0.offset) })
        
        return Indexer(tokens: tokens, tokenToIndexLookup: tokenToIndexLookup, openingToken: openingToken, closingToken: closingToken)
    }
}
