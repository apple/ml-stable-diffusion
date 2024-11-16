//
//  BPETokenizer.swift
//
//  Created by Apple ML Team on DATE.
//  Implements Byte Pair Encoding (BPE) tokenizer.
//

import Foundation

@available(iOS 16.2, macOS 13.1, *)
public struct BPETokenizer {
    // Dictionary for merges and vocabulary.
    let merges: [TokenPair : Int]
    let vocabulary: [String: Int]
    
    // Tokens used for padding, start, end, and unknown sequences.
    let padToken: String
    let startToken: String = ""
    let endToken: String = ""
    let unknownToken: String = ""

    // Computed property to get the ID for the unknown token.
    var unknownTokenID: Int {
        vocabulary[unknownToken, default: 0]
    }

    // Initializes the tokenizer with preloaded merges and vocabulary.
    public init(merges: [TokenPair: Int], vocabulary: [String: Int], padToken: String = "") {
        self.merges = merges
        self.vocabulary = vocabulary
        self.padToken = padToken
    }

    // Initializes the tokenizer by reading merges and vocabulary from URLs.
    public init(mergesAt mergesURL: URL, vocabularyAt vocabularyURL: URL, padToken: String = "") throws {
        self.merges = try Self.readMerges(url: mergesURL)
        self.vocabulary = try Self.readVocabulary(url: vocabularyURL)
        self.padToken = padToken
    }

    // Tokenizes the input string into tokens and their corresponding IDs.
    public func tokenize(input: String, minCount: Int? = nil) -> (tokens: [String], tokenIDs: [Int]) {
        var tokens: [String] = [startToken] + encode(input: input) + [endToken]
        
        // Pad tokens to ensure minimum count.
        if let minLen = minCount, minLen > tokens.count {
            tokens.append(contentsOf: repeatElement(padToken, count: minLen - tokens.count))
        }
        let ids = tokens.map { vocabulary[$0, default: unknownTokenID] }
        return (tokens, ids)
    }

    // Returns the token ID for a given token string.
    public func tokenID(for token: String) -> Int? {
        vocabulary[token]
    }

    // Returns the token string for a given ID.
    public func token(id: Int) -> String? {
        vocabulary.first { $0.value == id }?.key
    }

    // Decodes an array of tokens back into a string.
    public func decode(tokens: [String]) -> String {
        tokens.joined()
            .replacingOccurrences(of: "</w>", with: " ")
            .replacingOccurrences(of: startToken, with: "")
            .replacingOccurrences(of: endToken, with: "")
    }

    // Encodes an input string into an array of tokens.
    func encode(input: String) -> [String] {
        // Normalize input by trimming whitespace and converting to lowercase.
        let normalized = input.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return normalized.split(separator: " ").flatMap { encode(word: $0) }
    }

    // Encodes a single word into an array of sub-tokens.
    func encode(word: Substring) -> [String] {
        var tokens = word.map { String($0) }
        if let last = tokens.indices.last {
            // Add end-of-word marker to the last token.
            tokens[last] += "</w>"
        }

        // Iteratively merge token pairs until no more pairs can be merged.
        while true {
            let pairs = self.pairs(for: tokens)
            
            // Fix: Safeguard against empty merges by ensuring pairs exist.
            guard let shouldMerge = pairs.compactMap({ merges[$0] != nil ? $0 : nil }).min(by: { merges[$0]! < merges[$1]! }) else {
                break
            }
            tokens = update(tokens, merging: shouldMerge)
        }
        return tokens
    }

    // Returns the set of token pairs in the input token array.
    func pairs(for tokens: [String]) -> Set<TokenPair> {
        guard tokens.count > 1 else {
            return Set()
        }
        return Set(zip(tokens, tokens.dropFirst()).map { TokenPair($0, $1) })
    }

    // Updates the token array by merging the specified bigram.
    func update(_ tokens: [String], merging bigram: TokenPair) -> [String] {
        guard tokens.count > 1 else {
            return tokens
        }
        var newTokens = [String]()
        var index = 0
        while index < tokens.count {
            let remainingTokens = tokens[index...]
            if let startMatchIndex = remainingTokens.firstIndex(of: bigram.first),
               startMatchIndex < tokens.count - 1, tokens[startMatchIndex + 1] == bigram.second {
                // Append merged bigram and skip the next token.
                newTokens.append(contentsOf: tokens[index..<startMatchIndex])
                newTokens.append(bigram.first + bigram.second)
                index = startMatchIndex + 2
            } else {
                // Append current token and move to the next.
                newTokens.append(tokens[index])
                index += 1
            }
        }
        return newTokens
    }
}

@available(iOS 16.2, macOS 13.1, *)
extension BPETokenizer {
    // Represents a pair of tokens to be merged.
    public struct TokenPair: Hashable {
        let first: String
        let second: String

        init(_ first: String, _ second: String) {
            self.first = first
            self.second = second
        }
    }

    // Reads merge rules from a file URL.
    static func readMerges(url: URL) throws -> [TokenPair: Int] {
        let data = try Data(contentsOf: url)
        let string = String(data: data, encoding: .utf8)!
        let lines = string.split(separator: "\n")
        var merges = [TokenPair: Int]()
        for (index, line) in lines.enumerated() {
            let tokens = line.split(separator: " ")
            if tokens.count == 2 {
                let pair = TokenPair(String(tokens[0]), String(tokens[1]))
                merges[pair] = index
            }
        }
        return merges
    }

    // Reads vocabulary from a file URL.
    static func readVocabulary(url: URL) throws -> [String: Int] {
        let data = try Data(contentsOf: url)
        let vocabulary = try JSONDecoder().decode([String: Int].self, from: data)
        return vocabulary
    }
}
