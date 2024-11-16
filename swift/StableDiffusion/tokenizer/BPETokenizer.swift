//
//  BPETokenizer.swift
//
//  For licensing see accompanying LICENSE.md file.
//  Copyright (C) 2022 Apple Inc. All Rights Reserved.
//

import Foundation

/// A tokenizer based on byte pair encoding.
@available(iOS 16.2, macOS 13.1, *)
public struct BPETokenizer {
    /// A dictionary that maps pairs of tokens to the rank/order of the merge.
    let merges: [TokenPair: Int]

    /// A dictionary from tokens to identifiers.
    let vocabulary: [String: Int]

    /// The token used for padding.
    let padToken: String

    /// The start token.
    let startToken: String = "<|startoftext|>"

    /// The end token.
    let endToken: String = "<|endoftext|>"

    /// The unknown token.
    let unknownToken: String = "<|endoftext|>"

    /// The ID of the unknown token, or 0 by default.
    var unknownTokenID: Int {
        vocabulary[unknownToken, default: 0]
    }

    /// Creates a tokenizer.
    ///
    /// - Parameters:
    ///   - merges: A dictionary that maps pairs of tokens to the rank/order of the merge.
    ///   - vocabulary: A dictionary from tokens to identifiers.
    public init(merges: [TokenPair: Int], vocabulary: [String: Int], padToken: String = "<|endoftext|>") {
        self.merges = merges
        self.vocabulary = vocabulary
        self.padToken = padToken
    }

    /// Creates a tokenizer by loading merges and vocabulary from URLs.
    ///
    /// - Parameters:
    ///   - mergesURL: The URL of a text file containing merges.
    ///   - vocabularyURL: The URL of a JSON file containing the vocabulary.
    public init(mergesAt mergesURL: URL, vocabularyAt vocabularyURL: URL, padToken: String = "<|endoftext|>") throws {
        // Improved error handling for file reading
        self.merges = try Self.readMerges(url: mergesURL)
        self.vocabulary = try Self.readVocabulary(url: vocabularyURL)
        self.padToken = padToken
    }

    /// Tokenizes an input string.
    ///
    /// - Parameters:
    ///   - input: A string.
    ///   - minCount: The minimum number of tokens to return.
    /// - Returns: An array of tokens and an array of token identifiers.
    public func tokenize(input: String, minCount: Int? = nil) -> (tokens: [String], tokenIDs: [Int]) {
        var tokens: [String] = [startToken] + encode(input: input) + [endToken]

        // Pad if there was a minimum length specified
        if let minLen = minCount, minLen > tokens.count {
            tokens.append(contentsOf: repeatElement(padToken, count: minLen - tokens.count))
        }

        let ids = tokens.map { vocabulary[$0, default: unknownTokenID] }
        return (tokens: tokens, tokenIDs: ids)
    }

    /// Returns the token identifier for a token.
    public func tokenID(for token: String) -> Int? {
        vocabulary[token]
    }

    /// Returns the token for a token identifier.
    public func token(id: Int) -> String? {
        vocabulary.first(where: { $0.value == id })?.key
    }

    /// Decodes a sequence of tokens into a fully formed string.
    public func decode(tokens: [String]) -> String {
        tokens.joined()
            .replacingOccurrences(of: "</w>", with: " ")
            .replacingOccurrences(of: startToken, with: "")
            .replacingOccurrences(of: endToken, with: "")
    }

    /// Encodes an input string into a sequence of tokens.
    func encode(input: String) -> [String] {
        let normalized = input.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return normalized.split(separator: " ").flatMap { encode(word: $0) }
    }

    /// Encodes a single word into a sequence of tokens.
    func encode(word: Substring) -> [String] {
        var tokens = word.map { String($0) }
        if let last = tokens.indices.last {
            tokens[last] += "</w>"
        }

        while true {
            let pairs = pairs(for: tokens)
            let canMerge = pairs.compactMap { merges[$0] }

            if canMerge.isEmpty {
                break
            }

            // Select the pair with the lowest rank
            let shouldMerge = canMerge.min()!
            tokens = update(tokens, merging: shouldMerge)
        }
        return tokens
    }

    /// Gets the set of adjacent pairs/bigrams from a sequence of tokens.
    func pairs(for tokens: [String]) -> Set<TokenPair> {
        guard tokens.count > 1 else { return [] }
        return Set(zip(tokens, tokens.dropFirst()).map { TokenPair($0.0, $0.1) })
    }

    /// Updates the sequence of tokens by greedily merging instances of a specific bigram.
    func update(_ tokens: [String], merging bigram: TokenPair) -> [String] {
        guard tokens.count > 1 else { return tokens }

        var newTokens = [String]()
        var skipNext = false

        for index in 0..<tokens.count {
            if skipNext {
                skipNext = false
                continue
            }

            if index < tokens.count - 1 && tokens[index] == bigram.first && tokens[index + 1] == bigram.second {
                newTokens.append(bigram.first + bigram.second)
                skipNext = true
            } else {
                newTokens.append(tokens[index])
            }
        }
        return newTokens
    }

    /// Reads merges from a file.
    static func readMerges(url: URL) throws -> [TokenPair: Int] {
        let data = try Data(contentsOf: url)
        let lines = String(data: data, encoding: .utf8)!.split(separator: "\n")
        var merges = [TokenPair: Int]()
        for (index, line) in lines.enumerated() {
            let tokens = line.split(separator: " ")
            if tokens.count == 2 {
                merges[TokenPair(String(tokens[0]), String(tokens[1]))] = index
            }
        }
        return merges
    }

    /// Reads vocabulary from a file.
    static func readVocabulary(url: URL) throws -> [String: Int] {
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode([String: Int].self, from: data)
    }
}

@available(iOS 16.2, macOS 13.1, *)
extension BPETokenizer {
    /// A hashable tuple of strings representing a token pair.
    public struct TokenPair: Hashable {
        let first: String
        let second: String

        init(_ first: String, _ second: String) {
            self.first = first
            self.second = second
        }
    }
}
