// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2022 Apple Inc. All Rights Reserved.

import Foundation

@available(iOS 16.2, macOS 13.1, *)
extension BPETokenizer {
    enum FileReadError: Error {
        case invalidMergeFileLine(Int)
    }

    /// Read vocab.json file at URL into a dictionary mapping a String to its Int token id
    static func readVocabulary(url: URL) throws -> [String: Int] {
        let content = try Data(contentsOf: url)
        return try JSONDecoder().decode([String: Int].self, from: content)
    }

    /// Read merges.txt file at URL into a dictionary mapping bigrams to the line number/rank/priority
    static func readMerges(url: URL) throws -> [TokenPair: Int] {
        let content = try String(contentsOf: url)
        let lines = content.split(separator: "\n")

        let merges: [(TokenPair, Int)] = try lines.enumerated().compactMap { (index, line) in
            if line.hasPrefix("#") {
                return nil
            }
            let pair = line.split(separator: " ")
            if pair.count != 2 {
                throw FileReadError.invalidMergeFileLine(index+1)
            }
            return (TokenPair(String(pair[0]), String(pair[1])),index)
        }
        return [TokenPair : Int](uniqueKeysWithValues: merges)
    }
}
