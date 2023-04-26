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
        let data = try Data(contentsOf: url)
        var merges = [TokenPair: Int]()
        var index = 0
        var line = [UInt8]()
        for byte in data {
            if byte == UInt8(ascii: "\n") {
                if let pair = try parseMergesLine(line, index: index) {
                    merges[pair] = index
                }
                line.removeAll(keepingCapacity: true)
                index += 1
            } else {
                line.append(byte)
            }
        }

        return merges
    }

    static func parseMergesLine(_ line: [UInt8], index: Int) throws -> TokenPair? {
        if line.isEmpty || line.first == UInt8(ascii: "#") {
            return nil
        }
        let pair = line.split(separator: UInt8(ascii: " "))
        if pair.count != 2 {
            throw FileReadError.invalidMergeFileLine(index + 1)
        }
        return TokenPair( String(bytes: pair[0]), String(bytes: pair[1]))
    }
}

extension String {
    init(bytes: some Collection<UInt8>) {
        self.init(unsafeUninitializedCapacity: bytes.count) { pointer in
            _ = pointer.initialize(fromContentsOf: bytes)
            return bytes.count
        }
    }
}
