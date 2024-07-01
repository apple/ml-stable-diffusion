import Foundation

@available(iOS 16.2, macOS 13.1, *)
public struct BPETokenizer {
    let merges: [TokenPair : Int]
    let vocabulary: [String: Int]
    let padToken: String
    let startToken: String = ""
    let endToken: String = ""
    let unknownToken: String = ""

    var unknownTokenID: Int {
        vocabulary[unknownToken, default: 0]
    }

    public init(merges: [TokenPair: Int], vocabulary: [String: Int], padToken: String = "") {
        self.merges = merges
        self.vocabulary = vocabulary
        self.padToken = padToken
    }

    public init(mergesAt mergesURL: URL, vocabularyAt vocabularyURL: URL, padToken: String = "") throws {
        self.merges = try Self.readMerges(url: mergesURL)
        self.vocabulary = try Self.readVocabulary(url: vocabularyURL)
        self.padToken = padToken
    }

    public func tokenize(input: String, minCount: Int? = nil) -> (tokens: [String], tokenIDs: [Int]) {
        var tokens: [String] = [startToken] + encode(input: input) + [endToken]
        if let minLen = minCount, minLen > tokens.count {
            tokens.append(contentsOf: repeatElement(padToken, count: minLen - tokens.count))
        }
        let ids = tokens.map { vocabulary[$0, default: unknownTokenID] }
        return (tokens, ids)
    }

    public func tokenID(for token: String) -> Int? {
        vocabulary[token]
    }

    public func token(id: Int) -> String? {
        vocabulary.first { $0.value == id }?.key
    }

    public func decode(tokens: [String]) -> String {
        tokens.joined().replacingOccurrences(of: "</w>", with: " ")
            .replacingOccurrences(of: startToken, with: "")
            .replacingOccurrences(of: endToken, with: "")
    }

    func encode(input: String) -> [String] {
        let normalized = input.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return normalized.split(separator: " ").flatMap { encode(word: $0) }
    }

    func encode(word: Substring) -> [String] {
        var tokens = word.map { String($0) }
        if let last = tokens.indices.last {
            tokens[last] += "</w>"
        }

        while true {
            let pairs = self.pairs(for: tokens)
            guard let shouldMerge = pairs.compactMap({ merges[$0] != nil ? $0 : nil }).min(by: { merges[$0]! < merges[$1]! }) else {
                break
            }
            tokens = update(tokens, merging: shouldMerge)
        }
        return tokens
    }

    func pairs(for tokens: [String]) -> Set<TokenPair> {
        guard tokens.count > 1 else {
            return Set()
        }
        return Set(zip(tokens, tokens.dropFirst()).map { TokenPair($0, $1) })
    }

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
                newTokens.append(contentsOf: tokens[index..<startMatchIndex])
                newTokens.append(bigram.first + bigram.second)
                index = startMatchIndex + 2
            } else {
                newTokens.append(tokens[index])
                index += 1
            }
        }
        return newTokens
    }
}

@available(iOS 16.2, macOS 13.1, *)
extension BPETokenizer {
    public struct TokenPair: Hashable {
        let first: String
        let second: String

        init(_ first: String, _ second: String) {
            self.first = first
            self.second = second
        }
    }

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

    static func readVocabulary(url: URL) throws -> [String: Int] {
        let data = try Data(contentsOf: url)
        let vocabulary = try JSONDecoder().decode([String: Int].self, from: data)
        return vocabulary
    }
}
