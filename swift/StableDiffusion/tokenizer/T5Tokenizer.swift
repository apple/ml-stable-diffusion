// Copied from https://github.com/huggingface/swift-transformers/tree/main/Sources/Tokenizers

import Foundation

// MARK: - Configuration files with dynamic lookup

@dynamicMemberLookup
public struct Config {
    public private(set) var dictionary: [String: Any]

    public init(_ dictionary: [String: Any]) {
        self.dictionary = dictionary
    }
    
    /// Assumes the file is already present at local url.
    /// `fileURL` is a complete local file path for the given model
    public init(fileURL: URL) throws {
        let data = try Data(contentsOf: fileURL)
        let parsed = try JSONSerialization.jsonObject(with: data, options: [])
        guard let dictionary = parsed as? [String: Any] else { throw TokenizerError.malformedVocab }
        self.dictionary = dictionary
    }

    func camelCase(_ string: String) -> String {
        return string
            .split(separator: "_")
            .enumerated()
            .map { $0.offset == 0 ? $0.element.lowercased() : $0.element.capitalized }
            .joined()
    }
    
    func uncamelCase(_ string: String) -> String {
        let scalars = string.unicodeScalars
        var result = ""
        
        var previousCharacterIsLowercase = false
        for scalar in scalars {
            if CharacterSet.uppercaseLetters.contains(scalar) {
                if previousCharacterIsLowercase {
                    result += "_"
                }
                let lowercaseChar = Character(scalar).lowercased()
                result += lowercaseChar
                previousCharacterIsLowercase = false
            } else {
                result += String(scalar)
                previousCharacterIsLowercase = true
            }
        }
        
        return result
    }


    public subscript(dynamicMember member: String) -> Config? {
        let key = dictionary[member] != nil ? member : uncamelCase(member)
        if let value = dictionary[key] as? [String: Any] {
            return Config(value)
        } else if let value = dictionary[key] {
            return Config(["value": value])
        }
        return nil
    }

    public var value: Any? {
        return dictionary["value"]
    }
    
    public var intValue: Int? { value as? Int }
    public var boolValue: Bool? { value as? Bool }
    public var stringValue: String? { value as? String }
    
    // Instead of doing this we could provide custom classes and decode to them
    public var arrayValue: [Config]? {
        guard let list = value as? [Any] else { return nil }
        return list.map { Config($0 as! [String : Any]) }
    }
    
    /// Tuple of token identifier and string value
    public var tokenValue: (UInt, String)? { value as? (UInt, String) }
}

enum TokenizerError : Error {
    case missingConfig
    case missingTokenizerClassInConfig
    case unsupportedTokenizer(String)
    case missingVocab
    case malformedVocab

    case tooLong(String)
}

public protocol TokenizingModel {
    func tokenize(input text: String) -> (tokens: [String], tokenIDs: [Int])

    // Alias for `tokenize`
    func callAsFunction(_ text: String) -> [String]

    func convertTokenToId(_ token: String) -> Int?
    func convertTokensToIds(_ tokens: [String]) -> [Int?]

    func convertIdToToken(_ id: Int) -> String?
    func convertIdsToTokens(_ ids: [Int]) -> [String?]

    var bosToken: String? { get }
    var bosTokenId: Int { get }
    var padTokenId: Int { get }
    var maskTokenId: Int { get }
    var eosToken: String? { get }
    var eosTokenId: Int { get }
    var unknownToken: String? { get }
    var unknownTokenId: Int? { get }
}

public extension TokenizingModel {
    func callAsFunction(_ text: String) -> [String] {
        tokenize(input: text).tokens
    }

    func convertTokensToIds(_ tokens: [String]) -> [Int?] {
        return tokens.map { convertTokenToId($0) }
    }

    func convertIdsToTokens(_ ids: [Int]) -> [String?] {
        return ids.map { convertIdToToken($0) }
    }
}

/// A tokenizer model that is set up with Hub configuration data
public protocol PreTrainedTokenizerModel: TokenizingModel {
    init(tokenizerConfig: Config, tokenizerData: Config, addedTokens: [String : Int]) throws
}

struct TokenizerModel {
    static func unknownToken(from tokenizerConfig: Config) -> String? {
        return tokenizerConfig.unkToken?.content?.stringValue ?? tokenizerConfig.unkToken?.stringValue
    }

    public static func from(tokenizerConfig: Config, tokenizerData: Config, addedTokens: [String : Int]) throws -> TokenizingModel {
        return try UnigramTokenizer.init(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData, addedTokens: addedTokens)
    }
}

public protocol Tokenizer {
    func tokenize(input text: String) -> (tokens: [String], tokenIDs: [Int])
    func tokenize(input text: String) -> [String]

    /// Main entry point
    func encode(text: String) -> [Int]
    func callAsFunction(_ text: String) -> [Int]

    /// Decode
    func decode(tokens: [Int]) -> String

    func convertTokenToId(_ token: String) -> Int?
    func convertTokensToIds(_ tokens: [String]) -> [Int?]

    func convertIdToToken(_ id: Int) -> String?
    func convertIdsToTokens(_ ids: [Int]) -> [String?]

    var bosToken: String? { get }
    var bosTokenId: Int { get }
    var padTokenId: Int { get }
    var maskTokenId: Int { get }
    var eosToken: String? { get }
    var eosTokenId: Int { get }
    var unknownToken: String? { get }
    var unknownTokenId: Int? { get }
}

public extension Tokenizer {
    func callAsFunction(_ text: String) -> [Int] {
        encode(text: text)
    }

    func convertTokensToIds(_ tokens: [String]) -> [Int?] {
        return tokens.map { convertTokenToId($0) }
    }

    func convertIdsToTokens(_ ids: [Int]) -> [String?] {
        return ids.map { convertIdToToken($0) }
    }
}

public class PreTrainedTokenizer: Tokenizer {
    let model: TokenizingModel

    public var bosToken: String? { model.bosToken }
    public var bosTokenId: Int { model.bosTokenId }
    public var padTokenId: Int { model.padTokenId }
    public var maskTokenId: Int { model.maskTokenId }
    public var eosToken: String? { model.eosToken }
    public var eosTokenId: Int { model.eosTokenId }
    public var unknownToken: String? { model.unknownToken }
    public var unknownTokenId: Int? { model.unknownTokenId }

    private let addedTokens: Set<String>
    private let specialTokens: [String: Int]
    private let addedTokensRegex: NSRegularExpression?

    private let preTokenizer: PreTokenizer?
    private let normalizer: Normalizer?
    private let postProcessor: PostProcessor?
    private let decoder: TokenDecoder?

    private let cleanUpTokenizationSpaces: Bool
    
    convenience public init(tokenizerConfigURL: URL, tokenizerDataURL: URL) throws {
        let tokenizerConfig = try Config(fileURL: tokenizerConfigURL)
        let tokenizerData = try Config(fileURL: tokenizerDataURL)
        
        try self.init(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData)
    }

    required public init(tokenizerConfig: Config, tokenizerData: Config) throws {
        var addedTokens: [String : Int] = [:]
        var specialTokens: [String : Int] = [:]
        for addedToken in tokenizerData.addedTokens?.arrayValue ?? [] {
            guard let id = addedToken.id?.intValue else { continue /* malformed: token with no id */ }
            guard let content = addedToken.content?.stringValue else { continue /* malformed: token with no content */ }
            addedTokens[content] = id

            if addedToken.special?.boolValue ?? false {
                specialTokens[content] = id
            }
        }

        let addedTokensRegexString = (tokenizerData.addedTokens?.arrayValue ?? []).compactMap { addedToken in
               guard let content = addedToken.content?.stringValue else { return nil }
               let prefix = (addedToken.lstrip?.boolValue ?? false ? #"\s*"# : "")
               let suffix = (addedToken.rstrip?.boolValue ?? false ? #"\s*"# : "")
               let token = NSRegularExpression.escapedPattern(for: content)
               return "\(prefix)(\(token))\(suffix)"
        }.joined(separator: "|")
        addedTokensRegex = try? NSRegularExpression(pattern: addedTokensRegexString, options: [])

        // TODO: specialTokens are stored but never used
        self.specialTokens = specialTokens
        self.addedTokens = Set(addedTokens.keys)

        self.preTokenizer = PreTokenizerFactory.fromConfig(config: tokenizerData.preTokenizer)
        self.normalizer = NormalizerFactory.fromConfig(config: tokenizerData.normalizer)
        self.postProcessor = PostProcessorFactory.fromConfig(config: tokenizerData.postProcessor)
        self.decoder = DecoderFactory.fromConfig(config: tokenizerData.decoder)
        self.cleanUpTokenizationSpaces = tokenizerConfig.cleanUpTokenizationSpaces?.boolValue ?? true

        model = try TokenizerModel.from(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData, addedTokens: addedTokens)
    }

    func preTokenize(_ text: String, options: PreTokenizerOptions) -> [String] {
        guard let preTokenizer = preTokenizer else { return [text] }
        return preTokenizer(text: text, options: options)
    }

    func normalize(_ text: String) -> String {
        guard let normalizer = normalizer else { return text }
        return normalizer(text: text)
    }

    func postProcess(_ tokens: [String]) -> [String] {
        guard let postProcessor = postProcessor else { return tokens }
        return postProcessor(tokens: tokens)
    }
    
    func decodeTokens(_ tokens: [String]) -> [String] {
        guard let tokenDecoder = decoder else { return tokens }
        return tokenDecoder(tokens: tokens)
    }

    /// Clean up a list of simple English tokenization artifacts like spaces before punctuations and abbreviated forms
    func cleanUp(text: String) -> String {
        guard cleanUpTokenizationSpaces else { return text }

        return text.replacingOccurrences(of: " .", with: ".")
            .replacingOccurrences(of: " ?", with: "?")
            .replacingOccurrences(of: " !", with: "!")
            .replacingOccurrences(of: " ,", with: ",")
            .replacingOccurrences(of: " ' ", with: "'")
            .replacingOccurrences(of: " n't", with: "n't")
            .replacingOccurrences(of: " 'm", with: "'m")
            .replacingOccurrences(of: " 's", with: "'s")
            .replacingOccurrences(of: " 've", with: "'ve")
            .replacingOccurrences(of: " 're", with: "'re")
    }

    public func tokenize(input text: String) -> [String] {
        return tokenize(input: text).tokens
    }
    
    public func tokenize(input text: String) -> (tokens: [String], tokenIDs: [Int]) {
        // Take care of special tokens first
        let sections: [String]
        if let regex = self.addedTokensRegex {
            sections = text.split(by: regex)
        } else {
            sections = [text]
        }
        let tokens = sections.enumerated().flatMap { (section, x) -> [String] in
            if addedTokens.contains(x) {
                return [x]
            }
            // Normalize and pre-tokenize the text, then process each token with `model` and flatten the results
            return preTokenize(normalize(x), options: section == 0 ? [.firstSection] : []).flatMap { model($0) }
        }        
        let tokenIds = convertTokensToIds(tokens).map { $0 ?? 0 }
        return (tokens, tokenIds)
    }


    /// Main entry point
    public func encode(text: String) -> [Int] {
        return postProcess(tokenize(input: text)).map { model.convertTokenToId($0)! }
    }

    /// Decode
    public func decode(tokens: [Int]) -> String {
        // IDs to tokens
        let tokenStrings = tokens.map { model.convertIdToToken($0)! }
        let decoded = decodeTokens(tokenStrings)
        // At this point we should have a single String
        return cleanUp(text: decoded.joined(separator: ""))
    }

    public func convertTokenToId(_ token: String) -> Int? {
        model.convertTokenToId(token)
    }

    public func convertIdToToken(_ id: Int) -> String? {
        model.convertIdToToken(id)
    }
}

public class UnigramTokenizer: PreTrainedTokenizerModel {
    struct SentencePieceToken {
        var token: String
        var score: Float
    }
    let vocab: [SentencePieceToken]
    
    let unknownPiece: SentencePieceToken
    var unknownTokenScore: Float { unknownPiece.score }
    
    public let unknownTokenId: Int?
    public var unknownToken: String? { unknownPiece.token }
    
    let minScore: Float
    let tokensToIds: [String: Int]
    
    public let bosToken: String? = " "
    public let bosTokenId: Int = 0
    public let padTokenId: Int = 0
    public let maskTokenId: Int = 1
    public let eosToken: String?
    public let eosTokenId: Int = 1
    
    private let trie: Trie<Character>
    
        
    required public init(tokenizerConfig: Config, tokenizerData: Config, addedTokens: [String : Int]) throws {
        guard let configVocab = tokenizerData.model?.vocab?.value as? [[Any]] else {
            throw TokenizerError.missingVocab
        }
        
        vocab = try configVocab.map { piece in
            guard let token = piece.first as? String,
                  let scoreValue = piece.last else {
                throw TokenizerError.malformedVocab
            }

            let score: Float
            if let floatScore = scoreValue as? Float {
                score = floatScore
            } else if let numberScore = scoreValue as? NSNumber {
                score = numberScore.floatValue
            } else {
                throw TokenizerError.malformedVocab
            }
            
            return SentencePieceToken(token: token, score: score)
        }
        
        minScore = vocab.reduce(999) { partial, token in
            min(partial, token.score)
        }
        
        guard let unknownTokenId = tokenizerData.model?.unkId?.intValue else { throw TokenizerError.malformedVocab }
        self.unknownTokenId = unknownTokenId
        self.unknownPiece = SentencePieceToken(token: vocab[unknownTokenId].token, score: minScore - 10)
        
        tokensToIds = Dictionary(uniqueKeysWithValues: vocab.map { $0.token }.enumerated().map { ($1, $0) })
        
        eosToken = tokenizerConfig.eosToken?.stringValue
        
        trie = Trie()
        trie.append(contentsOf: vocab.map { Array($0.token) })
    }
    
    public func convertTokenToId(_ token: String) -> Int? {
        return tokensToIds[token] ?? self.unknownTokenId
    }
    
    public func convertIdToToken(_ id: Int) -> String? {
        return vocab[id].token
    }
        
    public func tokenize(input text: String) -> (tokens: [String], tokenIDs: [Int]) {
        var lattice = TokenLattice(sentence: text, bosTokenId: bosTokenId, eosTokenId: eosTokenId)
        
        // Populate nodes
        let sentence = lattice.sentence
        var beginPos = 0
        while beginPos < sentence.count {
            let mblen = 1
            var hasSingleNode = false
            
            let beginIndex = sentence.index(sentence.startIndex, offsetBy: beginPos)
            for token in trie.commonPrefixSearchIterator(sentence[beginIndex...]).map({ String($0) }) {
                guard let tokenId = tokensToIds[token] else { fatalError("Token not in vocab: \(token)") }
                let tokenScore = vocab[tokenId].score
                lattice.insert(startOffset: beginPos, length: token.count, score: tokenScore, tokenId: tokenId)
                if !hasSingleNode && token.count == mblen {
                    hasSingleNode = true
                }
            }
            if !hasSingleNode {
                lattice.insert(startOffset: beginPos, length: mblen, score: unknownTokenScore, tokenId: unknownTokenId ?? 0)
            }
            beginPos += mblen
        }

        return (lattice.tokens, lattice.tokenIds)
    }
}

/// Implements a TokenLattice to implement the Viterbi algorithm
/// We could make it generic so TokenLatticeNode stores an opaque type, but it's overkill right now.
/// Based on https://github.com/huggingface/tokenizers/blob/b58227c7f1ccf8b73ee2268354336da56d91e492/tokenizers/src/models/unigram/lattice.rs#L137
/// and https://github.com/xenova/transformers.js/blob/b07336d8f7ff57453cc164cc68aead2a79cbd57e/src/utils/data-structures.js#L269C28-L269C28
public struct TokenLattice {
    let sentence: String
    let bosTokenId: Int
    let eosTokenId: Int
    
    var nodes: [TokenLatticeNode] = []
    var beginNodes: [[TokenLatticeNode]]
    var endNodes: [[TokenLatticeNode]]
    
    var count: Int { sentence.count }

    init(sentence: String, bosTokenId: Int, eosTokenId: Int) {
        self.sentence = sentence
        self.bosTokenId = bosTokenId
        self.eosTokenId = eosTokenId
        
        beginNodes = Array(repeating: [], count: sentence.count+1)
        endNodes = Array(repeating: [], count: sentence.count+1)
        
        let bos = TokenLatticeNode(tokenId: bosTokenId, startOffset: 0, length: 0, score: 0)
        let eos = TokenLatticeNode(tokenId: eosTokenId, startOffset: sentence.count, length: 0, score: 0)
        
        nodes.append(bos)
        nodes.append(eos)
        
        beginNodes[sentence.count].append(eos)
        endNodes[0].append(bos)
    }
}

public extension TokenLattice {
    /// Insert a new token into the node lattice.
    ///
    ///  - Parameters:
    ///      - startOffset: Starting position of the token in the sentence.
    ///      - length: Number of characters in the token.
    ///      - score: Token score.
    ///      - tokenId: Token id in the tokenizer.
    mutating func insert(startOffset: Int, length: Int, score: Float, tokenId: Int) {
        let node = TokenLatticeNode(tokenId: tokenId, startOffset: startOffset, length: length, score: score)
        beginNodes[startOffset].append(node)
        endNodes[startOffset + length].append(node)
        nodes.append(node)
    }
}

extension TokenLattice {
    /// Implements the Viterbi algorithm to compute the most likely sequence of tokens.
    /// It's unfortunate that it can't be lazy or cached as the node arrays are not immutable.
    /// We could create another type that holds the nodes and use it as an immutable var  in TokenLattice.
    func viterbi() -> [TokenLatticeNode] {
        for offset in 0...count {
            guard beginNodes[offset].count > 0 else { return [] }
            
            for rnode in beginNodes[offset] {
                rnode.prev = nil
                var bestScore: Float = 0
                var bestNode: TokenLatticeNode? = nil
                for lnode in endNodes[offset] {
                    let score = lnode.backtraceScore + rnode.score
                    if bestNode == nil || score > bestScore {
                        bestNode = lnode.clone()
                        bestScore = score
                    }
                }
                
                if bestNode != nil {
                    rnode.prev = bestNode
                    rnode.backtraceScore = bestScore
                }
            }
        }
        
        let root = beginNodes[count][0]
        guard let prev = root.prev else { return [] }

        // TODO: the reference implementations have a few more clones here: verify
        var result: [TokenLatticeNode] = []
        var node = prev     //.clone()
        while node.prev != nil {
            result.append(node.clone())
            node = node.prev!   //.clone()
        }
        return result.reversed()
    }
    
    /// Returns the substring of the sentence to be tokenized associated to the specified node
    ///
    /// - Parameters:
    ///     - node: The node defining the token to be extracted
    ///
    /// - Returns: A **Substring** – i.e., a reference to the original positions, not a copy of the characters.
    func piece(_ node: TokenLatticeNode) -> any StringProtocol {
        let start = sentence.index(sentence.startIndex, offsetBy: node.startOffset)
        let end = sentence.index(start, offsetBy: node.length)
        return sentence[start..<end]
    }
}

public extension TokenLattice {
    var tokens: [String] {
        viterbi().map { String(piece($0)) }
    }
    
    var tokenIds: [Int] {
        viterbi().map { $0.tokenId }
    }
}

class TokenLatticeNode {
    let tokenId: Int
    let startOffset: Int
    let length: Int
    let score: Float
    
    var prev: TokenLatticeNode? = nil
    var backtraceScore: Float = 0
    
    init(tokenId: Int, startOffset: Int, length: Int, score: Float, prev: TokenLatticeNode? = nil, backtraceScore: Float = 0) {
        self.tokenId = tokenId
        self.startOffset = startOffset
        self.length = length
        self.score = score
        self.prev = prev
        self.backtraceScore = backtraceScore
    }
}

extension TokenLatticeNode {
    // This is a reference type because structs can't contain references to the same type
    // We could implement NSCopying, but frankly I don't see the point
    func clone() -> TokenLatticeNode {
        TokenLatticeNode(tokenId: tokenId, startOffset: startOffset, length: length, score: score, prev: prev, backtraceScore: backtraceScore)
    }
}

extension TokenLatticeNode: CustomStringConvertible {
    var description: String {
        "TokenLatticeNode(tokenId: \(tokenId), startOffset: \(startOffset), length: \(length), score: \(score), prev: \(prev != nil), backtraceScore: \(backtraceScore)"
    }
}

// Reference:

public struct Trie<T: Hashable> {
    public typealias Node = TrieNode<T>
    
    var root: Node
    
    public init(root: Node? = nil) {
        self.root = root ?? Node()
    }
}

public extension Trie {
    func insert(_ element: [T]) {
        var node = root
        for item in element {
            if let child = node.children[item] {
                node = child
            } else {
                let child = Node()
                node.children[item] = child
                node = child
            }
        }
        node.isLeaf = true
    }
    
    func append(contentsOf container: [[T]]) {
        for t in container { insert(t) }
    }
        
    /// Find all leaf nodes that share a common prefix with the input sequence (usually a text)
    /// Returns an array
    func commonPrefixSearch(_ text: any Sequence<T>) -> [[T]] {
        var node = root
        var seqs: [[T]] = []
        var seq: [T] = []
        for item in text {
            seq.append(item)
            guard let child = node.children[item] else { return seqs }
            node = child
            if node.isLeaf {
                seqs.append(seq)
            }
        }
        return seqs
    }
    
    /// Find all leaf nodes that share a common prefix with the input sequence (usually a text)
    /// Returns an iterator
    func commonPrefixSearchIterator(_ text: any Sequence<T>) -> LeavesWithCommonPrefixIterator<T> {
        return LeavesWithCommonPrefixIterator(node: root, text: text)
    }
}

public extension Trie {
    // Only used for testing, could migrate to collection
    func get(_ element: any Sequence<T>) -> Node? {
        var node = root
        for item in element {
            guard let child = node.children[item] else { return nil }
            node = child
        }
        return node
    }
}

public class TrieNode<T: Hashable> {
    var isLeaf: Bool = false
    var children: [T: TrieNode] = [:]
}

public struct LeavesWithCommonPrefixIterator<T: Hashable> : Sequence, IteratorProtocol {
    var node: TrieNode<T>
    var text: any Sequence<T>
    var seq: [T] = []
    lazy var iterator = text.makeIterator() as any IteratorProtocol<T>
    
    public mutating func next() -> [T]? {
        while true {
            guard let item = iterator.next() else { return nil }
            seq.append(item)
            guard let child = node.children[item] else { return nil }
            node = child
            if node.isLeaf {
                return seq
            }
        }
    }
}

public protocol PostProcessor {
    func postProcess(tokens: [String], tokensPair: [String]?) -> [String]
    func callAsFunction(tokens: [String], tokensPair: [String]?) -> [String]
    
    init(config: Config)
}

extension PostProcessor {
    func callAsFunction(tokens: [String], tokensPair: [String]? = nil) -> [String] {
        return postProcess(tokens: tokens, tokensPair: tokensPair)
    }
}

enum PostProcessorType: String {
    case TemplateProcessing
    case ByteLevel
    case RobertaProcessing
}

struct PostProcessorFactory {
    static func fromConfig(config: Config?) -> PostProcessor? {
        guard let config = config else { return nil }
        guard let typeName = config.type?.stringValue else { return nil }
        let type = PostProcessorType(rawValue: typeName)
        switch type {
        case .TemplateProcessing: return TemplateProcessing(config: config)
        case .ByteLevel         : return ByteLevelPostProcessor(config: config)
        case .RobertaProcessing : return RobertaProcessing(config: config)
        default                 : fatalError("Unsupported PostProcessor type: \(typeName)")
        }
    }
}

class TemplateProcessing: PostProcessor {
    let single: [Config]
    let pair: [Config]
    
    required public init(config: Config) {
        guard let single = config.single?.arrayValue else { fatalError("Missing `single` processor configuration") }
        guard let pair = config.pair?.arrayValue else { fatalError("Missing `pair` processor configuration") }
        
        self.single = single
        self.pair = pair
    }
    
    func postProcess(tokens: [String], tokensPair: [String]? = nil) -> [String] {
        let config = tokensPair == nil ? single : pair
                
        var toReturn: [String] = []
        for item in config {
            if let specialToken = item.SpecialToken {
                toReturn.append(specialToken.id!.stringValue!)
            } else if let sequence = item.Sequence {
                if sequence.id?.stringValue == "A" {
                    toReturn += tokens
                } else if sequence.id?.stringValue == "B" {
                    toReturn += tokensPair!
                }
            }
        }
        return toReturn
    }
}

class ByteLevelPostProcessor: PostProcessor {
    required public init(config: Config) {}
    func postProcess(tokens: [String], tokensPair: [String]? = nil) -> [String] { tokens }
}

class RobertaProcessing: PostProcessor {
    private let sep: (UInt, String)
    private let cls: (UInt, String)
    /// Trim all remaining space, or leave one space character if `addPrefixSpace` is `true`.
    private let trimOffset: Bool
    /// Keep one space character on each side. Depends on `trimOffsets` being `true`.
    private let addPrefixSpace: Bool

    required public init(config: Config) {
        guard let sep = config.sep?.tokenValue else { fatalError("Missing `sep` processor configuration") }
        guard let cls = config.cls?.tokenValue else { fatalError("Missing `cls` processor configuration") }
        self.sep = sep
        self.cls = cls
        self.trimOffset = config.trimOffset?.boolValue ?? true
        self.addPrefixSpace = config.addPrefixSpace?.boolValue ?? true
    }
    
    func postProcess(tokens: [String], tokensPair: [String]?) -> [String] {
        var outTokens = tokens
        var tokensPair = tokensPair
        if trimOffset {
            if addPrefixSpace {
                outTokens = outTokens.map({ trimExtraSpaces(token: $0) })
                tokensPair = tokensPair?.map({ trimExtraSpaces(token: $0) })
           } else {
                outTokens = outTokens.map({ $0.trimmingCharacters(in: .whitespaces) })
                tokensPair = tokensPair?.map({ $0.trimmingCharacters(in: .whitespaces) })
            }
        }

        outTokens = [self.cls.1] + outTokens + [self.sep.1]
        if let tokensPair = tokensPair, !tokensPair.isEmpty {
            // Yes, it adds another `sep`.
            // https://github.com/facebookresearch/fairseq/blob/main/fairseq/models/roberta/hub_interface.py#L58-L65
            outTokens += [self.sep.1] + tokensPair + [self.sep.1]
        }

        return outTokens
    }

    /// Some tokens need one space around them
    /// https://github.com/huggingface/tokenizers/blob/main/tokenizers/src/pre_tokenizers/byte_level.rs#L203-L235
    private func trimExtraSpaces(token: String) -> String {
        let prefixOffset = findPrefixIndex(text: token)
        let suffixOffset = findSuffixIndex(text: token)
        let prefixIndex = token.index(token.startIndex, offsetBy: prefixOffset)
        let suffixIndex = token.index(token.startIndex, offsetBy: token.count - suffixOffset)
        return String(token[prefixIndex..<suffixIndex])
    }

    private func findPrefixIndex(text: String) -> Int {
        guard !text.isEmpty, text.first!.isWhitespace else { return 0 }
        return text.prefix(while: { $0.isWhitespace }).count - 1
    }

    private func findSuffixIndex(text: String) -> Int {
        guard !text.isEmpty, text.last!.isWhitespace else { return 0 }
        return text.reversed().prefix(while: { $0.isWhitespace }).count - 1
    }
}

public enum PreTokenizerOption: String {
    case firstSection
}

public typealias PreTokenizerOptions = Set<PreTokenizerOption>

public protocol PreTokenizer {
    func preTokenize(text: String, options: PreTokenizerOptions) -> [String]
    func preTokenize(texts: [String], options: PreTokenizerOptions) -> [String]
    func callAsFunction(texts: [String], options: PreTokenizerOptions) -> [String]
    func callAsFunction(text: String, options: PreTokenizerOptions) -> [String]

    init(config: Config)
}

extension PreTokenizer {
    func preTokenize(texts: [String], options: PreTokenizerOptions = [.firstSection]) -> [String] {
        texts.flatMap { preTokenize(text: $0, options: options) }
    }

    func callAsFunction(texts: [String], options: PreTokenizerOptions = [.firstSection]) -> [String] {
        return preTokenize(texts: texts, options: options)
    }
    
    func callAsFunction(text: String, options: PreTokenizerOptions = [.firstSection]) -> [String] {
        return preTokenize(text: text, options: options)
    }
}

enum PreTokenizerType: String {
    case Sequence
    case ByteLevel
    case Punctuation
    case Digits
    case Split
    case Whitespace
    case WhitespaceSplit
    case Metaspace
    // Several more to be supported
    case Unknown = ""
}

struct PreTokenizerFactory {
    static func fromConfig(config: Config?) -> PreTokenizer? {
        guard let config = config else { return nil }
        guard let typeName = config.type?.stringValue else { return nil }
        let type = PreTokenizerType(rawValue: typeName)
        switch type {
        case .Sequence : return PreTokenizerSequence(config: config)
        case .ByteLevel: return ByteLevelPreTokenizer(config: config)
        case .Punctuation: return PunctuationPreTokenizer(config: config)
        case .Digits: return DigitsPreTokenizer(config: config)
        case .Split: return SplitPreTokenizer(config: config)
        case .Whitespace, .WhitespaceSplit: return WhitespacePreTokenizer(config: config)
        case .Metaspace: return MetaspacePreTokenizer(config: config)
        default: fatalError("Unsupported PreTokenizer type: \(typeName)")
        }
    }
}

class PreTokenizerSequence: PreTokenizer {
    let preTokenizers: [PreTokenizer]
    
    required init(config: Config) {
        guard let configs = config.pretokenizers?.arrayValue else { fatalError("No pretokenizers in Sequence") }
        preTokenizers = configs.compactMap { PreTokenizerFactory.fromConfig(config: $0) }
    }
    
    func preTokenize(text: String, options: PreTokenizerOptions = [.firstSection]) -> [String] {
        preTokenizers.reduce([text]) { current, preTokenizer in
            preTokenizer(texts: current, options: options)
        }
    }
}

class WhitespacePreTokenizer: PreTokenizer {
    let re: String

    required init(config: Config) {
        re = #"\S+"#
    }

    func preTokenize(text: String, options: PreTokenizerOptions = [.firstSection]) -> [String] {
        return text.ranges(of: re).map { String(text[$0]) }
    }
}

/// PreTokenizer that replaces spaces with the given replacement character, adds a prefix space if requested,
class MetaspacePreTokenizer: PreTokenizer {
    /// Whether to add a prefix space to the first token
    let addPrefixSpace: Bool
    
    /// Replacement character
    let replacement: String
    
    /// Optional string representation of the replacement character.
    let stringReplacement: String
    
    enum PrependScheme: String {
        case first
        case never
        case always
        
        static var defaultScheme: PrependScheme { .always }
        static func from(rawValue value: String?) -> PrependScheme {
            guard let value = value else { return defaultScheme }
            return PrependScheme(rawValue: value) ?? defaultScheme
        }
    }
    
    /// The metaspace prepend scheme, see https://github.com/huggingface/tokenizers/pull/1357
    let prependScheme: PrependScheme
    
    required init(config: Config) {
        addPrefixSpace = config.addPrefixSpace?.boolValue ?? false
        replacement = config.replacement?.stringValue ?? " "
        stringReplacement = config.strRep?.stringValue ?? replacement
        prependScheme = PrependScheme.from(rawValue: config.prependScheme?.stringValue)
    }
    
    // https://github.com/huggingface/tokenizers/blob/accd0650b802f2180df40ef1def3bce32156688e/tokenizers/src/pre_tokenizers/metaspace.rs#L114
    // https://github.com/xenova/transformers.js/blob/b07336d8f7ff57453cc164cc68aead2a79cbd57e/src/tokenizers.js#L2153
    func preTokenize(text: String, options: PreTokenizerOptions = [.firstSection]) -> [String] {
        let normalized = text.replacingOccurrences(of: " ", with: stringReplacement)
        
        // We add a prefix space if:
        //  (1) The addPrefixSpace option is enabled and the normalized
        //      token does not already start with the replacement character.
        //  and (2) either:
        //  (a) prependScheme is 'always'
        //  (b) prependScheme is 'first' and this is the first section
        // FIXME: (2b) always prepends, we are not passing section info

        var prepend = ""
        if addPrefixSpace && !normalized.hasPrefix(replacement) {
            if prependScheme == .always {
                prepend = stringReplacement
            }
            if prependScheme == .first && options.contains(.firstSection) {
                prepend = stringReplacement
            }
        }
        
        // Split in `MergedWithNext` mode, although usually the input to this function is already pre-tokenized
        // https://github.com/huggingface/tokenizers/blob/accd0650b802f2180df40ef1def3bce32156688e/tokenizers/src/pre_tokenizers/metaspace.rs#L127
        return (prepend + normalized).split(by: replacement, behavior: .mergedWithNext)
    }
}

class ByteLevelPreTokenizer: PreTokenizer {
    let addPrefixSpace: Bool
    let trimOffsets: Bool
    let useRegex: Bool
    let RE = #"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"#
    
    required init(config: Config) {
        addPrefixSpace = config.addPrefixSpace?.boolValue ?? false
        trimOffsets = config.trimOffsets?.boolValue ?? true
        useRegex = config.useRegex?.boolValue ?? true
    }
    
    func preTokenize(text: String, options: PreTokenizerOptions = [.firstSection]) -> [String] {
        // Split on whitespace and punctuation
        let tokens = useRegex ? text.ranges(of: RE).map({ String(text[$0]) }) : [text]
        return tokens.map { token in
            if addPrefixSpace && !token.hasPrefix(" ") {
                return " " + token
            }
            return token
        }.map { token in
            return Array(token.utf8).map { byteEncoder[$0]! }.joined()
        }
    }
}

class PunctuationPreTokenizer: PreTokenizer {
    let PUNCTUATION_REGEX = #"\p{P}\u0021-\u002F\u003A-\u0040\u005B-\u0060\u007B-\u007E"#
    let re: String

    required init(config: Config) {
        re = "[^\(PUNCTUATION_REGEX)]+|[\(PUNCTUATION_REGEX)]+"
    }

    func preTokenize(text: String, options: PreTokenizerOptions = [.firstSection]) -> [String] {
        // Ref: https://github.com/xenova/transformers.js/blob/27920d84831e323275b38f0b5186644b7936e1a2/src/tokenizers.js#L1138
        return text.ranges(of: re).map { String(text[$0]) }
    }
}

class DigitsPreTokenizer: PreTokenizer {
    let re: String

    required init(config: Config) {
        let individualDigits = config.individualDigits?.boolValue ?? false
        re = "[^\\d]+|\\d\(individualDigits ? "" : "+")"
    }

    func preTokenize(text: String, options: PreTokenizerOptions = [.firstSection]) -> [String] {
        return text.ranges(of: re).map { String(text[$0]) }
    }
}

class SplitPreTokenizer: PreTokenizer {
    let pattern: StringSplitPattern?
    let invert: Bool

    required init(config: Config) {
        pattern = StringSplitPattern.from(config: config)
        invert = config.invert?.boolValue ?? false
    }

    func preTokenize(text: String, options: PreTokenizerOptions = [.firstSection]) -> [String] {
        guard let pattern = pattern else { return [text] }
        return pattern.split(text, invert: invert)
    }
}

enum StringSplitPattern {
    case regexp(regexp: String)
    case string(pattern: String)
}

extension StringSplitPattern {
    func split(_ text: String, invert: Bool = true) -> [String] {
        switch self {
        case .regexp(let regexp):
            return text.split(by: regexp, includeSeparators: !invert)
        case .string(let substring):
            return text.split(by: substring, options: [], includeSeparators: !invert)
        }
    }
}

extension StringSplitPattern {
    static func from(config: Config) -> StringSplitPattern? {
        if let pattern = config.pattern?.String?.stringValue {
            return StringSplitPattern.string(pattern: pattern)
        }
        if let pattern = config.pattern?.Regex?.stringValue {
            return StringSplitPattern.regexp(regexp: pattern)
        }
        return nil
    }
}

public extension String {
    func ranges(of string: String, options: CompareOptions = .regularExpression) -> [Range<Index>] {
        var result: [Range<Index>] = []
        var start = startIndex
        while let range = range(of: string, options: options, range: start..<endIndex) {
            result.append(range)
            start = range.lowerBound < range.upperBound ? range.upperBound : index(range.lowerBound, offsetBy: 1, limitedBy: endIndex) ?? endIndex
        }
        return result
    }
        
    func split(by string: String, options: CompareOptions = .regularExpression, includeSeparators: Bool = false, omittingEmptySubsequences: Bool = true) -> [String] {
        var result: [String] = []
        var start = startIndex
        while let range = range(of: string, options: options, range: start..<endIndex) {
            // Prevent empty strings
            if omittingEmptySubsequences && start < range.lowerBound {
                result.append(String(self[start..<range.lowerBound]))
            }
            if includeSeparators {
                result.append(String(self[range]))
            }
            start = range.upperBound
        }
        
        result.append(String(self[start...]))
        return result
    }

    /// This version supports capture groups, wheres the one above doesn't
    func split(by captureRegex: NSRegularExpression) -> [String] {
        // Find the matching capture groups
        let selfRange = NSRange(startIndex..<endIndex, in: self)
        let matches = captureRegex.matches(in: self, options: [], range: selfRange)

        if matches.first == nil { return [self] }

        var result: [String] = []
        var start = startIndex
        for match in matches {
            // Append prefix before matched separator
            let prefixEnd = index(startIndex, offsetBy: match.range.lowerBound)
            if start < prefixEnd {
                result.append(String(self[start..<prefixEnd]))
            }
            start = index(startIndex, offsetBy: match.range.upperBound)

            // Append separator, supporting capture groups
            for r in (0..<match.numberOfRanges).reversed() {
                let matchRange = match.range(at: r)
                if let sepRange = Range(matchRange, in:self) {
                    result.append(String(self[sepRange]))
                    break
                }
            }
        }

        // Append remaining suffix
        let beginningOfEnd = index(startIndex, offsetBy: matches.last!.range.upperBound)
        if beginningOfEnd < endIndex {
            result.append(String(self[beginningOfEnd...]))
        }

        return result
    }
}

public enum SplitDelimiterBehavior {
    case removed
    case isolated
    case mergedWithPrevious
    case mergedWithNext
}

public extension String {
    func split(by string: String, options: CompareOptions = .regularExpression, behavior: SplitDelimiterBehavior) -> [String] {
        func mergedWithNext(ranges: [Range<String.Index>]) -> [Range<String.Index>] {
            var merged: [Range<String.Index>] = []
            var currentStart = startIndex
            for range in ranges {
                if range.lowerBound == startIndex { continue }
                let mergedRange = currentStart..<range.lowerBound
                currentStart = range.lowerBound
                merged.append(mergedRange)
            }
            if currentStart < endIndex {
                merged.append(currentStart..<endIndex)
            }
            return merged
        }
        
        func mergedWithPrevious(ranges: [Range<String.Index>]) -> [Range<String.Index>] {
            var merged: [Range<String.Index>] = []
            var currentStart = startIndex
            for range in ranges {
                let mergedRange = currentStart..<range.upperBound
                currentStart = range.upperBound
                merged.append(mergedRange)
            }
            if currentStart < endIndex {
                merged.append(currentStart..<endIndex)
            }
            return merged
        }

        switch behavior {
        case .removed:
            return split(by: string, options: options, includeSeparators: false)
        case .isolated:
            return split(by: string, options: options, includeSeparators: true)
        case .mergedWithNext:
            // Obtain ranges and merge them
            // "the-final--countdown" -> (3, 4), (9, 10), (10, 11) -> (start, 2), (3, 8), (9, 9), (10, end)
            let ranges = ranges(of: string, options: options)
            let merged = mergedWithNext(ranges: ranges)
            return merged.map { String(self[$0]) }
        case .mergedWithPrevious:
            // Obtain ranges and merge them
            // "the-final--countdown" -> (3, 4), (9, 10), (10, 11) -> (start, 3), (4, 9), (10, 10), (11, end)
            let ranges = ranges(of: string, options: options)
            let merged = mergedWithPrevious(ranges: ranges)
            return merged.map { String(self[$0]) }
        }
    }
}


public protocol TokenDecoder {
    func decode(tokens: [String]) -> [String]
    func callAsFunction(tokens: [String]) -> [String]
    
    init(config: Config)
}

extension TokenDecoder {
    func callAsFunction(tokens: [String]) -> [String] {
        return decode(tokens: tokens)
    }
}

enum DecoderType: String {
    case Sequence
//    case WordPiece
    case ByteLevel
    case Replace
    case ByteFallback
    case Fuse
    case Strip
    case Metaspace
    case Unknown = ""
}

struct DecoderFactory {
    static func fromConfig(config: Config?, addedTokens: Set<String>? = nil) -> TokenDecoder? {
        // TODO: not sure if we need to include `addedTokens` in all the decoder initializers (and the protocol)
        guard let config = config else { return nil }
        guard let typeName = config.type?.stringValue else { return nil }
        let type = DecoderType(rawValue: typeName)
        switch type {
        case .Sequence    : return DecoderSequence(config: config)
        case .ByteLevel   : return ByteLevelDecoder(config: config, addedTokens: addedTokens)
        case .Replace     : return ReplaceDecoder(config: config)
        case .ByteFallback: return ByteFallbackDecoder(config: config)
        case .Fuse        : return FuseDecoder(config: config)
        case .Strip       : return StripDecoder(config: config)
        case .Metaspace   : return MetaspaceDecoder(config: config)
        default           : fatalError("Unsupported Decoder type: \(typeName)")
        }
    }
}

class DecoderSequence: TokenDecoder {
    let decoders: [TokenDecoder]
    
    required public init(config: Config) {
        guard let configs = config.decoders?.arrayValue else { fatalError("No decoders in Sequence") }
        decoders = configs.compactMap { DecoderFactory.fromConfig(config: $0) }
    }
    
    func decode(tokens: [String]) -> [String] {
        decoders.reduce(tokens) { current, decoder in
            decoder(tokens: current)
        }
    }
}

class ByteLevelDecoder: TokenDecoder {
    let addedTokens: Set<String>
    
    required public init(config: Config) {
        self.addedTokens = []
    }
    
    init(config: Config, addedTokens: Set<String>?) {
        self.addedTokens = addedTokens ?? []
    }
    
    func decode(tokens: [String]) -> [String] {
        var subTexts: [String] = []
        var currentSubText: [String] = []
        
        func convertTokensToString(_ tokens: [String]) -> String {
            let text = tokens.joined(separator: "")
            
            let utfCodepoints = text.map { byteDecoder[String($0)]! }
            return String(decoding: utfCodepoints, as: UTF8.self)
        }
        
        for token in tokens {
            if addedTokens.contains(token) {
                if !currentSubText.isEmpty {
                    subTexts.append(convertTokensToString(currentSubText))
                    currentSubText = []
                }
                subTexts.append(token)
            } else {
                currentSubText.append(token)
            }
        }
        
        if !currentSubText.isEmpty {
            subTexts.append(convertTokensToString(currentSubText))
        }
        
        return subTexts
    }
}

class ReplaceDecoder: TokenDecoder {
    let pattern: StringReplacePattern?
    
    required public init(config: Config) {
        self.pattern = StringReplacePattern.from(config: config)
    }
    
    func decode(tokens: [String]) -> [String] {
        guard let pattern = pattern else { return tokens }
        return tokens.map { pattern.replace($0) }
    }
}

class ByteFallbackDecoder: TokenDecoder {
    required public init(config: Config) {}
    
    func decode(tokens: [String]) -> [String] {
        var newTokens: [String] = []
        var byteTokens: [Int] = []

        func parseByte(_ token: String) -> Int? {
            guard token.count == 6 && token.hasPrefix("<0x") && token.hasSuffix(">") else {
                return nil
            }
            let startIndex = token.index(token.startIndex, offsetBy: 3)
            let endIndex = token.index(token.startIndex, offsetBy: 5)
            return Int(token[startIndex..<endIndex], radix: 16)
        }
        
        for token in tokens {
            if let byte = parseByte(token) {
                byteTokens.append(byte)
            } else {
                if !byteTokens.isEmpty {
                    // decode as utf8 and append
                    let codeUnits = byteTokens.map { UTF8.CodeUnit($0) }
                    newTokens.append(String(decoding: codeUnits, as: UTF8.self))
                    byteTokens.removeAll()
                }
                newTokens.append(token)
            }
        }
        return newTokens
    }
}

class FuseDecoder: TokenDecoder {
    required public init(config: Config) {}
    
    func decode(tokens: [String]) -> [String] {
        [tokens.joined(separator: "")]
    }
}

class StripDecoder: TokenDecoder {
    let content: String
    let start: Int
    let stop: Int
    
    required public init(config: Config) {
        guard let content = config.content?.stringValue else { fatalError("Incorrect StripDecoder configuration: can't parse `content`.") }
        guard let start = config.start?.intValue else { fatalError("Incorrect StripDecoder configuration: can't parse `start`.") }
        guard let stop = config.stop?.intValue else { fatalError("Incorrect StripDecoder configuration: can't parse `stop`.") }
        self.content = content
        self.start = start
        self.stop = stop
    }
    
    func decode(tokens: [String]) -> [String] {
        tokens.map { token in
            token.trimmingFromStart(upto: start).trimmingFromEnd(upto: stop)
        }
    }
}

class MetaspaceDecoder: TokenDecoder {
    let addPrefixSpace: Bool
    let replacement: String
    
    required public init(config: Config) {
        addPrefixSpace = config.addPrefixSpace?.boolValue ?? false
        replacement = config.replacement?.stringValue ?? "_"
    }

    func decode(tokens: [String]) -> [String] {
        var replaced = tokens.map { token in
            token.replacingOccurrences(of: replacement, with: " ")
        }
        if addPrefixSpace && replaced.first?.starts(with: " ") ?? false {
            replaced[0].removeFirst()
        }
        return replaced
    }
}

// We could use firstIndex(where:), lastIndex(where:) for possibly better efficiency (and do both ends at once)
public extension String {
    func trimmingFromStart(character: Character = " ", upto: Int) -> String {
        var result = self
        var trimmed = 0
        while trimmed < upto && result.first == character {
            result.removeFirst()
            trimmed += 1
        }
        return result
    }

    func trimmingFromEnd(character: Character = " ", upto: Int) -> String {
        var result = self
        var trimmed = 0
        while trimmed < upto && result.last == character {
            result.removeLast()
            trimmed += 1
        }
        return result
    }
}

public protocol Normalizer {
    func normalize(text: String) -> String
    func callAsFunction(text: String) -> String
    
    init(config: Config)
}

extension Normalizer {
    func callAsFunction(text: String) -> String {
        return normalize(text: text)
    }
}

enum NormalizerType: String {
    case Sequence
    case Prepend
    case Replace
    case Lowercase
    case NFD
    case NFC
    case NFKD
    case NFKC
    case Bert
    case Precompiled
    case StripAccents
    case Unknown = ""
}

struct NormalizerFactory {
    static func fromConfig(config: Config?) -> Normalizer? {
        guard let config = config else { return nil }
        guard let typeName = config.type?.stringValue else { return nil }
        let type = NormalizerType(rawValue: typeName)
        switch type {
        case .Sequence: return NormalizerSequence(config: config)
        case .Prepend : return PrependNormalizer(config: config)
        case .Replace : return ReplaceNormalizer(config: config)
        case .Lowercase : return LowercaseNormalizer(config: config)
        case .NFD : return NFDNormalizer(config: config)
        case .NFC : return NFCNormalizer(config: config)
        case .NFKD : return NFKDNormalizer(config: config)
        case .NFKC : return NFKCNormalizer(config: config)
        case .Bert : return BertNormalizer(config: config)
        case .Precompiled : return PrecompiledNormalizer(config: config)
        case .StripAccents : return StripAccentsNormalizer(config: config)
        default       : fatalError("Unsupported Normalizer type: \(typeName)")
        }
    }
}

class NormalizerSequence: Normalizer {
    let normalizers: [Normalizer]
    
    required public init(config: Config) {
        guard let configs = config.normalizers?.arrayValue else { fatalError("No normalizers in Sequence") }
        normalizers = configs.compactMap { NormalizerFactory.fromConfig(config: $0) }
    }
    
    public func normalize(text: String) -> String {
        normalizers.reduce(text) { current, normalizer in
            normalizer(text: current)
        }
    }
}

class PrependNormalizer: Normalizer {
    let prepend: String
    
    required public init(config: Config) {
        prepend = config.prepend?.stringValue ?? ""
    }
    
    public func normalize(text: String) -> String {
        return prepend + text
    }
}

class ReplaceNormalizer: Normalizer {
    let pattern: StringReplacePattern?
    
    required public init(config: Config) {
        self.pattern = StringReplacePattern.from(config: config)
    }
    
    public func normalize(text: String) -> String {
        guard let pattern = pattern else { return text }
        return pattern.replace(text)
    }
}

class LowercaseNormalizer: Normalizer {
    required public init(config: Config) {}

    public func normalize(text: String) -> String {
        text.lowercased()
    }
}

class NFDNormalizer: Normalizer {
    required public init(config: Config) {}

    public func normalize(text: String) -> String {
        text.decomposedStringWithCanonicalMapping
    }
}

class NFCNormalizer: Normalizer {
    required public init(config: Config) {}

    public func normalize(text: String) -> String {
        text.precomposedStringWithCanonicalMapping
    }
}

class NFKDNormalizer: Normalizer {
    required init(config: Config) {}

    func normalize(text: String) -> String {
        text.decomposedStringWithCompatibilityMapping
    }
}

class NFKCNormalizer: Normalizer {
    required init(config: Config) {}

    func normalize(text: String) -> String {
        text.precomposedStringWithCompatibilityMapping
    }
}

class BertNormalizer: Normalizer {
    let shouldCleanText: Bool
    let shouldHandleChineseChars: Bool
    let shouldStripAccents: Bool?
    let shouldLowercase: Bool

    required init(config: Config) {
        self.shouldCleanText = config.cleanText?.boolValue ?? true
        self.shouldHandleChineseChars = config.handleChineseChars?.boolValue ?? true
        self.shouldStripAccents = config.stripAccents?.boolValue
        self.shouldLowercase = config.lowercase?.boolValue ?? true
    }

    func normalize(text: String) -> String {
        var output = text
        if shouldCleanText {
            output = cleanText(text: output)
        }
        if shouldHandleChineseChars {
            output = handleChineseChars(text: output)
        }
        if shouldStripAccents ?? false {
            output = stripAccents(text: output)
        }
        if shouldLowercase {
            output = output.lowercased()
        }

        return output
    }

    private func cleanText(text: String) -> String {
        text.map { c in
            guard let scalar = c.unicodeScalars.first,
                  scalar.value != 0x0,
                  scalar.value != 0xFFFD,
                  !isControl(scalar)
            else { return "\(c)" }

            // Replace whitespace: \t, \n, \r
            if scalar.value == 0x009 ||
                scalar.value == 0x00A ||
                scalar.value == 0x000D {
                return " "
            } else {
                return "\(c)"
            }
        }
        .joined()
    }

    private func isControl(_ c: UnicodeScalar) -> Bool {
        if c.value == 0x009 || c.value == 0x00A || c.value == 0x000D {
            // Except \t, \n, \r that will be spaces.
            return false
        } else {
            // https://unicode.org/reports/tr44/#GC_Values_Table
            // Other Cc | Cf | Cs | Co | Cn
            return isOther(c.properties.generalCategory)
        }
    }

    private func isOther(_ c: Unicode.GeneralCategory) -> Bool {
        c == .control ||
        c == .format ||
        c == .surrogate ||
        c == .privateUse ||
        c == .unassigned
    }

    private func handleChineseChars(text: String) -> String {
        text.map { c in
            if let scalar = c.unicodeScalars.first, Self.isChineseChar(scalar) {
                " \(c) "
            } else {
               "\(c)"
            }
        }
        .joined()
    }
    
    /// Checks if a character is considered Chinese
    /// https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    static func isChineseChar(_ c: UnicodeScalar) -> Bool {
        (c.value >= 0x4E00 && c.value <= 0x9FFF) ||
        (c.value >= 0x3400 && c.value <= 0x4DBF) ||
        (c.value >= 0x20000 && c.value <= 0x2A6DF) ||
        (c.value >= 0x2A700 && c.value <= 0x2B73F) ||
        (c.value >= 0x2B740 && c.value <= 0x2B81F) ||
        (c.value >= 0x2B820 && c.value <= 0x2CEAF) ||
        (c.value >= 0xF900 && c.value <= 0xFAFF) ||
        (c.value >= 0x2F800 && c.value <= 0x2FA1F)
    }

    private func stripAccents(text: String) -> String {
        text.decomposedStringWithCanonicalMapping
            .filter { $0.unicodeScalars.allSatisfy { scalar in
                !(0x0300 <= scalar.value && scalar.value <= 0x036F)
            }}
    }
}

class PrecompiledNormalizer: Normalizer {
    // TODO: use `precompiledCharsmap` (base64-encoded string) from the configuration
    required init(config: Config) {}

    func normalize(text: String) -> String {
        // TODO: This is a simplified implementation.
        // - The following comments also apply here:
        // https://github.com/xenova/transformers.js/blob/main/src/tokenizers.js#L2237-L2247
        // - For a proper implementation, see:
        // https://github.com/huggingface/tokenizers/blob/b58227c7f1ccf8b73ee2268354336da56d91e492/tokenizers/src/normalizers/precompiled.rs#L36
        var output: String = ""
        var hasFullwidthTilde = false

        for scalar in text.unicodeScalars {
            switch scalar.value {
            case 0x0001...0x0008, 0x000B, 0x000E...0x001F, 0x007F, 0x008F, 0x009F:
                // Non-printing control characters
                output.append("")
            case 0x0009, 0x000A, 0x000C, 0x000D, 0x1680, 0x200B...0x200F, 0x2028, 0x2029, 0x2581, 0xFEFF, 0xFFFD:
                // Separators
                output.append(" ")
            case 0xFF5E:
                hasFullwidthTilde = true
                fallthrough
            default:
                output.append(Character(scalar))
            }
        }

        if hasFullwidthTilde {
            return output
                .split(by: "\u{FF5E}")
                .map({ $0.precomposedStringWithCompatibilityMapping })
                .joined(separator: "\u{FF5E}")
        } else {
            return output.precomposedStringWithCompatibilityMapping
        }
    }
}

class StripAccentsNormalizer: Normalizer {
    required init(config: Config) {}

    func normalize(text: String) -> String {
        text.precomposedStringWithCompatibilityMapping
    }
}

enum StringReplacePattern {
    case regexp(regexp: NSRegularExpression, replacement: String)
    case string(pattern: String, replacement: String)
}

extension StringReplacePattern {
    func replace(_ text: String) -> String {
        switch self {
        case .regexp(let regexp, let replacement):
            let range = NSRange(text.startIndex..., in: text)
            let replaced = regexp.stringByReplacingMatches(in: text, options: [], range: range, withTemplate: replacement)
            return replaced
        case .string(let toReplace, let replacement):
            return text.replacingOccurrences(of: toReplace, with: replacement)
        }
    }
}

extension StringReplacePattern {
    static func from(config: Config) -> StringReplacePattern? {
        guard let replacement = config.content?.stringValue else { return nil }
        if let pattern = config.pattern?.String?.stringValue {
            return StringReplacePattern.string(pattern: pattern, replacement: replacement)
        }
        if let pattern = config.pattern?.Regex?.stringValue {
            guard let regexp = try? NSRegularExpression(pattern: pattern, options: []) else {
                fatalError("Cannot build regexp from \(pattern)")
            }
            return StringReplacePattern.regexp(regexp: regexp, replacement: replacement)
        }
        return nil
    }
}

let byteEncoder: Dictionary<UTF8.CodeUnit, String> = [
    33: "!",
    34: "\"",
    35: "#",
    36: "$",
    37: "%",
    38: "&",
    39: "'",
    40: "(",
    41: ")",
    42: "*",
    43: "+",
    44: ",",
    45: "-",
    46: ".",
    47: "/",
    48: "0",
    49: "1",
    50: "2",
    51: "3",
    52: "4",
    53: "5",
    54: "6",
    55: "7",
    56: "8",
    57: "9",
    58: ":",
    59: ";",
    60: "<",
    61: "=",
    62: ">",
    63: "?",
    64: "@",
    65: "A",
    66: "B",
    67: "C",
    68: "D",
    69: "E",
    70: "F",
    71: "G",
    72: "H",
    73: "I",
    74: "J",
    75: "K",
    76: "L",
    77: "M",
    78: "N",
    79: "O",
    80: "P",
    81: "Q",
    82: "R",
    83: "S",
    84: "T",
    85: "U",
    86: "V",
    87: "W",
    88: "X",
    89: "Y",
    90: "Z",
    91: "[",
    92: "\\",
    93: "]",
    94: "^",
    95: "_",
    96: "`",
    97: "a",
    98: "b",
    99: "c",
    100: "d",
    101: "e",
    102: "f",
    103: "g",
    104: "h",
    105: "i",
    106: "j",
    107: "k",
    108: "l",
    109: "m",
    110: "n",
    111: "o",
    112: "p",
    113: "q",
    114: "r",
    115: "s",
    116: "t",
    117: "u",
    118: "v",
    119: "w",
    120: "x",
    121: "y",
    122: "z",
    123: "{",
    124: "|",
    125: "}",
    126: "~",
    161: "\u{00a1}",
    162: "\u{00a2}",
    163: "\u{00a3}",
    164: "\u{00a4}",
    165: "\u{00a5}",
    166: "\u{00a6}",
    167: "\u{00a7}",
    168: "\u{00a8}",
    169: "\u{00a9}",
    170: "\u{00aa}",
    171: "\u{00ab}",
    172: "\u{00ac}",
    174: "\u{00ae}",
    175: "\u{00af}",
    176: "\u{00b0}",
    177: "\u{00b1}",
    178: "\u{00b2}",
    179: "\u{00b3}",
    180: "\u{00b4}",
    181: "\u{00b5}",
    182: "\u{00b6}",
    183: "\u{00b7}",
    184: "\u{00b8}",
    185: "\u{00b9}",
    186: "\u{00ba}",
    187: "\u{00bb}",
    188: "\u{00bc}",
    189: "\u{00bd}",
    190: "\u{00be}",
    191: "\u{00bf}",
    192: "\u{00c0}",
    193: "\u{00c1}",
    194: "\u{00c2}",
    195: "\u{00c3}",
    196: "\u{00c4}",
    197: "\u{00c5}",
    198: "\u{00c6}",
    199: "\u{00c7}",
    200: "\u{00c8}",
    201: "\u{00c9}",
    202: "\u{00ca}",
    203: "\u{00cb}",
    204: "\u{00cc}",
    205: "\u{00cd}",
    206: "\u{00ce}",
    207: "\u{00cf}",
    208: "\u{00d0}",
    209: "\u{00d1}",
    210: "\u{00d2}",
    211: "\u{00d3}",
    212: "\u{00d4}",
    213: "\u{00d5}",
    214: "\u{00d6}",
    215: "\u{00d7}",
    216: "\u{00d8}",
    217: "\u{00d9}",
    218: "\u{00da}",
    219: "\u{00db}",
    220: "\u{00dc}",
    221: "\u{00dd}",
    222: "\u{00de}",
    223: "\u{00df}",
    224: "\u{00e0}",
    225: "\u{00e1}",
    226: "\u{00e2}",
    227: "\u{00e3}",
    228: "\u{00e4}",
    229: "\u{00e5}",
    230: "\u{00e6}",
    231: "\u{00e7}",
    232: "\u{00e8}",
    233: "\u{00e9}",
    234: "\u{00ea}",
    235: "\u{00eb}",
    236: "\u{00ec}",
    237: "\u{00ed}",
    238: "\u{00ee}",
    239: "\u{00ef}",
    240: "\u{00f0}",
    241: "\u{00f1}",
    242: "\u{00f2}",
    243: "\u{00f3}",
    244: "\u{00f4}",
    245: "\u{00f5}",
    246: "\u{00f6}",
    247: "\u{00f7}",
    248: "\u{00f8}",
    249: "\u{00f9}",
    250: "\u{00fa}",
    251: "\u{00fb}",
    252: "\u{00fc}",
    253: "\u{00fd}",
    254: "\u{00fe}",
    255: "\u{00ff}",
    0: "\u{0100}",
    1: "\u{0101}",
    2: "\u{0102}",
    3: "\u{0103}",
    4: "\u{0104}",
    5: "\u{0105}",
    6: "\u{0106}",
    7: "\u{0107}",
    8: "\u{0108}",
    9: "\u{0109}",
    10: "\u{010a}",
    11: "\u{010b}",
    12: "\u{010c}",
    13: "\u{010d}",
    14: "\u{010e}",
    15: "\u{010f}",
    16: "\u{0110}",
    17: "\u{0111}",
    18: "\u{0112}",
    19: "\u{0113}",
    20: "\u{0114}",
    21: "\u{0115}",
    22: "\u{0116}",
    23: "\u{0117}",
    24: "\u{0118}",
    25: "\u{0119}",
    26: "\u{011a}",
    27: "\u{011b}",
    28: "\u{011c}",
    29: "\u{011d}",
    30: "\u{011e}",
    31: "\u{011f}",
    32: "\u{0120}",
    127: "\u{0121}",
    128: "\u{0122}",
    129: "\u{0123}",
    130: "\u{0124}",
    131: "\u{0125}",
    132: "\u{0126}",
    133: "\u{0127}",
    134: "\u{0128}",
    135: "\u{0129}",
    136: "\u{012a}",
    137: "\u{012b}",
    138: "\u{012c}",
    139: "\u{012d}",
    140: "\u{012e}",
    141: "\u{012f}",
    142: "\u{0130}",
    143: "\u{0131}",
    144: "\u{0132}",
    145: "\u{0133}",
    146: "\u{0134}",
    147: "\u{0135}",
    148: "\u{0136}",
    149: "\u{0137}",
    150: "\u{0138}",
    151: "\u{0139}",
    152: "\u{013a}",
    153: "\u{013b}",
    154: "\u{013c}",
    155: "\u{013d}",
    156: "\u{013e}",
    157: "\u{013f}",
    158: "\u{0140}",
    159: "\u{0141}",
    160: "\u{0142}",
    173: "\u{0143}",
]

/// Invert a (k, v) dictionary
func invert<K, V>(_ dict: Dictionary<K, V>) -> Dictionary<V, K> {
    var inverted: [V: K] = [:]
    for (k, v) in dict {
        inverted[v] = k
    }
    return inverted
}

let byteDecoder = invert(byteEncoder)
