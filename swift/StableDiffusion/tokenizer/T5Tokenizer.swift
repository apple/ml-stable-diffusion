// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2024 Apple Inc. All Rights Reserved.

import Foundation
import Hub
import Tokenizers

/// Extension to swift-transfomers Hub.swift to load local Config files
public extension Config {
    /// Assumes the file is already present at local url.
    /// `fileURL` is a complete local file path for the given model
    public init(fileURL: URL) throws  {
        let data = try Data(contentsOf: fileURL)
        let parsed = try JSONSerialization.jsonObject(with: data, options: [])
        guard var dictionary = parsed as? [String: Any] else { throw Hub.HubClientError.parse }
        
        // Necessary override for loading local tokenizer configs
        dictionary["tokenizer_class"] = "T5Tokenizer"
        self.init(dictionary)
    }
}
