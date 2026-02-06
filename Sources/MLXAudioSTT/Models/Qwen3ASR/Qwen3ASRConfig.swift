
import MLXLMCommon

import Foundation

public struct AudioEncoderConfig: Codable {
    numMelBins: Int
    encoderLayers: Int
    encoderAttentionHeads: Int
    encoderFfnDim: Int
    dModel: Int
    dropout: Float
    attentionDropout: Float
    activationFunction: String
    activationDropout: Float
    scaleEmbedding: Bool
    maxSourcePositions: Int
    nWindow: Int
    outputDim: Int
    nWindowInfer: Int
    convChunkSize: Int
    downsampleHiddenSize: Int

    // Coding keys
    enum CodingKeys: String, CodingKey {
        case numMelBins = "num_mel_bins"
        case encoderLayers = "encoder_layers"
        case encoderAttentionHeads = "encoder_attention_heads"
        case encoderFfnDim = "encoder_ffn_dim"
        case dModel = "d_model"
        case dropout = "dropout"
        case attentionDropout = "attention_dropout"
        case activationFunction = "activation_function"
        case activationDropout = "activation_dropout"
        case scaleEmbedding = "scale_embedding"
        case maxSourcePositions = "max_source_positions"
        case nWindow = "n_window"
        case outputDim = "output_dim"
        case nWindowInfer = "n_window_infer"
        case convChunkSize = "conv_chunk_size"
        case downsampleHiddenSize = "downsample_hidden_size"
    }

    // Initializer
    public init(
        numMelBins: Int = 128,
        encoderLayers: Int = 12,
        encoderAttentionHeads: Int = 12,
        encoderFfnDim: Int = 128,
        dModel: Int = 128,
        dropout: Float = 0.0,
        attentionDropout: Float = 0.0,
        activationFunction: String = "gelu",
        activationDropout: Float = 0.0,
        scaleEmbedding: Bool = false,
        maxSourcePositions: Int = 1500,
        nWindow: Int = 10,
        outputDim: Int = 128,
        nWindowInfer: Int = 10,
        convChunkSize: Int = 10,
        downsampleHiddenSize: Int = 128
    ) {
        self.numMelBins = numMelBins
        self.encoderLayers = encoderLayers
        self.encoderAttentionHeads = encoderAttentionHeads
        self.encoderFfnDim = encoderFfnDim
        self.dModel = dModel
        self.dropout = dropout
        self.attentionDropout = attentionDropout
        self.activationFunction = activationFunction
        self.activationDropout = activationDropout
        self.scaleEmbedding = scaleEmbedding
        self.maxSourcePositions = maxSourcePositions
        self.nWindow = nWindow
        self.outputDim = outputDim
        self.nWindowInfer = nWindowInfer
        self.convChunkSize = convChunkSize
        self.downsampleHiddenSize = downsampleHiddenSize
    }

    // Decoder
    public init(from decoder: Swift.Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        numMelBins = try container.decodeIfPresent(Int.self, forKey: .numMelBins) ?? 128
        encoderLayers = try container.decodeIfPresent(Int.self, forKey: .encoderLayers) ?? 12
        encoderAttentionHeads = try container.decodeIfPresent(Int.self, forKey: .encoderAttentionHeads) ?? 12
        encoderFfnDim = try container.decodeIfPresent(Int.self, forKey: .encoderFfnDim) ?? 128
        dModel = try container.decodeIfPresent(Int.self, forKey: .dModel) ?? 128
        dropout = try container.decodeIfPresent(Float.self, forKey: .dropout) ?? 0.0
        attentionDropout = try container.decodeIfPresent(Float.self, forKey: .attentionDropout) ?? 0.0
        activationFunction = try container.decodeIfPresent(String.self, forKey: .activationFunction) ?? "gelu"
        activationDropout = try container.decodeIfPresent(Float.self, forKey: .activationDropout) ?? 0.0
        scaleEmbedding = try container.decodeIfPresent(Bool.self, forKey: .scaleEmbedding) ?? false
        maxSourcePositions = try container.decodeIfPresent(Int.self, forKey: .maxSourcePositions) ?? 1500
        nWindow = try container.decodeIfPresent(Int.self, forKey: .nWindow) ?? 10
        outputDim = try container.decodeIfPresent(Int.self, forKey: .outputDim) ?? 128
        nWindowInfer = try container.decodeIfPresent(Int.self, forKey: .nWindowInfer) ?? 10
        convChunkSize = try container.decodeIfPresent(Int.self, forKey: .convChunkSize) ?? 10
        downsampleHiddenSize = try container.decodeIfPresent(Int.self, forKey: .downsampleHiddenSize) ?? 128
    }

    // Encoder
    public func encode(to encoder: Swift.Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(numMelBins, forKey: .numMelBins)
        try container.encode(encoderLayers, forKey: .encoderLayers)
        try container.encode(encoderAttentionHeads, forKey: .encoderAttentionHeads)
        try container.encode(encoderFfnDim, forKey: .encoderFfnDim)
        try container.encode(dModel, forKey: .dModel)
        try container.encode(dropout, forKey: .dropout)
        try container.encode(attentionDropout, forKey: .attentionDropout)
        try container.encode(activationFunction, forKey: .activationFunction)
        try container.encode(activationDropout, forKey: .activationDropout)
        try container.encode(scaleEmbedding, forKey: .scaleEmbedding)
        try container.encode(maxSourcePositions, forKey: .maxSourcePositions)
        try container.encode(nWindow, forKey: .nWindow)
        try container.encode(outputDim, forKey: .outputDim)
        try container.encode(nWindowInfer, forKey: .nWindowInfer)
        try container.encode(convChunkSize, forKey: .convChunkSize)
        try container.encode(downsampleHiddenSize, forKey: .downsampleHiddenSize)
    }
}


public struct TextConfig: Codable {
    public var modelType: String
    public var vocabSize: Int
    public var hiddenSize: Int
    public var intermediateSize: Int
    public var numHiddenLayers: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var headDim: Int
    public var hiddenAct: String
    public var maxPositionEmbeddings: Int
    public var initializerRange: Float
    public var rmsNormEps: Float
    public var useCache: Bool
    public var tieWordEmbeddings: Bool
    public var ropeTheta: Float
    public var ropeScaling: [String: AnyCodable]?
    public var attentionBias: Bool
    public var attentionDropout: Float

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case hiddenAct = "hidden_act"
        case maxPositionEmbeddings = "max_position_embeddings"
        case initializerRange = "initializer_range"
        case rmsNormEps = "rms_norm_eps"
        case useCache = "use_cache"
        case tieWordEmbeddings = "tie_word_embeddings"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case attentionBias = "attention_bias"
        case attentionDropout = "attention_dropout"
    }

    public init(
        modelType: String = "qwen3",
        vocabSize: Int = 151936,
        hiddenSize: Int = 2048,
        intermediateSize: Int = 6144,
        numHiddenLayers: Int = 28,
        numAttentionHeads: Int = 16,
        numKeyValueHeads: Int = 8,
        headDim: Int = 128,
        hiddenAct: String = "silu",
        maxPositionEmbeddings: Int = 65536,
        initializerRange: Float = 0.02,
        rmsNormEps: Float = 1e-6,
        useCache: Bool = true,
        tieWordEmbeddings: Bool = true,
        ropeTheta: Float = 1000000.0,
        ropeScaling: [String: AnyCodable]? = nil,
        attentionBias: Bool = false,
        attentionDropout: Float = 0.0
    ) {
        self.modelType = modelType
        self.vocabSize = vocabSize
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.headDim = headDim
        self.hiddenAct = hiddenAct
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.initializerRange = initializerRange
        self.rmsNormEps = rmsNormEps
        self.useCache = useCache
        self.tieWordEmbeddings = tieWordEmbeddings
        self.ropeTheta = ropeTheta
        self.ropeScaling = ropeScaling
        self.attentionBias = attentionBias
        self.attentionDropout = attentionDropout
    }

    public init(from decoder: Swift.Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "qwen3"
        vocabSize = try container.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 151936
        hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 2048
        intermediateSize = try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 6144
        numHiddenLayers = try container.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 28
        numAttentionHeads = try container.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 16
        numKeyValueHeads = try container.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 8
        headDim = try container.decodeIfPresent(Int.self, forKey: .headDim) ?? 128
        hiddenAct = try container.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "silu"
        maxPositionEmbeddings = try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 65536
        initializerRange = try container.decodeIfPresent(Float.self, forKey: .initializerRange) ?? 0.02
        rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        useCache = try container.decodeIfPresent(Bool.self, forKey: .useCache) ?? true
        tieWordEmbeddings = try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
        ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1000000.0
        ropeScaling = try container.decodeIfPresent([String: AnyCodable].self, forKey: .ropeScaling)
        attentionBias = try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        attentionDropout = try container.decodeIfPresent(Float.self, forKey: .attentionDropout) ?? 0.0
    }

    public func encode(to encoder: Swift.Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(modelType, forKey: .modelType)
        try container.encode(vocabSize, forKey: .vocabSize)
        try container.encode(hiddenSize, forKey: .hiddenSize)
        try container.encode(intermediateSize, forKey: .intermediateSize)
        try container.encode(numHiddenLayers, forKey: .numHiddenLayers)
        try container.encode(numAttentionHeads, forKey: .numAttentionHeads)
        try container.encode(numKeyValueHeads, forKey: .numKeyValueHeads)
        try container.encode(headDim, forKey: .headDim)
        try container.encode(hiddenAct, forKey: .hiddenAct)
        try container.encode(maxPositionEmbeddings, forKey: .maxPositionEmbeddings)
        try container.encode(initializerRange, forKey: .initializerRange)
        try container.encode(rmsNormEps, forKey: .rmsNormEps)
        try container.encode(useCache, forKey: .useCache)
        try container.encode(tieWordEmbeddings, forKey: .tieWordEmbeddings)
        try container.encode(ropeTheta, forKey: .ropeTheta)
        try container.encodeIfPresent(ropeScaling, forKey: .ropeScaling)
        try container.encode(attentionBias, forKey: .attentionBias)
        try container.encode(attentionDropout, forKey: .attentionDropout)
    }
}

public struct Qwen3ASRConfig: Codable {
    public var audioConfig: Qwen3AudioEncoderConfig
    public var textConfig: Qwen3TextConfig
    public var modelType: String
    public var modelRepo: String?
    public var audioTokenId: Int
    public var audioStartTokenId: Int
    public var audioEndTokenId: Int
    public var supportLanguages: [String]

    enum CodingKeys: String, CodingKey {
        case audioConfig = "audio_config"
        case textConfig = "text_config"
        case modelType = "model_type"
        case modelRepo = "model_repo"
        case audioTokenId = "audio_token_id"
        case audioStartTokenId = "audio_start_token_id"
        case audioEndTokenId = "audio_end_token_id"
        case supportLanguages = "support_languages"
        case thinkerConfig = "thinker_config"
    }

    public init(
        audioConfig: Qwen3AudioEncoderConfig = Qwen3AudioEncoderConfig(),
        textConfig: Qwen3TextConfig = Qwen3TextConfig(),
        modelType: String = "qwen3_asr",
        modelRepo: String? = nil,
        audioTokenId: Int = 151676,
        audioStartTokenId: Int = 151669,
        audioEndTokenId: Int = 151670,
        supportLanguages: [String] = []
    ) {
        self.audioConfig = audioConfig
        self.textConfig = textConfig
        self.modelType = modelType
        self.modelRepo = modelRepo
        self.audioTokenId = audioTokenId
        self.audioStartTokenId = audioStartTokenId
        self.audioEndTokenId = audioEndTokenId
        self.supportLanguages = supportLanguages
    }

    public init(from decoder: Swift.Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        // Try to decode from thinker_config first (HF format)
        if let thinkerContainer = try? container.nestedContainer(keyedBy: CodingKeys.self, forKey: .thinkerConfig) {
            audioConfig = try thinkerContainer.decodeIfPresent(Qwen3AudioEncoderConfig.self, forKey: .audioConfig) ?? Qwen3AudioEncoderConfig()
            textConfig = try thinkerContainer.decodeIfPresent(Qwen3TextConfig.self, forKey: .textConfig) ?? Qwen3TextConfig()
            audioTokenId = try thinkerContainer.decodeIfPresent(Int.self, forKey: .audioTokenId) ?? 151676
            audioStartTokenId = try thinkerContainer.decodeIfPresent(Int.self, forKey: .audioStartTokenId) ?? 151669
            audioEndTokenId = try thinkerContainer.decodeIfPresent(Int.self, forKey: .audioEndTokenId) ?? 151670
        } else {
            audioConfig = try container.decodeIfPresent(Qwen3AudioEncoderConfig.self, forKey: .audioConfig) ?? Qwen3AudioEncoderConfig()
            textConfig = try container.decodeIfPresent(Qwen3TextConfig.self, forKey: .textConfig) ?? Qwen3TextConfig()
            audioTokenId = try container.decodeIfPresent(Int.self, forKey: .audioTokenId) ?? 151676
            audioStartTokenId = try container.decodeIfPresent(Int.self, forKey: .audioStartTokenId) ?? 151669
            audioEndTokenId = try container.decodeIfPresent(Int.self, forKey: .audioEndTokenId) ?? 151670
        }

        modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "qwen3_asr"
        modelRepo = try container.decodeIfPresent(String.self, forKey: .modelRepo)
        supportLanguages = try container.decodeIfPresent([String].self, forKey: .supportLanguages) ?? []
    }

    public func encode(to encoder: Swift.Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(audioConfig, forKey: .audioConfig)
        try container.encode(textConfig, forKey: .textConfig)
        try container.encode(modelType, forKey: .modelType)
        try container.encodeIfPresent(modelRepo, forKey: .modelRepo)
        try container.encode(audioTokenId, forKey: .audioTokenId)
        try container.encode(audioStartTokenId, forKey: .audioStartTokenId)
        try container.encode(audioEndTokenId, forKey: .audioEndTokenId)
        try container.encode(supportLanguages, forKey: .supportLanguages)
    }
}

// Helper type for decoding arbitrary JSON values
public struct AnyCodable: Codable {
    public let value: Any

    public init(_ value: Any) {
        self.value = value
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()

        if let bool = try? container.decode(Bool.self) {
            value = bool
        } else if let int = try? container.decode(Int.self) {
            value = int
        } else if let double = try? container.decode(Double.self) {
            value = double
        } else if let string = try? container.decode(String.self) {
            value = string
        } else if let array = try? container.decode([AnyCodable].self) {
            value = array.map { $0.value }
        } else if let dictionary = try? container.decode([String: AnyCodable].self) {
            value = dictionary.mapValues { $0.value }
        } else {
            value = NSNull()
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()

        switch value {
        case let bool as Bool:
            try container.encode(bool)
        case let int as Int:
            try container.encode(int)
        case let double as Double:
            try container.encode(double)
        case let string as String:
            try container.encode(string)
        case let array as [Any]:
            try container.encode(array.map { AnyCodable($0) })
        case let dictionary as [String: Any]:
            try container.encode(dictionary.mapValues { AnyCodable($0) })
        default:
            try container.encodeNil()
        }
    }
}

