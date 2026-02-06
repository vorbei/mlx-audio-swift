//
//  Qwen3ASR.swift
//  MLXAudioSTT
//
// Created by Prince Canuma on 06/02/2026.
//

import Foundation
import MLX
import MLXNN
import MLXAudioCore
import MLXLMCommon
import HuggingFace
import Tokenizers


private func floorDiv(a: MLXArray, b: Int) -> MLXArray {
    return floor(a.asType(.float32) / Float(b)).asType(.int32)
}

private function getFeatExtractOutputLength(inputLengths: MLXArray) -> Int {
    // Comput output length of the conv layers
    let inputLethsLeave = inputLengths % 100
    let featLengths = floorDiv(inputLengthsLeave - 1, 2) + 1
    let outputLengths = (
        floorDiv(floorDiv(featLengths - 1, 2) + 1 - 1, 2)
        + 1
        + Float(inputLengths / 100) * 13
    )

    return outputLengths.asInt()


// MARK: - SinusodialPE

class Qwen3ASRSinusoidalPE: Module {
    let dim: Int
    let scale: Float

    init(length: Int, channels: Int, maxTimescale: Float = 10000.0) {
        if (channels % 2 != 0) {
            fatalError("SinsoidalPE channels must be even")
        }
        let logTimescaleIncrement = log(maxTimescale) / Float(channels / 2 - 1)

        let invTimescales = exp(
            -logTimescaleIncrement * MLXArray(0..<(channels / 2)).asType(.float32)
        )

        let positions = MLXArray(0..<length).asType(.float32).reshaped([-1, 1])

        let scaledTime = positions * invTimescales.reshaped([1, -1])
        self._positionalEmbeddings.wrappedValue = MX.concatenated(
            [MX.sin(scaledTime), MX.cos(scaledTime)], axis: 1
        )
    }

    func callAsFunction(_ seqLen: Int) -> MLXArray {
        return positionalEmbeddings[:seqLen, :]
    }
}

// MARK: - Attention

class Qwen3ASRAttention: Module {
    let config: Qwen3ASRConfig
    let scale: Float


    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear



    init(config: Qwen3ASRConfig) {
        self.config = config
        let dim = config.textConfig.dModel
        let numHeads = config.textConfig.numAttentionHeads
        let headDim = dim / numHeads
        self.scale = pow(Float(headDim), -0.5)

        if (headDim * numHeads != dim) {
            fatalError("embedDim must be divisible by numAttentionHeads got embedDim: \(dim) and numHeads: \(numHeads)")
        }

        self._wq.wrappedValue = Linear(dim, dim, bias: true)
        self._wk.wrappedValue = Linear(dim, dim, bias: false)
        self._wv.wrappedValue = Linear(dim, dim, bias: true)
        self._wo.wrappedValue = Linear(dim, dim, bias: true)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = wq(x)
        var keys = wk(x)
        var values = wv(x)

        queries = queries.reshaped(B, L, config.textConfig.numAttentionHeads, headDim).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, config.textConfig.numAttentionHeads, headDim).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, config.textConfig.numAttentionHeads, headDim).transposed(0, 2, 1, 3)

        let attn = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask
        )
        attn = attn.transposed(0, 2, 1).reshaped(B, L, dim)
        return wo(attn)
    }
}


class Qwen3ASRAudioEncoderLayer: Module {
    let config: Qwen3ASRConfig

    @ModuleInfo(key: "self_attn") var selfAttn: Qwen3ASRAttention
    @ModuleInfo(key: "self_attn_layer_norm") var selfAttnLayerNorm: LayerNorm
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: LayerNorm

    init(config: Qwen3ASRConfig) {
        self.config = config
        self._selfAttn.wrappedValue = Qwen3ASRAttention(config: config)
        self._selfAttnLayerNorm.wrappedValue = LayerNorm(dimensions: config.textConfig.dModel)
        self._fc1.wrappedValue = Linear(config.textConfig.dModel, config.textConfig.dModel * 4)
        self._fc2.wrappedValue = Linear(config.textConfig.dModel * 4, config.textConfig.dModel)
        self._finalLayerNorm.wrappedValue = LayerNorm(dimensions: config.textConfig.dModel)
    }

    func callAsFunction(_ h: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        var r = h
        h = selfAttnLayerNorm(h)
        h = selfAttn(h, mask: mask)
        h = r + h

        r = h
        r = finalLayerNorm(h)
        r = fc2(gelu(fc1(r)))
        h = r + h
        return finalLayerNorm(h)
    }
}

class Qwen3ASRAudioEncoder: Module {
    let config: Qwen3ASRAudioEncoderConfig

    @ModuleInfo(key: "conv2d1") var conv2d1: Conv2D
    @ModuleInfo(key: "conv2d2") var conv2d2: Conv2D
    @ModuleInfo(key: "conv2d3") var conv2d3: Conv2D
    @ModuleInfo(key: "conv_out") var convOut: Conv2D
    @ModuleInfo(key: "positional_embeddings") var positionalEmbeddings: Qwen3ASRSinusoidalPE
    @ModuleInfo(key: "layers") var layers: [Qwen3ASRAudioEncoderLayer]
    @ModuleInfo(key: "ln_post") var lnPost: LayerNorm
    @ModuleInfo(key: "proj1") var proj1: Linear
    @ModuleInfo(key: "proj2") var proj2: Linear

    init(config: Qwen3ASRConfig) {
        self.config = config
        let embedDim = config.dModel
        let numMelBins = config.numMelBins
        let maxSourcePositions = config.maxSourcePositions
        let embedScale = config.scaleEmbedding ? sqrt(Float(embedDim)) : 1.0
        let nWindow = config.nWindow
        let nWindowInfer = config.nWindowInfer
        let convChunksize = config.convChunkSize

        self._conv2d1.wrappedValue = Conv2D(
            inputChannels: config.audioConfig.numMelBins,
            outputChannels: config.textConfig.dModel,
            kernelSize: (3, 3),
            stride: (2, 2),
            padding: (1, 1)
        )
        self._conv2d2.wrappedValue = Conv2D(
            inputChannels: config.textConfig.dModel,
            outputChannels: config.textConfig.dModel,
            kernelSize: (3, 3),
            stride: (2, 2),
            padding: (1, 1)
        )
        self._conv2d3.wrappedValue = Conv2D(
            inputChannels: config.textConfig.dModel,
            outputChannels: config.textConfig.dModel,
            kernelSize: (3, 3),
            stride: (2, 2),
            padding: (1, 1)
        )
        self._convOut.wrappedValue = Conv2D(
            inputChannels: config.textConfig.dModel,
            outputChannels: config.textConfig.dModel,
            kernelSize: (3, 3),
            stride: (2, 2),
            padding: (1, 1)
        )
        self._positionalEmbeddings.wrappedValue = Qwen3ASRSinusoidalPE(length: config.textConfig.maxSeqLen, channels: config.textConfig.dModel)
        self._layers.wrappedValue = [Qwen3ASRAudioEncoderLayer(config: config)]
        self._lnPost.wrappedValue = LayerNorm(dimensions: config.textConfig.dModel)
        self._proj1.wrappedValue = Linear(config.textConfig.dModel, config.textConfig.dModel)
        self._proj2.wrappedValue = Linear(config.textConfig.dModel, config.textConfig.dModel)
    }

    private function createBlockAttentionMask(seqLen: Int cuSeqLens: Int, dtype: MLXDataType) -> MLXArray {
        let mask = MX.full(shape: (seqLen, seqLen), fillValue: -1e9, dtype: dtype)
        var start = 0
        for i in 0..<(curSeqLen-1) {
            start = curSeqLens[i]
            end = curSeqLens[i+1]
            mask[start:end, start:end] = 0.0
        }
        return mask
    }

    func callAsFunction(_ inputFeatures: MLXArray, featureAttentionMask: MLXArray? = nil) -> MLXArray {
        if featureAttentionMask != nil {
            let featureLens = featureAttentionMask.sum(axis: -1).astype(MLX.int32)
        } else {
            let featureLens = MX.array([inputFeatures.shape[-1]] * inputFeatures.shape[0], dtype: MLX.int32)
        }

        let featureLensNP = MX.array(featureLens)
        let aftercnnLens = getFeatExtractOutputLength(inputLengths: featureLensNP)
        let chunkSize = self.nWindow * 2
        let chunkNum = featureLensNP / chunkSize

        let chunkLengths: [Int] = []
        var numChunks = 0
        var featureLen = 0
        var remainder = 0
        for i in 0..<featureLensNP {
            numChunks = chunkNum[i]
            featureLen = featureLensNP[i]
            for j in 0..<numChunks {
                if j == numChunks - 1 {
                    remainder = featureLen % chunkSize
                    chunkLengths.append(remainder == 0 ? chunkSize : remainder)
                } else {
                    chunkLengths.append(chunkSize)
                }
            }
        }
        let chunks: [Int] = []
        var feat: MLXArray
        var featLen: Int
        var clen: Int
        var numChunks: Int
        for i in 0..<featureLensNP.count {
            feat = inputFeatures[i]
            featLen = Int(featureLensNP[i])
            numChunks = Int(chunkNum[i])
            var pos = 0
            for j in 0..<numChunks {
                if j == numChunks - 1 {
                    let remainder = featLen % chunkSize
                    clen = remainder == 0 ? chunkSize : remainder
                } else {
                    clen = chunkSize
                }
                let chunk = feat[0..., pos..<(pos + clen)]
                chunks.append(chunk)
                pos += clen
            }
        }

        let maxChunkLen = Int(chunkLengths.max()!)

        var paddedChunks: [MLXArray] = []

        for i in 0..<chunks.count {
            let chunk = chunks[i]
            let clen = Int(chunkLengths[i])
            var paddedChunk = chunk
            if clen < maxChunkLen {
                let padWidth = maxChunkLen - clen
                paddedChunk = MX.pad(chunk, [(0, 0), (0, padWidth)])
            }
            paddedChunks.append(paddedChunk)
        }

        let paddedFeature = MX.stack(paddedChunks, axis: 0)

        let featureLensAfterCnn = getFeatExtractOutputLength(
            inputLengths: MX.array(chunkLengths)
        )
        let featureLensAfterCnnNP = MX.array(featureLensAfterCnn)
        let maxLenAfterCnn = Int(featureLensAfterCnnNP.max()!)

        var paddedMaskAfterCnn = [[Bool]](
            repeating: [Bool](repeating: false, count: maxLenAfterCnn),
            count: chunkLengths.count
        )
        for i in 0..<featureLensAfterCnnNP.count {
            let length = Int(featureLensAfterCnnNP[i])
            for j in 0..<length {
                paddedMaskAfterCnn[i][j] = true
            }
        }

        var x = paddedFeature[0..., 0..., 0..., .newAxis]
        x = MX.gelu(self.conv2d1(x))
        x = MX.gelu(self.conv2d2(x))
        x = MX.gelu(self.conv2d3(x))

        let b = x.shape[0]
        let f = x.shape[1]
        let t = x.shape[2]
        let c = x.shape[3]
        x = x.transposed(0, 2, 3, 1).reshaped(b, t, c * f)
        x = self.convOut(x)

        let posEmb = self.positionalEmbedding(x.shape[1])
        x = x + posEmb[.newAxis, 0..., 0...]

        var hiddenList: [MLXArray] = []
        for i in 0..<x.shape[0] {
            let validLen = Int(featureLensAfterCnnNP[i])
            hiddenList.append(x[i, 0..<validLen])
        }

        var hiddenStates = MX.concatenated(hiddenList, axis: 0)

        let aftercnnLensNP = MX.array(aftercnnLens)
        let windowAftercnn = maxLenAfterCnn * (
            self.nWindowInfer / (self.nWindow * 2)
        )

        var cuChunkLens: [Int] = [0]
        for i in 0..<aftercnnLensNP.count {
            let cnnLen = Int(aftercnnLensNP[i])
            let numFullWindows = cnnLen / windowAftercnn
            for _ in 0..<numFullWindows {
                cuChunkLens.append(windowAftercnn)
            }
            let remainder = cnnLen % windowAftercnn
            if remainder != 0 {
                cuChunkLens.append(remainder)
            }
        }

        var cuSeqlens: [Int] = []
        var cumSum = 0
        for len in cuChunkLens {
            cumSum += len
            cuSeqlens.append(cumSum)
        }

        let seqLen = hiddenStates.shape[0]
        var attentionMask = self.createBlockAttentionMask(
            seqLen, cuSeqlens, hiddenStates.dtype
        )
        attentionMask = attentionMask[.newAxis, .newAxis, 0..., 0...]

        hiddenStates = hiddenStates[.newAxis, 0..., 0...]

        for layer in self.layers {
            hiddenStates = layer(hiddenStates, mask: attentionMask)
        }

        hiddenStates = hiddenStates[0]
        hiddenStates = self.lnPost(hiddenStates)
        hiddenStates = MX.gelu(self.proj1(hiddenStates))
        hiddenStates = self.proj2(hiddenStates)

        return hiddenStates
    }
}

class TextAttention: Module {
    /// Multi-headed attention for text decoder with Q/K norms.

    let config: Qwen3TextConfig
    let layerIdx: Int
    let hiddenSize: Int
    let numHeads: Int
    let numKvHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm
    @ModuleInfo(key: "rope") var rope: RoPE
    let rope: RoPE

    init(_ config: Qwen3TextConfig, layerIdx: Int) {
        self.config = config
        self.layerIdx = layerIdx
        self.hiddenSize = config.hiddenSize
        self.numHeads = config.numAttentionHeads
        self.numKvHeads = config.numKeyValueHeads
        self.headDim = config.headDim
        self.scale = pow(Float(self.headDim), -0.5)

        self.qProj = Linear(config.hiddenSize, self.numHeads * self.headDim, bias: false)
        self.kProj = Linear(config.hiddenSize, self.numKvHeads * self.headDim, bias: false)
        self.vProj = Linear(config.hiddenSize, self.numKvHeads * self.headDim, bias: false)
        self.oProj = Linear(self.numHeads * self.headDim, config.hiddenSize, bias: false)

        self.qNorm = RMSNorm(dimensions: self.headDim, eps: config.rmsNormEps)
        self.kNorm = RMSNorm(dimensions: self.headDim, eps: config.rmsNormEps)
        self.rope = RoPE(dimensions: self.headDim, traditional: false, base: config.ropeTheta)
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        cache: KVCache? = nil
    ) -> MLXArray {
        let B = hiddenStates.shape[0]
        let L = hiddenStates.shape[1]

        var queries = qProj(hiddenStates)
        var keys = kProj(hiddenStates)
        var values = vProj(hiddenStates)

        queries = queries.reshaped(B, L, numHeads, headDim)
        keys = keys.reshaped(B, L, numKvHeads, headDim)
        values = values.reshaped(B, L, numKvHeads, headDim)

        queries = qNorm(queries)
        keys = kNorm(keys)

        queries = queries.transposed(0, 2, 1, 3)
        keys = keys.transposed(0, 2, 1, 3)
        values = values.transposed(0, 2, 1, 3)

        let offset: Int
        if let cache = cache {
            offset = cache.offset
            queries = rope(queries, offset: offset)
            keys = rope(keys, offset: offset)
        } else {
            offset = 0
            queries = rope(queries)
            keys = rope(keys)
        }

        if let cache = cache {
            (keys, values) = cache.update(keys: keys, values: values)
        }

        let queryLen = queries.shape[2]
        let mask = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: createAdditiveCausalMask(queryLen, offset: offset).asType(queries.dtype)
        )

        let output = mask.transposed(0, 2, 1, 3).reshaped(B, queryLen, -1)
        return oProj(output)
    }
}

class TextMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(config: TextConfig) {
        self.hiddenSize = config.hiddenSize
        self.intermediateSize = config.intermediateSize
        self.gateProj = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        self.upProj = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        self.downProj = Linear(config.intermediateSize, config.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return downProj(silu(gateProj(x)) * upProj(x))
    }
}

