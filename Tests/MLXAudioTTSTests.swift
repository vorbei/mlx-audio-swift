//
//  MLXAudioTTSTests.swift
//  MLXAudioTests
//
//  Created by Prince Canuma on 31/12/2025.
//

import Testing
import MLX
import MLXLMCommon
import Foundation

@testable import MLXAudioCore
@testable import MLXAudioTTS
@testable import MLXAudioCodecs


// Run Qwen3 tests with:  xcodebuild test \
// -scheme MLXAudio-Package \
// -destination 'platform=macOS' \
// -only-testing:MLXAudioTests/Qwen3TTSTests \
// 2>&1 | grep -E "(Suite.*started|Test test.*started|Loading|Loaded|Generating|Generated|Saved|passed after|failed after|TEST SUCCEEDED|TEST FAILED|Suite.*passed|Test run)"


struct Qwen3TTSTests {

    /// Test basic text-to-speech generation with Qwen3 model
    @Test func testQwen3Generate() async throws {
        // 1. Load Qwen3 model from HuggingFace
        print("\u{001B}[33mLoading Qwen3 TTS model...\u{001B}[0m")
        let model = try await Qwen3Model.fromPretrained("mlx-community/VyvoTTS-EN-Beta-4bit")
        print("\u{001B}[32mQwen3 model loaded!\u{001B}[0m")

        // 2. Generate audio from text
        let text = "Hello, this is a test of the Qwen3 text to speech model."
        print("\u{001B}[33mGenerating audio for: \"\(text)\"...\u{001B}[0m")

        let parameters = GenerateParameters(
            maxTokens: 500,
            temperature: 0.6,
            topP: 0.8,
            repetitionPenalty: 1.3,
            repetitionContextSize: 20
        )

        let audio = try await model.generate(
            text: text,
            voice: nil,
            parameters: parameters
        )

        print("\u{001B}[32mGenerated audio shape: \(audio.shape)\u{001B}[0m")

        // 3. Basic checks
        #expect(audio.shape[0] > 0, "Audio should have samples")

        // 4. Save generated audio
        let outputURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen3_test_output.wav")
        try saveAudioArray(audio, sampleRate: Double(model.sampleRate), to: outputURL)
        print("\u{001B}[32mSaved generated audio to\u{001B}[0m: \(outputURL.path)")
    }

    /// Test streaming generation with Qwen3 model
    @Test func testQwen3GenerateStream() async throws {
        // 1. Load Qwen3 model from HuggingFace
        print("\u{001B}[33mLoading Qwen3 TTS model...\u{001B}[0m")
        let model = try await Qwen3Model.fromPretrained("mlx-community/VyvoTTS-EN-Beta-4bit")
        print("\u{001B}[32mQwen3 model loaded!\u{001B}[0m")

        // 2. Generate audio with streaming
        let text = "Streaming test for Qwen3 model."
        print("\u{001B}[33mStreaming generation for: \"\(text)\"...\u{001B}[0m")

        let parameters = GenerateParameters(
            maxTokens: 300,
            temperature: 0.6,
            topP: 0.8,
            repetitionPenalty: 1.3,
            repetitionContextSize: 20
        )

        var tokenCount = 0
        var finalAudio: MLXArray?
        var generationInfo: Qwen3GenerationInfo?

        for try await event in model.generateStream(text: text, parameters: parameters) {
            switch event {
            case .token(_):
                tokenCount += 1
                if tokenCount % 50 == 0 {
                    print("  Generated \(tokenCount) tokens...")
                }
            case .info(let info):
                generationInfo = info
                print("\u{001B}[36m\(info.summary)\u{001B}[0m")
            case .audio(let audio):
                finalAudio = audio
                print("\u{001B}[32mReceived final audio: \(audio.shape)\u{001B}[0m")
            }
        }

        // 3. Verify results
        #expect(tokenCount > 0, "Should have generated tokens")
        #expect(finalAudio != nil, "Should have received final audio")
        #expect(generationInfo != nil, "Should have received generation info")

        if let audio = finalAudio {
            #expect(audio.shape[0] > 0, "Audio should have samples")

            // Save the audio
            let outputURL = FileManager.default.temporaryDirectory
                .appendingPathComponent("qwen3_stream_test_output.wav")
            try saveAudioArray(audio, sampleRate: Double(model.sampleRate), to: outputURL)
            print("\u{001B}[32mSaved streamed audio to\u{001B}[0m: \(outputURL.path)")
        }
    }


}


// Run LlamaTTS tests with:  xcodebuild test \
// -scheme MLXAudio-Package \
// -destination 'platform=macOS' \
// -only-testing:MLXAudioTests/LlamaTTSTests \
// 2>&1 | grep -E "(Suite.*started|Test test.*started|Loading|Loaded|Generating|Generated|Saved|passed after|failed after|TEST SUCCEEDED|TEST FAILED|Suite.*passed|Test run)"


struct LlamaTTSTests {

    /// Test basic text-to-speech generation with LlamaTTS model (Orpheus)
    @Test func testLlamaTTSGenerate() async throws {
        // 1. Load LlamaTTS model from HuggingFace
        print("\u{001B}[33mLoading LlamaTTS (Orpheus) model...\u{001B}[0m")
        let model = try await LlamaTTSModel.fromPretrained("mlx-community/orpheus-3b-0.1-ft-bf16")
        print("\u{001B}[32mLlamaTTS model loaded!\u{001B}[0m")

        // 2. Generate audio from text
        let text = "Hello, this is a test of the Orpheus text to speech model."
        print("\u{001B}[33mGenerating audio for: \"\(text)\"...\u{001B}[0m")

        let parameters = GenerateParameters(
            maxTokens: 800,
            temperature: 0.7,
            topP: 0.8,
            repetitionPenalty: 1.3,
            repetitionContextSize: 20
        )

        let audio = try await model.generate(
            text: text,
            voice: "tara",
            parameters: parameters
        )

        print("\u{001B}[32mGenerated audio shape: \(audio.shape)\u{001B}[0m")

        // 3. Basic checks
        #expect(audio.shape[0] > 0, "Audio should have samples")

        // 4. Save generated audio
        let outputURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("llama_tts_test_output.wav")
        try saveAudioArray(audio, sampleRate: Double(model.sampleRate), to: outputURL)
        print("\u{001B}[32mSaved generated audio to\u{001B}[0m: \(outputURL.path)")
    }

    /// Test streaming generation with LlamaTTS model (Orpheus)
    @Test func testLlamaTTSGenerateStream() async throws {
        // 1. Load LlamaTTS model from HuggingFace
        print("\u{001B}[33mLoading LlamaTTS (Orpheus) model...\u{001B}[0m")
        let model = try await LlamaTTSModel.fromPretrained("mlx-community/orpheus-3b-0.1-ft-bf16")
        print("\u{001B}[32mLlamaTTS model loaded!\u{001B}[0m")

        // 2. Generate audio with streaming
        let text = "Streaming test for Orpheus model."
        print("\u{001B}[33mStreaming generation for: \"\(text)\"...\u{001B}[0m")

        let parameters = GenerateParameters(
            maxTokens: 300,
            temperature: 0.6,
            topP: 0.8,
            repetitionPenalty: 1.3,
            repetitionContextSize: 20
        )

        var tokenCount = 0
        var finalAudio: MLXArray?
        var generationInfo: LlamaTTSGenerationInfo?

        for try await event in model.generateStream(text: text, voice: "tara", parameters: parameters) {
            switch event {
            case .token(_):
                tokenCount += 1
                if tokenCount % 50 == 0 {
                    print("  Generated \(tokenCount) tokens...")
                }
            case .info(let info):
                generationInfo = info
                print("\u{001B}[36m\(info.summary)\u{001B}[0m")
            case .audio(let audio):
                finalAudio = audio
                print("\u{001B}[32mReceived final audio: \(audio.shape)\u{001B}[0m")
            }
        }

        // 3. Verify results
        #expect(tokenCount > 0, "Should have generated tokens")
        #expect(finalAudio != nil, "Should have received final audio")
        #expect(generationInfo != nil, "Should have received generation info")

        if let audio = finalAudio {
            #expect(audio.shape[0] > 0, "Audio should have samples")

            // Save the audio
            let outputURL = FileManager.default.temporaryDirectory
                .appendingPathComponent("llama_tts_stream_test_output.wav")
            try saveAudioArray(audio, sampleRate: Double(model.sampleRate), to: outputURL)
            print("\u{001B}[32mSaved streamed audio to\u{001B}[0m: \(outputURL.path)")
        }
    }




}


// Run Soprano tests with:  xcodebuild test \
// -scheme MLXAudio-Package \
// -destination 'platform=macOS' \
// -only-testing:MLXAudioTests/SopranoTTSTests \
// 2>&1 | grep -E "(Suite.*started|Test test.*started|Loading|Loaded|Generating|Generated|Saved|passed after|failed after|TEST SUCCEEDED|TEST FAILED|Suite.*passed|Test run)"


struct SopranoTTSTests {

    /// Test basic text-to-speech generation with Soprano model
    @Test func testSopranoGenerate() async throws {
        // 1. Load Soprano model from HuggingFace
        print("\u{001B}[33mLoading Soprano TTS model...\u{001B}[0m")
        let model = try await SopranoModel.fromPretrained("mlx-community/Soprano-80M-bf16")
        print("\u{001B}[32mSoprano model loaded!\u{001B}[0m")

        // 2. Generate audio from text
        let text = "Performance Optimization: Automatic model quantization and hardware optimization that delivers 30%-100% faster inference than standard implementations."
        print("\u{001B}[33mGenerating audio for: \"\(text)\"...\u{001B}[0m")

        // Use temperature=0.0 for deterministic generation (same as hello world test)
        let parameters = GenerateParameters(
            maxTokens: 200,
            temperature: 0.3,
            topP: 0.95,
        )

        let audio = try await model.generate(
            text: text,
            voice: nil,
            parameters: parameters
        )

        print("\u{001B}[32mGenerated audio shape: \(audio.shape)\u{001B}[0m")

        // 3. Basic checks
        #expect(audio.shape[0] > 0, "Audio should have samples")

        // 4. Save generated audio
        let outputURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("soprano_test_output.wav")
        try saveAudioArray(audio, sampleRate: Double(model.sampleRate), to: outputURL)
        print("\u{001B}[32mSaved generated audio to\u{001B}[0m: \(outputURL.path)")
    }

    /// Test streaming generation with Soprano model
    @Test func testSopranoGenerateStream() async throws {
        // 1. Load Soprano model from HuggingFace
        print("\u{001B}[33mLoading Soprano TTS model...\u{001B}[0m")
        let model = try await SopranoModel.fromPretrained("mlx-community/Soprano-80M-bf16")
        print("\u{001B}[32mSoprano model loaded!\u{001B}[0m")

        // 2. Generate audio with streaming
        let text = "Streaming test for Soprano model. I think it's working."
        print("\u{001B}[33mStreaming generation for: \"\(text)\"...\u{001B}[0m")

        // Use temperature=0.0 for deterministic generation
        let parameters = GenerateParameters(
            maxTokens: 100,
            temperature: 0.3,
            topP: 1.0
        )

        var tokenCount = 0
        var finalAudio: MLXArray?
        var generationInfo: SopranoGenerationInfo?

        for try await event in model.generateStream(text: text, parameters: parameters) {
            switch event {
            case .token(_):
                tokenCount += 1
                if tokenCount % 50 == 0 {
                    print("  Generated \(tokenCount) tokens...")
                }
            case .info(let info):
                generationInfo = info
                print("\u{001B}[36m\(info.summary)\u{001B}[0m")
            case .audio(let audio):
                finalAudio = audio
                print("\u{001B}[32mReceived final audio: \(audio.shape)\u{001B}[0m")
            }
        }

        // 3. Verify results
        #expect(tokenCount > 0, "Should have generated tokens")
        #expect(finalAudio != nil, "Should have received final audio")
        #expect(generationInfo != nil, "Should have received generation info")

        if let audio = finalAudio {
            #expect(audio.shape[0] > 0, "Audio should have samples")

            // Save the audio
            let outputURL = FileManager.default.temporaryDirectory
                .appendingPathComponent("soprano_stream_test_output.wav")
            try saveAudioArray(audio, sampleRate: Double(model.sampleRate), to: outputURL)
            print("\u{001B}[32mSaved streamed audio to\u{001B}[0m: \(outputURL.path)")
        }
    }

    /// Test text cleaning utilities
    @Test func testTextCleaning() {
        // Test number normalization
        let text1 = "I have $100 and 50 cents."
        let cleaned1 = cleanTextForSoprano(text1)
        #expect(cleaned1.contains("one hundred dollars"), "Should expand dollar amounts")

        // Test abbreviations
        let text2 = "Dr. Smith went to the API conference."
        let cleaned2 = cleanTextForSoprano(text2)
        #expect(cleaned2.contains("doctor"), "Should expand Dr. to doctor")
        #expect(cleaned2.contains("a p i"), "Should expand API")

        // Test ordinals
        let text3 = "This is the 1st and 2nd test."
        let cleaned3 = cleanTextForSoprano(text3)
        #expect(cleaned3.contains("first"), "Should expand 1st to first")
        #expect(cleaned3.contains("second"), "Should expand 2nd to second")

        print("\u{001B}[32mText cleaning tests passed!\u{001B}[0m")
    }


}


// Run Kokoro tests with:  xcodebuild test \
// -scheme MLXAudio-Package \
// -destination 'platform=macOS' \
// -only-testing:MLXAudioTests/KokoroTTSTests \
// 2>&1 | grep -E "(Suite.*started|Test test.*started|Loading|Loaded|Generating|Generated|Saved|passed after|failed after|TEST SUCCEEDED|TEST FAILED|Suite.*passed|Test run)"


/// Thread-safe storage for audio chunks used in streaming tests
final class AudioChunkStorage: @unchecked Sendable {
    private var chunks: [MLXArray] = []
    private let lock = NSLock()

    var count: Int {
        lock.lock()
        defer { lock.unlock() }
        return chunks.count
    }

    func append(_ chunk: MLXArray) {
        lock.lock()
        defer { lock.unlock() }
        chunks.append(chunk)
    }

    func getAll() -> [MLXArray] {
        lock.lock()
        defer { lock.unlock() }
        return chunks
    }
}


struct KokoroTTSTests {

    /// Test basic text-to-speech generation with Kokoro model
    @Test func testKokoroGenerate() async throws {
        // 1. Load Kokoro model from HuggingFace
        print("\u{001B}[33mLoading Kokoro TTS model...\u{001B}[0m")
        let model = try await Kokoro.fromHub(repoId: kokoroDefaultRepo)
        print("\u{001B}[32mKokoro model loaded!\u{001B}[0m")

        // 2. Generate audio from text
        let text = "Hello, this is a test of the Kokoro text to speech model."
        print("\u{001B}[33mGenerating audio for: \"\(text)\"...\u{001B}[0m")

        let audio = try model.generateAudioForSentence(
            voice: .afBella,
            text: text,
            speed: 1.0
        )

        print("\u{001B}[32mGenerated audio shape: \(audio.shape)\u{001B}[0m")

        // 3. Basic checks
        #expect(audio.shape[0] > 0, "Audio should have samples")

        // 4. Save generated audio
        let outputURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("kokoro_test_output.wav")
        try saveAudioArray(audio, sampleRate: 24000, to: outputURL)
        print("\u{001B}[32mSaved generated audio to\u{001B}[0m: \(outputURL.path)")
    }

    /// Test streaming generation with Kokoro model using callback
    @Test func testKokoroGenerateStream() async throws {
        // 1. Load Kokoro model from HuggingFace
        print("\u{001B}[33mLoading Kokoro TTS model...\u{001B}[0m")
        let model = try await Kokoro.fromHub(repoId: kokoroDefaultRepo)
        print("\u{001B}[32mKokoro model loaded!\u{001B}[0m")

        // 2. Generate audio with streaming callback
        let text = "This is a streaming test for Kokoro. It should generate audio in chunks as each sentence is processed."
        print("\u{001B}[33mStreaming generation for: \"\(text)\"...\u{001B}[0m")

        // Use thread-safe storage for audio chunks
        let chunkStorage = AudioChunkStorage()

        try model.generateAudio(voice: .afBella, text: text, speed: 1.0) { chunk in
            chunkStorage.append(chunk)
            print("  Received audio chunk: \(chunk.shape)")
        }

        // Wait for the background task to process all sentences
        // The text has 2 sentences, so we wait enough time for processing
        try await Task.sleep(for: .seconds(10))

        // 3. Verify results
        let audioChunks = chunkStorage.getAll()
        #expect(audioChunks.count > 0, "Should have received at least one audio chunk")

        if let firstChunk = audioChunks.first {
            #expect(firstChunk.shape[0] > 0, "Audio chunk should have samples")

            // Save the first chunk
            let outputURL = FileManager.default.temporaryDirectory
                .appendingPathComponent("kokoro_stream_test_output.wav")
            try saveAudioArray(firstChunk, sampleRate: 24000, to: outputURL)
            print("\u{001B}[32mSaved first audio chunk to\u{001B}[0m: \(outputURL.path)")
        }

        print("\u{001B}[32mReceived \(audioChunks.count) audio chunks total\u{001B}[0m")
    }

    /// Test different Kokoro voices
    @Test func testKokoroVoices() async throws {
        // 1. Load Kokoro model
        print("\u{001B}[33mLoading Kokoro TTS model...\u{001B}[0m")
        let model = try await Kokoro.fromHub(repoId: kokoroDefaultRepo)
        print("\u{001B}[32mKokoro model loaded!\u{001B}[0m")

        // 2. Test a few different voices
        let voices: [KokoroVoice] = [.afBella, .amAdam, .afNova]
        let text = "Hello world."

        for voice in voices {
            print("\u{001B}[33mTesting voice: \(voice.rawValue)...\u{001B}[0m")

            let audio = try model.generateAudioForSentence(
                voice: voice,
                text: text,
                speed: 1.0
            )

            #expect(audio.shape[0] > 0, "Audio for \(voice.rawValue) should have samples")
            print("\u{001B}[32mVoice \(voice.rawValue) generated audio: \(audio.shape)\u{001B}[0m")
        }

        print("\u{001B}[32mAll voice tests passed!\u{001B}[0m")
    }

    /// Test Kokoro speed parameter
    @Test func testKokoroSpeed() async throws {
        // 1. Load Kokoro model
        print("\u{001B}[33mLoading Kokoro TTS model...\u{001B}[0m")
        let model = try await Kokoro.fromHub(repoId: kokoroDefaultRepo)
        print("\u{001B}[32mKokoro model loaded!\u{001B}[0m")

        let text = "Testing different speech speeds."

        // 2. Generate at different speeds
        let speeds: [Float] = [0.8, 1.0, 1.2]
        var audioLengths: [Float: Int] = [:]

        for speed in speeds {
            print("\u{001B}[33mGenerating at speed \(speed)...\u{001B}[0m")

            let audio = try model.generateAudioForSentence(
                voice: .afBella,
                text: text,
                speed: speed
            )

            audioLengths[speed] = audio.shape[0]
            print("\u{001B}[32mSpeed \(speed): \(audio.shape[0]) samples\u{001B}[0m")
        }

        // Faster speed should generally produce shorter audio (fewer samples)
        // Note: This is a soft check as the relationship isn't always perfectly linear
        #expect(audioLengths[0.8]! > audioLengths[1.2]!, "Slower speed should produce more samples than faster speed")

        print("\u{001B}[32mSpeed tests passed!\u{001B}[0m")
    }

}
