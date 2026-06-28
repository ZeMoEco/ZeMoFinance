import Foundation
// 假设已桥接 OnnxRuntimeGenAI 的 C API 或 Swift Wrapper
// 这里写核心逻辑伪代码

var model: OGAModel?
var tokenizer: OGATokenizer?

public func loadModel(_ path: String) -> Bool {
    do {
        model = try OGAModel(path: path)
        tokenizer = try OGATokenizer(model: model!)
        return true
    } catch {
        print("Error loading model: \(error)")
        return false
    }
}

public func chatStream(_ prompt: String, _ callback: @escaping (String, Bool) -> Void) {
    guard let model = model, let tokenizer = tokenizer else {
        callback("Model not ready", true)
        return
    }

    DispatchQueue.global(qos: .userInitiated).async {
        do {
            let params = try OGAGeneratorParams(model: model)
            let formattedPrompt = "<|user|>\n\(prompt)<|end|>\n<|assistant|>"
            let inputIds = try tokenizer.encode(formattedPrompt)
            
            try params.setInput(inputIds)
            let generator = try OGAGenerator(model: model, params: params)

            while !generator.isDone() {
                try generator.computeLogits()
                try generator.generateNextToken()
                
                let newTokens = generator.getSequence(0)
                if let lastToken = newTokens.last {
                    let text = try tokenizer.decode([lastToken])
                    
                    // 回调主线程
                    DispatchQueue.main.async {
                        callback(text, false)
                    }
                }
            }
            
            DispatchQueue.main.async {
                callback("", true)
            }
            
        } catch {
            DispatchQueue.main.async {
                callback("Error: \(error)", true)
            }
        }
    }
}