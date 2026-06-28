# ZeMo AI - LiteRT Native AI Engine

基于 Google MediaPipe LiteRT 的本地AI引擎，支持LLM推理、向量RAG和函数调用。

## 功能特性

### 1. LLM推理
- 使用 MediaPipe `tasks-genai:0.10.14` 进行本地大模型推理
- 支持流式输出和同步调用
- 多轮对话上下文管理

### 2. 原生向量RAG
- 轻量级字符频率 + Bigram哈希向量嵌入
- 余弦相似度检索
- 自动知识积累（成功/失败经验）

### 3. 函数调用 (Function Calling)
- ReAct Agent模式
- 多格式JSON解析支持
- 内置函数 + 动态插件注册

## 使用方法

### 初始化

```typescript
import { AIAgentExecutor } from '@/services/ai/AIAgentExecutor.uts';

// 初始化AI引擎
const modelPath = '/storage/emulated/0/zemo/models/gemma-2b.bin';
AIAgentExecutor.init(modelPath);
```

### 对话

```typescript
// 流式对话
AIAgentExecutor.chat("帮我创建一个任务：明天开会", (text, done) => {
    console.log(text);
    if (done) {
        console.log("对话完成");
    }
});
```

### 添加知识

```typescript
// 添加知识到RAG
AIAgentExecutor.addKnowledge(
    "guide_1",
    "当用户说想休息时，推荐使用番茄钟休息功能",
    { type: "guide", scene: "rest" }
);
```

### 注册自定义函数

```typescript
import { PluginRegistry } from '@/services/ai/PluginRegistry.uts';

// 注册自定义插件
PluginRegistry.register(JSON.stringify({
    name: "playMusic",
    description: "播放指定歌曲",
    parameters: {
        type: "object",
        properties: {
            songName: { type: "string", description: "歌曲名称" }
        },
        required: ["songName"]
    },
    implementation: "native:MusicModule:play"
}));
```

## API 参考

### LLM Core

| 函数 | 说明 |
|------|------|
| `initModel(path, embedderPath?)` | 初始化LLM模型 |
| `chatStream(prompt, callback)` | 流式对话 |
| `chatSync(prompt)` | 同步对话 |
| `chatWithContext(msg, system, cb)` | 带上下文多轮对话 |
| `clearHistory()` | 清除对话历史 |
| `resetModel()` | 重置模型（清除KV Cache） |

### Native RAG

| 函数 | 说明 |
|------|------|
| `ragAddDocument(id, text, meta)` | 添加单个文档 |
| `ragAddBatch(docsJson)` | 批量添加文档 |
| `ragSearch(query, topK)` | 搜索相似文档 |
| `ragGetContext(query, topK)` | 获取格式化上下文 |
| `ragClear()` | 清空知识库 |
| `ragSize()` | 获取知识库大小 |

### Function Calling

| 函数 | 说明 |
|------|------|
| `registerFunction(defJson)` | 注册单个函数 |
| `registerFunctions(defsJson)` | 批量注册函数 |
| `getFunctionDefinitions()` | 获取所有函数定义 |
| `parseFunctionCall(output)` | 解析LLM输出中的函数调用 |
| `buildFunctionPrompt(sys, user, rag)` | 构建带函数调用的Prompt |
| `executeFunctionCall(parsed)` | 执行已解析的函数调用 |

## 架构图

```
┌─────────────────────────────────────────────────────────┐
│                    UTS Layer (服务层)                    │
├─────────────────────────────────────────────────────────┤
│  AIAgentExecutor                                        │
│  ├── ReAct Loop (推理→执行→观察→修正)                    │
│  ├── PromptFactorService (Prompt因子管理)               │
│  └── PluginRegistry (动态插件管理)                       │
├─────────────────────────────────────────────────────────┤
│                  Bridge Layer (桥接层)                   │
├─────────────────────────────────────────────────────────┤
│  index.uts                                              │
│  ├── LLM API (init, chat, context)                     │
│  ├── RAG API (add, search, context)                    │
│  └── FC API (register, parse, execute)                 │
├─────────────────────────────────────────────────────────┤
│                 Native Layer (原生层)                    │
├─────────────────────────────────────────────────────────┤
│  ZemoAIImpl.kt                                          │
│  ├── LlmInference (MediaPipe LiteRT)                   │
│  ├── VectorStore (轻量级向量存储)                        │
│  ├── FunctionRegistry (函数注册表)                       │
│  └── Reflection Invoker (反射调用器)                    │
└─────────────────────────────────────────────────────────┘
```

## 依赖配置 (config.json)

```json
{
  "dependencies": [
    "com.google.mediapipe:tasks-genai:0.10.14"
  ],
  "minSdkVersion": 24
}
```

## 模型支持

支持 MediaPipe LiteRT 格式的模型文件 (`.bin`)：
- Gemma 2B
- Gemma 7B  
- Phi-2
- StableLM
- 其他 LiteRT 兼容模型

## 开发文档
- [UTS 语法](https://uniapp.dcloud.net.cn/tutorial/syntax-uts.html)
- [UTS API插件](https://uniapp.dcloud.net.cn/plugin/uts-plugin.html)
- [Hello UTS](https://gitcode.net/dcloud/hello-uts)