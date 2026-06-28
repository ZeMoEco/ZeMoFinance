package uts.sdk.modules.zemoAi

import android.util.Log
import io.dcloud.uts.UTSAndroid
import java.io.File
import java.io.RandomAccessFile
import kotlinx.coroutines.*
import org.json.JSONObject

// LiteRT-LM Imports
import com.google.ai.edge.litertlm.Conversation
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Message
import com.google.ai.edge.litertlm.MessageCallback

class GenAIEngineImpl {
    private val TAG = "GenAIEngine"
    
    private var genAiEngine: Engine? = null
    private var genAiConversation: Conversation? = null
    private var isGenAiReady = false
    private var maxTokens = 512

    companion object {
        // Shared scope if needed, or stick to instance
    }

    @JvmOverloads
    fun initModel(path: String, useNpu: Boolean = true): Boolean {
        val absPath = resolvePath(path) ?: return false
        
        val file = File(absPath)
        val size = file.length()
        Log.i(TAG, "Checking GenAI Model: $absPath (Size: $size bytes)")
        
        if (size < 1024) {
             Log.e(TAG, "GenAI Model file is too small ($size bytes). Likely corrupted.")
             return false
        }

        // Check format
        if (!absPath.endsWith(".bin") && !absPath.endsWith(".tflite") && !absPath.endsWith(".litertlm")) {
             Log.w(TAG, "Warning: Non-standard extension. Ensure this model is compatible with LiteRT-LM 0.8.0.")
        }
        
        // Reset
        genAiConversation = null
        genAiEngine = null
        isGenAiReady = false
        
        // Strategy: Async Loading to prevent UI Freeze
        // We return true immediately if file check passes, but actual engine loading happens in background.
        // A status check method (isReady) should be polled or callback used (but this signature returns Boolean).
        
        CoroutineScope(Dispatchers.IO).launch {
             // Strategy Update: Prioritize GPU (MediaPipe Default) -> NPU -> CPU
             // NPU compilation can be very slow (5-10s) and fail on many devices. 
             // GPU is generally faster to initialize and supported on more devices for LLMs.
            if (useNpu) {
                try {
                     Log.i(TAG, "Attempting GPU (Async)...")
                     if (initBackend(absPath, Backend.GPU)) return@launch
                } catch(e: Throwable) { 
                     Log.w(TAG, "GPU Init failed: ${e.message}")
                }

                Log.i(TAG, "Attempting NPU (Async)...")
                if (initBackend(absPath, Backend.NPU)) return@launch
            }
            
            Log.i(TAG, "Attempting CPU (Async)...")
            initBackend(absPath, Backend.CPU)
        }
        
        return true
    }

    private fun initBackend(path: String, backend: Backend): Boolean {
        try {
            val config = EngineConfig(path, backend)
            val engine = Engine(config)
            
            // Explicitly Initialize the Engine (Required for LiteRT-LM)
            Log.d(TAG, "Initializing Engine...")
            // Note: engine.initialize() might block, but we are inside an async-friendly flow or should be.
            // Since this is native code called from JS, ideally we should be careful, 
            // but the plugin logic seems to call this synchronously at startup.
            // The documentation warns it can take time (10s).
            val startTime = System.currentTimeMillis()
            var methodFound = false
            try {
                // Try finding the initialize method via specific signature if needed, or just call it directly
                // Based on docs: engine.initialize()
                val methods = engine.javaClass.methods
                for (m in methods) {
                    if (m.name == "initialize") {
                         m.invoke(engine) // Reflection fallback if signature varies
                         methodFound = true
                         break
                    }
                }
                // If direct call is compiled, use it (Assuming library matches docs):
                if (!methodFound) {
                     // Try standard Kotlin call - 
                     // IMPORTANT: Since I can't see the JAR signatures, I'll rely on the user report
                     // User said "IllegalStateException: Engine is not initialized", implying 'initialize()' was missed.
                     // Please try uncommenting if the below doesn't work directly:
                     // engine.initialize() 
                     
                     // Direct invoke via reflection for safety in this blind-environment:
                     engine::class.java.getMethod("initialize").invoke(engine)
                }
            } catch (e: NoSuchMethodException) {
                Log.w(TAG, "Method initialize() not found. Trying implicit init...")
            }
            
            Log.d(TAG, "Engine Initialized (${System.currentTimeMillis() - startTime}ms)")

            // Critical Step: Create Conversation (initializes graph/tokenizer)
            val conversation = engine.createConversation()
            
            if (conversation != null) {
                genAiEngine = engine
                genAiConversation = conversation
                isGenAiReady = true
                Log.i(TAG, "GenAI Init Success ($backend)")
                return true
            }
        } catch (e: Throwable) {
            Log.w(TAG, "GenAI Init Failed ($backend): ${e.message}")
            if (e.message?.contains("initialized") == true) {
                 Log.e(TAG, ">>> TIP: 'Engine not initialized' usually means the Model Format (.litertlm/Gemma3) is too new for this Engine version (0.8.0). Try a Gemma2 .bin/.tflite model.")
            }
        }
        return false
    }

    fun chatAsync(prompt: String, callback: (String, Boolean) -> Unit) {
        Log.d(TAG, "chatAsync received prompt: '$prompt'") // Debug Log

        if (!isGenAiReady || genAiConversation == null) {
            Log.e(TAG, "chatAsync failed: Model not ready")
            safeCallback(callback, "Error: Model not ready", true)
            return
        }

        CoroutineScope(Dispatchers.IO).launch {
            try {
                Log.d(TAG, "Sending message to conversation...")
                // Use Message.of() for compatibility based on previous fixes
                genAiConversation?.sendMessageAsync(Message.of(prompt), object : MessageCallback {
                    override fun onMessage(message: Message) {
                        try {
                            // Fix: Extract text content instead of toString() which dumps object
                            // Note: Message structure varies. Trying standard accessors.
                            // If 'text' property exists: message.text
                            // If 'content' exists: message.content
                            // Falling back to reflection for "text" or "content" if simple access fails during comp.
                            // Assuming .toString() was returning "Message(text=Hello...)"
                            
                            var text = message.toString()
                            // Simple heuristic to clean up if it's dumping the object
                            if (text.startsWith("Message")) {
                                // Try finding content inside
                                // Try standard property first if known, else minimal cleanup
                            }
                            
                            // Try to get 'text' property via reflection if we don't know the exact class def
                            try {
                                val m = message.javaClass.getMethod("getText")
                                val t = m.invoke(message)
                                if (t != null) text = t.toString()
                            } catch(e: Exception) {
                                // Try distinct content getter?
                            }

                            Log.d(TAG, "onMessage: $text") // Debug Log
                            safeCallback(callback, text, false)
                        } catch (e: Exception) {
                             Log.e(TAG, "onMessage Error: ${e.message}")
                        }
                    }
                    override fun onDone() {
                        Log.d(TAG, "onDone") // Debug Log
                        safeCallback(callback, "", true)
                    }
                    override fun onError(t: Throwable) {
                        Log.e(TAG, "onError: ${t.message}") // Debug Log
                        t.printStackTrace()
                        safeCallback(callback, "GenAI Error: ${t.message}", true)
                    }
                })
            } catch (e: Exception) {
                Log.e(TAG, "chatAsync Critical: ${e.message}")
                e.printStackTrace()
                safeCallback(callback, "Critical: ${e.message}", true)
            }
        }
    }
    
    fun clearHistory() { 
        if (genAiEngine != null) {
             genAiConversation = genAiEngine?.createConversation()
        }
    }

    fun isReady(): Boolean = isGenAiReady

    /**
     * Native invoke (Reflection support)
     */
    fun invokeNativePlugin(className: String, methodName: String, argsJson: String): String {
        try {
            val clazz = Class.forName(className)
            val instance = try {
                 // Try to get a singleton or static instance if available, otherwise new instance
                 val instanceField = try { clazz.getDeclaredField("INSTANCE") } catch(e: Exception) { null }
                 if (instanceField != null) {
                     instanceField.isAccessible = true
                     instanceField.get(null)
                 } else {
                     clazz.getDeclaredConstructor().newInstance()
                 }
            } catch (e: Exception) {
                 clazz.getDeclaredConstructor().newInstance()
            }
            
            try {
                val method = clazz.getMethod(methodName, JSONObject::class.java)
                return method.invoke(instance, JSONObject(argsJson))?.toString() ?: ""
            } catch (e: NoSuchMethodException) {
                try {
                    val method = clazz.getMethod(methodName, String::class.java)
                    return method.invoke(instance, argsJson)?.toString() ?: ""
                } catch (e2: Exception) {
                    val method = clazz.getMethod(methodName)
                    return method.invoke(instance)?.toString() ?: ""
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Invoke failed: ${e.message}")
            return "Error: ${e.message}"
        }
    }

    // Helpers
    private fun resolvePath(path: String): String? {
        val context = UTSAndroid.getAppContext() ?: return null
        var absPath = UTSAndroid.convert2AbsFullPath(path)
        if (!File(absPath).exists()) {
             if (File(path).exists()) return path
             return null
        }
        return absPath
    }
    
    private fun isNpuSupported(): Boolean {
        return true // Add logic if needed
    }

    private fun safeCallback(cb: (String, Boolean) -> Unit, text: String, done: Boolean) {
        CoroutineScope(Dispatchers.Main).launch { cb(text, done) }
    }
}
