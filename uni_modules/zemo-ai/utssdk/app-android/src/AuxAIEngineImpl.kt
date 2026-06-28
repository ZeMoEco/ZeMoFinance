package uts.sdk.modules.zemoAi

import android.util.Log
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import io.dcloud.uts.UTSAndroid
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.sqrt

// LiteRT Standard Imports
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.BuiltinNpuAcceleratorProvider

class AuxAIEngineImpl {
    private val TAG = "AuxAIEngine"
    
    private val models = mutableMapOf<String, CompiledModel>()
    private val envs = mutableMapOf<String, Environment>()

    fun loadModel(key: String, path: String, useNpu: Boolean): Boolean {
        try {
            val absPath = resolvePath(path) ?: return false
            Log.i(TAG, "Loading Aux Model ($key): $absPath")
            
            // Clean prev
            unloadModel(key)

            val context = UTSAndroid.getAppContext()
            // Create Env
            val env = try {
                if (useNpu && context != null) Environment.create(BuiltinNpuAcceleratorProvider(context))
                else Environment.create() 
            } catch (e: Exception) { Environment.create() }

            val accelerator = if (useNpu) Accelerator.NPU else Accelerator.CPU
            val options = CompiledModel.Options(accelerator)

            try {
                val model = CompiledModel.create(absPath, options, env)
                models[key] = model
                envs[key] = env
                Log.d(TAG, "Loaded $key on $accelerator")
                return true
            } catch (e: Exception) {
                // Fallback
                 if (accelerator != Accelerator.CPU) {
                    try {
                        val cpuOpt = CompiledModel.Options(Accelerator.CPU)
                        val model = CompiledModel.create(absPath, cpuOpt, null)
                        models[key] = model
                        Log.d(TAG, "Loaded $key on CPU (Fallback)")
                        return true
                    } catch(ex: Exception) { }
                 }
                 Log.e(TAG, "Failed to load $key: ${e.message}")
                 return false
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error loading $key: ${e.message}")
            return false
        }
    }

    fun unloadModel(key: String) {
        models[key]?.close()
        envs[key]?.close()
        models.remove(key)
        envs.remove(key)
    }

    // --- Inference Specifics ---

    /**
     * Compute Embedding for RAG
     * Returns FloatArray or empty if failed
     */
    fun computeEmbedding(key: String, text: String): FloatArray {
        val model = models[key] ?: return FloatArray(0)
        
        try {
            val inputs = model.createInputBuffers()
            val outputs = model.createOutputBuffers()
            
            val maxSeq = 128 // Hardcoded for simplified demo, should match model signature
            val bytes = text.toByteArray()
            
            // Heuristic to guess input type (Float vs Int Token IDs) - Simplified
            try {
                 val floatInput = FloatArray(maxSeq) { k -> if (k < bytes.size) bytes[k].toFloat() / 255.0f else 0f }
                 inputs[0].writeFloat(floatInput)
            } catch (e: Exception) {
                 val tokenIds = IntArray(maxSeq) { 0 }
                 for (i in 0 until minOf(bytes.size, maxSeq)) tokenIds[i] = bytes[i].toInt()
                 inputs[0].writeInt(tokenIds)
            }

            model.run(inputs, outputs)
            return outputs[0].readFloat()
        } catch (e: Exception) {
            Log.e(TAG, "Embedding Error: ${e.message}")
            return FloatArray(0)
        }
    }

    /**
     * Image Classification
     */
    fun classify(key: String, imagePath: String): String {
        val model = models[key] ?: return "No Model"
        try {
            val bitmap = BitmapFactory.decodeFile(imagePath) ?: return "Load Failed"
            val scaled = Bitmap.createScaledBitmap(bitmap, 224, 224, true)
            
            val inputBuffer = ByteBuffer.allocateDirect(1 * 224 * 224 * 3 * 4) 
            inputBuffer.order(ByteOrder.nativeOrder())
            
            val intValues = IntArray(224 * 224)
            scaled.getPixels(intValues, 0, scaled.width, 0, 0, scaled.width, scaled.height)
            
            for (pixel in intValues) {
                // Normalize 0..1
                inputBuffer.putFloat(((pixel shr 16) and 0xFF) / 255.0f)
                inputBuffer.putFloat(((pixel shr 8) and 0xFF) / 255.0f)
                inputBuffer.putFloat((pixel and 0xFF) / 255.0f)
            }
            
            val inputs = model.createInputBuffers()
            val outputs = model.createOutputBuffers()
            
            // Write input
            val floatArr = FloatArray(224 * 224 * 3)
            inputBuffer.flip() // Reset pos
            inputBuffer.asFloatBuffer().get(floatArr)
            inputs[0].writeFloat(floatArr)
            
            model.run(inputs, outputs)
            
            val probs = outputs[0].readFloat()
            val maxIdx = probs.indices.maxByOrNull { probs[it] } ?: -1
            val conf = probs[maxIdx]
            
            return "Class: $maxIdx, Conf: $conf"
        } catch (e: Exception) {
            return "Error: ${e.message}"
        }
    }

    private fun resolvePath(path: String): String? {
        val context = UTSAndroid.getAppContext() ?: return null
        var absPath = UTSAndroid.convert2AbsFullPath(path)
        if (!File(absPath).exists()) {
             if (File(path).exists()) return path
             return null
        }
        return absPath
    }
}
