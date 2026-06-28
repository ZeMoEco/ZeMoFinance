package uts.sdk.modules.zemoOcr

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import com.baidu.paddle.lite.MobileConfig
import com.baidu.paddle.lite.PaddlePredictor
import com.baidu.paddle.lite.PowerMode
import java.util.ArrayList
import java.util.Collections
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

// --- Top-Level Data Classes for easier UTS import ---
// Renamed to avoid "Redeclaration" conflict with interface.uts classes
data class KOcrBox(
    val left: Int,
    val top: Int,
    val right: Int,
    val bottom: Int
)

data class KOcrItem(
    val text: String,
    val confidence: Float,
    val box: KOcrBox
)

data class KNativeResult(
    val ok: Boolean,
    val message: String,
    val fullText: String,
    val items: List<KOcrItem>
)

class ZemoOcrImpl {
    private var detPredictor: PaddlePredictor? = null
    private var recPredictor: PaddlePredictor? = null
    private var recDict: Array<String>? = null
    
    @Volatile
    private var isLoaded = false

    fun isInitialized(): Boolean = isLoaded

    @Synchronized
    fun init(detDir: String, recDir: String, dict: Array<String>) {
        android.util.Log.d("ZemoOcr", "Init called with det=$detDir, rec=$recDir")
        if (isLoaded) {
             android.util.Log.d("ZemoOcr", "Already loaded, skipping init")
             return
        }
        
        try {
            // Check native lib loading capability (Manual load test)
            try {
                // Ensure the native library is loaded. 
                // Usually PaddlePredictor.jar does this, but doing it manually helps debug.
                System.loadLibrary("paddle_lite_jni")
                android.util.Log.d("ZemoOcr", "System.loadLibrary('paddle_lite_jni') success")
            } catch (t: Throwable) {
                android.util.Log.e("ZemoOcr", "System.loadLibrary failed: ${t.message}")
                // Don't throw yet, maybe the jar handles it differently
            }

            if (detPredictor == null) {
                android.util.Log.d("ZemoOcr", "Initializing Det Predictor...")
                // No need to prepare/copy files, just check and config
                if (!checkModelExists(detDir)) {
                    android.util.Log.e("ZemoOcr", "Det model missing in $detDir")
                    throw RuntimeException("Det model not found in $detDir. Please ensure 'model.nb' exists.")
                }
                
                android.util.Log.d("ZemoOcr", "Creating Det MobileConfig")
                val detCfg = MobileConfig()
                // Use setModelFromFile for explicit .nb loading if supported, fallback to dir
                // detCfg.setModelDir(detDir) 
                detCfg.setModelFromFile(detDir + "/model.nb")
                
                detCfg.setPowerMode(PowerMode.LITE_POWER_HIGH)
                detCfg.setThreads(4)
                
                android.util.Log.d("ZemoOcr", "Creating Det PaddlePredictor")
                detPredictor = PaddlePredictor.createPaddlePredictor(detCfg)
                android.util.Log.d("ZemoOcr", "Det Predictor created")
            }
            if (recPredictor == null) {
                android.util.Log.d("ZemoOcr", "Initializing Rec Predictor...")
                if (!checkModelExists(recDir)) {
                    android.util.Log.e("ZemoOcr", "Rec model missing in $recDir")
                    throw RuntimeException("Rec model not found in $recDir. Please ensure 'model.nb' exists.")
                }

                android.util.Log.d("ZemoOcr", "Creating Rec MobileConfig")
                val recCfg = MobileConfig()
                // recCfg.setModelDir(recDir)
                recCfg.setModelFromFile(recDir + "/model.nb")

                recCfg.setPowerMode(PowerMode.LITE_POWER_HIGH)
                recCfg.setThreads(4)
                
                android.util.Log.d("ZemoOcr", "Creating Rec PaddlePredictor")
                recPredictor = PaddlePredictor.createPaddlePredictor(recCfg)
                android.util.Log.d("ZemoOcr", "Rec Predictor created")
            }
            recDict = dict
            isLoaded = true
            android.util.Log.d("ZemoOcr", "Init Validated & Completed. Dict Size: ${dict.size}")
        } catch (e: Throwable) {
            e.printStackTrace()
            android.util.Log.e("ZemoOcr", "OCR Init CRASHED: ${e.message}")
            // Throw to be caught by UTS
            throw RuntimeException("OCR Init Failed: ${e.message}", e)
        }
    }

    private fun checkModelExists(dirPath: String): Boolean {
         val dir = java.io.File(dirPath)
         if (!dir.exists()) return false
         
         // Only support .nb model for Mobile
         if (java.io.File(dir, "model.nb").exists()) return true
         
         return false
    }

    private fun prepareModelFiles(dirPath: String) {
        // Deprecated: No file manipulation needed if we use setModelFromFile
    }

    fun recognize(imagePath: String): KNativeResult {
        if (!isLoaded) return KNativeResult(false, "Engine not initialized", "", emptyList())
        
        android.util.Log.d("ZemoOcr", "Recognize start: $imagePath")

        return try {
            var path = imagePath
            if (path.startsWith("file://")) {
                path = path.substring(7)
            }
            
            // Safe decode
            val opts = BitmapFactory.Options()
            opts.inJustDecodeBounds = true
            BitmapFactory.decodeFile(path, opts)
            android.util.Log.d("ZemoOcr", "Image info: ${opts.outWidth}x${opts.outHeight} type=${opts.outMimeType}")
            
            opts.inJustDecodeBounds = false
            // Limit max dimension to avoid OOM or Native buffer overflow
            val maxSide = 1280
            if (opts.outWidth > maxSide || opts.outHeight > maxSide) { 
                 var r = 1
                 if (opts.outWidth > opts.outHeight) {
                     r = (opts.outWidth / maxSide.toFloat()).toInt()
                 } else {
                     r = (opts.outHeight / maxSide.toFloat()).toInt()
                 }
                 if (r <= 0) r = 1
                 opts.inSampleSize = r
                 android.util.Log.d("ZemoOcr", "Image too large, downsampling by $r")
            }
            // Ensure Config is ARGB_8888 (Paddle conversion expects standard format)
            opts.inPreferredConfig = Bitmap.Config.ARGB_8888
            
            val bmp = BitmapFactory.decodeFile(path, opts)
            if (bmp == null) {
                 android.util.Log.e("ZemoOcr", "Bitmap decode failed for $path")
                 return KNativeResult(false, "Decode bitmap failed: $path", "", emptyList())
            }

            android.util.Log.d("ZemoOcr", "Bitmap decoded: ${bmp.width}x${bmp.height}")

            val boxes = try {
                 android.util.Log.d("ZemoOcr", "Starting detection...")
                 // Create a copy to ensure memory layout is simple (unlikely to fix crash but safer)
                 // val cleanBmp = bmp.copy(Bitmap.Config.ARGB_8888, false)
                 // val res = runDet(cleanBmp)
                 val res = runDet(bmp)
                 android.util.Log.d("ZemoOcr", "Detection finished. Boxes: ${res.size}")
                 res
            } catch (t: Throwable) {
                 t.printStackTrace()
                 android.util.Log.e("ZemoOcr", "Detection crashed: ${t.message}")
                 return KNativeResult(false, "Detection failed: ${t.message}", "", emptyList())
            }

            val items = ArrayList<KOcrItem>()
            var boxIdx = 0
            for (box in boxes) {
                boxIdx++
                try {
                    val crop = cropBitmap(bmp, box)
                    if (crop != null) {
                        val res = runRec(crop)
                        android.util.Log.i("ZemoOcr", " >> Box #$boxIdx: '${res.text}' (conf=${res.confidence}) at [${box.left},${box.top}]")
                        
                        // Relaxed logic: Accept low confidence results if text is long enough, or just rely on text content
                        if (res.text.isNotEmpty()) {
                            items.add(KOcrItem(res.text, res.confidence, box))
                        }
                    }
                } catch(e: Exception) {
                    android.util.Log.e("ZemoOcr", "Box #$boxIdx failed: ${e.message}")
                }
            }
            
            android.util.Log.d("ZemoOcr", "Recognition finished. Items: ${items.size}")

            Collections.sort(items) { o1, o2 ->
                val diffY = abs(o1.box.top - o2.box.top)
                if (diffY < 20) {
                    o1.box.left - o2.box.left
                } else {
                    o1.box.top - o2.box.top
                }
            }

            val fullText = items.joinToString("\n") { it.text }
            KNativeResult(true, "Success", fullText, items)

        } catch (e: Throwable) {
            e.printStackTrace()
            android.util.Log.e("ZemoOcr", "Global recognize error: ${e.message}")
            KNativeResult(false, e.message ?: "Unknown error", "", emptyList())
        }
    }

    // --- Private Helpers ---

    private class PrepResult(
        val data: FloatArray,
        val w: Int,
        val h: Int,
        val scaleX: Float,
        val scaleY: Float
    )

    private fun runDet(bmp: Bitmap): List<KOcrBox> {
        val predictor = detPredictor ?: return emptyList()
        val prep = preprocessDet(bmp)
        
        android.util.Log.d("ZemoOcr", "Det Preprocess done. Input: 1, 3, ${prep.h}, ${prep.w}")

        val input = predictor.getInput(0)
        input.resize(longArrayOf(1, 3, prep.h.toLong(), prep.w.toLong()))
        input.setData(prep.data)
        
        android.util.Log.d("ZemoOcr", "Det Predictor Run start")
        predictor.run()
        android.util.Log.d("ZemoOcr", "Det Predictor Run end")
        
        val output = predictor.getOutput(0)
        val outData = output.floatData
        val shape = output.shape()
        
        android.util.Log.d("ZemoOcr", "Det Output: shape=${shape.joinToString()}, dataSize=${outData.size}")

        var outH = prep.h
        var outW = prep.w
        if (shape.size == 4) {
            outH = shape[2].toInt()
            outW = shape[3].toInt()
        }

        if (outData.size < outH * outW) {
             android.util.Log.e("ZemoOcr", "Det output size mismatch!")
             return emptyList()
        }

        // Debug Statistics for Detection Map
        var maxVal = -1.0f
        var sumVal = 0.0f
        for (v in outData) {
            if (v > maxVal) maxVal = v
            sumVal += v
        }
        val avgVal = sumVal / outData.size
        android.util.Log.d("ZemoOcr", "Det Model Output: MaxScore=$maxVal, AvgScore=$avgVal, Shape=${shape.joinToString()}")

        return postprocessDB(outData, outH, outW, prep.scaleX, prep.scaleY)
    }

    private fun preprocessDet(bmp: Bitmap): PrepResult {
        val w = bmp.width
        val h = bmp.height
        var ratio = 1.0f
        val maxSide = max(w, h)
        if (maxSide > 960) ratio = 960.0f / maxSide
        var rw = (w * ratio).roundToInt()
        var rh = (h * ratio).roundToInt()
        rw = max(32, (rw / 32.0).roundToInt() * 32)
        rh = max(32, (rh / 32.0).roundToInt() * 32)
        val sx = w.toFloat() / rw
        val sy = h.toFloat() / rh
        val scaled = Bitmap.createScaledBitmap(bmp, rw, rh, true)
        val pixels = IntArray(rw * rh)
        scaled.getPixels(pixels, 0, rw, 0, 0, rw, rh)
        val data = FloatArray(3 * rw * rh)
        val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
        val std = floatArrayOf(0.229f, 0.224f, 0.225f)
        for (i in 0 until pixels.size) {
            val c = pixels[i]
            val r = ((c shr 16) and 0xFF) / 255.0f
            val g = ((c shr 8) and 0xFF) / 255.0f
            val b = (c and 0xFF) / 255.0f
            data[i] = (r - mean[0]) / std[0]
            data[i + pixels.size] = (g - mean[1]) / std[1]
            data[i + 2 * pixels.size] = (b - mean[2]) / std[2]
        }
        return PrepResult(data, rw, rh, sx, sy)
    }

    private data class RecRes(val text: String, val confidence: Float)

    private fun runRec(bmp: Bitmap): RecRes {
        val predictor = recPredictor ?: return RecRes("", 0f)
        
        // Adaptive Input Shape Handling
        var targetH = 48
        val input = predictor.getInput(0)
        try {
            val inShape = input.shape()
            // inShape is usually [1, 3, H, W] or [?, 3, H, W]
            // Safe check for index 2 (Height)
            if (inShape.size == 4) {
                 val h = inShape[2].toInt()
                 // If model specifies 32 or 48, use it. If it is -1 (dynamic), default to 48.
                 if (h > 0) targetH = h
                 android.util.Log.d("ZemoOcr", "Rec Model Input H=$targetH [Raw: ${inShape.joinToString()}]")
            }
        } catch (e: Exception) {
             android.util.Log.w("ZemoOcr", "Failed to get input shape, defaulting to 48: ${e.message}")
        }
        
        val prep = preprocessRec(bmp, targetH)
        
        input.resize(longArrayOf(1, 3, targetH.toLong(), prep.w.toLong()))
        input.setData(prep.data)
        predictor.run()
        val output = predictor.getOutput(0)
        val outData = output.floatData
        val shape = output.shape()
        var timeStep = 0
        var classNum = 0
        if (shape.size == 3) {
            timeStep = shape[1].toInt()
            classNum = shape[2].toInt()
        }
        if (outData.size < timeStep * classNum) return RecRes("", 0f)
        return decodeCtc(outData, timeStep, classNum)
    }

    private fun preprocessRec(bmp: Bitmap, targetH: Int): PrepResult {
        val h = bmp.height
        val w = bmp.width
        val targetW = 320
        val ratio = targetH.toFloat() / h
        var newW = (w * ratio).roundToInt()
        if (newW > targetW) newW = targetW
        val scaled = Bitmap.createScaledBitmap(bmp, newW, targetH, true)
        val pixels = IntArray(newW * targetH)
        scaled.getPixels(pixels, 0, newW, 0, 0, newW, targetH)
        val count = targetH * targetW
        val data = FloatArray(3 * count)
        for (y in 0 until targetH) {
            for (x in 0 until newW) {
                val px = pixels[y * newW + x]
                val r = ((px shr 16) and 0xFF) / 255.0f
                val g = ((px shr 8) and 0xFF) / 255.0f
                val b = (px and 0xFF) / 255.0f
                val dst = y * targetW + x
                data[dst] = (r - 0.5f) / 0.5f
                data[dst + count] = (g - 0.5f) / 0.5f
                data[dst + 2 * count] = (b - 0.5f) / 0.5f
            }
        }
        return PrepResult(data, targetW, targetH, 1f, 1f)
    }

    private fun postprocessDB(data: FloatArray, h: Int, w: Int, sx: Float, sy: Float): List<KOcrBox> {
        val boxes = ArrayList<KOcrBox>()
        // Lowered threshold further for debugging/improving recall
        val thresh = 0.15f
        val total = h * w
        val mask = BooleanArray(total)
        var pxCount = 0
        for (i in 0 until total) {
            mask[i] = data[i] > thresh
            if (mask[i]) pxCount++
        }
        android.util.Log.d("ZemoOcr", "Det PostProcess: Pixels above thresh($thresh): $pxCount / $total")

        val visited = BooleanArray(total)
        for (i in 0 until total) {
            if (!mask[i] || visited[i]) continue
            var minX = i % w
            var maxX = minX
            var minY = i / w
            var maxY = minY
            val q = ArrayList<Int>()
            q.add(i)
            visited[i] = true
            var head = 0
            while (head < q.size) {
                val curr = q[head++]
                val cx = curr % w
                val cy = curr / w
                if (cx < minX) minX = cx
                if (cx > maxX) maxX = cx
                if (cy < minY) minY = cy
                if (cy > maxY) maxY = cy
                val nbs = intArrayOf(curr - 1, curr + 1, curr - w, curr + w)
                for (nb in nbs) {
                    if (nb in 0 until total && mask[nb] && !visited[nb]) {
                        val nx = nb % w
                        if (abs(nx - cx) > 1) continue
                        visited[nb] = true
                        q.add(nb)
                    }
                }
            }
            // Allow smaller boxes (was 20)
            if (q.size < 10) continue
            val padding = 2
            val l = max(0, minX - padding)
            val r = min(w, maxX + padding)
            val t = max(0, minY - padding)
            val b = min(h, maxY + padding)
            boxes.add(KOcrBox((l * sx).toInt(), (t * sy).toInt(), (r * sx).toInt(), (b * sy).toInt()))
        }
        return boxes
    }

    private fun decodeCtc(data: FloatArray, steps: Int, classes: Int): RecRes {
        val dict = recDict ?: return RecRes("", 0f)
        
        val sb = StringBuilder()
        var lastIdx = -1
        var totalConf = 0f
        var count = 0
        
        // Debug: Collect raw indices to see if model is outputting anything valid
        val debugRaw = StringBuilder()
        var hasValid = false

        for (t in 0 until steps) {
            var maxVal = -10000f
            var maxIdx = 0
            val offset = t * classes
            
            // Fast loop to find max
            if (offset + classes <= data.size) {
                for (c in 0 until classes) {
                    val v = data[offset + c]
                    if (v > maxVal) { maxVal = v; maxIdx = c }
                }
            }
            
            if (debugRaw.length < 100) debugRaw.append("$maxIdx,")

            if (maxIdx != 0 && maxIdx != lastIdx) {
                val dictIdx = maxIdx - 1
                if (dictIdx >= 0 && dictIdx < dict.size) {
                    sb.append(dict[dictIdx])
                    totalConf += maxVal
                    count++
                    hasValid = true
                } else {
                    // Index out of bounds or special token
                    android.util.Log.w("ZemoOcr", "Rec OOB: Idx=$maxIdx, DictMax=${dict.size}")
                }
            }
            lastIdx = maxIdx
        }
        
        if (!hasValid && steps > 0) {
             android.util.Log.d("ZemoOcr", "Rec Failed (All 0 or OOB). Raw: $debugRaw... DictSize=${dict.size} Classes=$classes")
        }

        return RecRes(sb.toString(), if (count > 0) totalConf / count else 0f)
    }

    private fun cropBitmap(bmp: Bitmap, box: KOcrBox): Bitmap? {
        return try {
            val x = max(0, box.left)
            val y = max(0, box.top)
            var wPath = box.right - box.left
            var hPath = box.bottom - box.top
            if (x + wPath > bmp.width) wPath = bmp.width - x
            if (y + hPath > bmp.height) hPath = bmp.height - y
            if (wPath <= 0 || hPath <= 0) return null
            Bitmap.createBitmap(bmp, x, y, wPath, hPath)
        } catch (e: Exception) { null }
    }
}
