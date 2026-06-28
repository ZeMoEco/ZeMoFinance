package uts.sdk.modules.zemoOcr

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.util.ArrayList
import java.util.Collections
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

class ZemoOcrOnnxImpl {
    private var env: OrtEnvironment? = null
    private var detSession: OrtSession? = null
    private var recSession: OrtSession? = null
    private var recDict: Array<String>? = null

    @Volatile
    private var isLoaded = false

    fun isInitialized(): Boolean = isLoaded

    @Synchronized
    fun init(detModelPath: String, recModelPath: String, dict: Array<String>) {
        android.util.Log.d("ZemoOcrOnnx", "Init called with det=$detModelPath, rec=$recModelPath")
        if (isLoaded) return

        try {
            val detFile = java.io.File(detModelPath)
            val recFile = java.io.File(recModelPath)
            if (!detFile.exists()) throw RuntimeException("Det ONNX model not found: $detModelPath")
            if (!recFile.exists()) throw RuntimeException("Rec ONNX model not found: $recModelPath")

            val runtime = OrtEnvironment.getEnvironment()
            val sessionOptions = OrtSession.SessionOptions()
            sessionOptions.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT)

            env = runtime
            detSession = runtime.createSession(detModelPath, sessionOptions)
            recSession = runtime.createSession(recModelPath, sessionOptions)
            recDict = dict
            isLoaded = true
            android.util.Log.d("ZemoOcrOnnx", "Init completed. Dict Size: ${dict.size}")
        } catch (e: Throwable) {
            e.printStackTrace()
            android.util.Log.e("ZemoOcrOnnx", "Init failed: ${e.message}")
            throw RuntimeException("ONNX OCR init failed: ${e.message}", e)
        }
    }

    fun recognize(imagePath: String): KNativeResult {
        if (!isLoaded) return KNativeResult(false, "ONNX engine not initialized", "", emptyList())

        return try {
            var path = imagePath
            if (path.startsWith("file://")) path = path.substring(7)

            val opts = BitmapFactory.Options()
            opts.inJustDecodeBounds = true
            BitmapFactory.decodeFile(path, opts)
            opts.inJustDecodeBounds = false
            val maxSide = 1280
            if (opts.outWidth > maxSide || opts.outHeight > maxSide) {
                var r = if (opts.outWidth > opts.outHeight) {
                    (opts.outWidth / maxSide.toFloat()).toInt()
                } else {
                    (opts.outHeight / maxSide.toFloat()).toInt()
                }
                if (r <= 0) r = 1
                opts.inSampleSize = r
            }
            opts.inPreferredConfig = Bitmap.Config.ARGB_8888

            val bmp = BitmapFactory.decodeFile(path, opts)
                ?: return KNativeResult(false, "Decode bitmap failed: $path", "", emptyList())

            val boxes = try {
                runDet(bmp)
            } catch (t: Throwable) {
                t.printStackTrace()
                return KNativeResult(false, "ONNX detection failed: ${t.message}", "", emptyList())
            }

            val items = ArrayList<KOcrItem>()
            var boxIdx = 0
            for (box in boxes) {
                boxIdx++
                try {
                    val crop = cropBitmap(bmp, box)
                    if (crop != null) {
                        val res = runRec(crop)
                        android.util.Log.i("ZemoOcrOnnx", "Box #$boxIdx: '${res.text}' conf=${res.confidence}")
                        if (res.text.isNotEmpty()) items.add(KOcrItem(res.text, res.confidence, box))
                    }
                } catch (e: Throwable) {
                    android.util.Log.e("ZemoOcrOnnx", "Box #$boxIdx failed: ${e.message}")
                }
            }

            Collections.sort(items) { o1, o2 ->
                val diffY = abs(o1.box.top - o2.box.top)
                if (diffY < 20) o1.box.left - o2.box.left else o1.box.top - o2.box.top
            }

            KNativeResult(true, "Success", items.joinToString("\n") { it.text }, items)
        } catch (e: Throwable) {
            e.printStackTrace()
            KNativeResult(false, e.message ?: "Unknown ONNX OCR error", "", emptyList())
        }
    }

    private class PrepResult(
        val data: FloatArray,
        val w: Int,
        val h: Int,
        val scaleX: Float,
        val scaleY: Float
    )

    private data class RecRes(val text: String, val confidence: Float)

    private fun runDet(bmp: Bitmap): List<KOcrBox> {
        val runtime = env ?: return emptyList()
        val session = detSession ?: return emptyList()
        val prep = preprocessDet(bmp)
        val inputName = session.inputNames.iterator().next()
        val inputShape = longArrayOf(1, 3, prep.h.toLong(), prep.w.toLong())
        val tensor = OnnxTensor.createTensor(runtime, directFloatBuffer(prep.data), inputShape)
        val result = session.run(mapOf(inputName to tensor))
        try {
            val output = result.iterator().next().value
            val value = output.value
            val outData = flattenFloats(value)
            val shape = readShape(value)
            var outH = prep.h
            var outW = prep.w
            if (shape.size >= 4) {
                if (shape[1] == 1) {
                    outH = shape[2]
                    outW = shape[3]
                } else if (shape[3] == 1) {
                    outH = shape[1]
                    outW = shape[2]
                }
            } else if (shape.size == 3) {
                outH = shape[1]
                outW = shape[2]
            } else if (shape.size == 2) {
                outH = shape[0]
                outW = shape[1]
            }
            if (outData.size < outH * outW) return emptyList()
            return postprocessDB(outData, outH, outW, prep.scaleX, prep.scaleY)
        } finally {
            result.close()
            tensor.close()
        }
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
        for (i in pixels.indices) {
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

    private fun runRec(bmp: Bitmap): RecRes {
        val runtime = env ?: return RecRes("", 0f)
        val session = recSession ?: return RecRes("", 0f)
        val inputName = session.inputNames.iterator().next()
        var targetH = 48
        var targetW = 320
        try {
            val info = session.inputInfo[inputName]?.info
            if (info is TensorInfo) {
                val shape = info.shape
                if (shape.size == 4) {
                    if (shape[2] > 0) targetH = shape[2].toInt()
                    if (shape[3] > 0) targetW = shape[3].toInt()
                }
            }
        } catch (_e: Throwable) {}

        val prep = preprocessRec(bmp, targetH, targetW)
        val tensor = OnnxTensor.createTensor(
            runtime,
            directFloatBuffer(prep.data),
            longArrayOf(1, 3, targetH.toLong(), prep.w.toLong())
        )
        val result = session.run(mapOf(inputName to tensor))
        try {
            val outputValue = result.iterator().next().value.value
            val outData = flattenFloats(outputValue)
            val shape = readShape(outputValue)
            var timeStep = 0
            var classNum = 0
            if (shape.size == 3) {
                timeStep = shape[1]
                classNum = shape[2]
            } else if (shape.size == 2) {
                timeStep = shape[0]
                classNum = shape[1]
            }
            if (timeStep <= 0 || classNum <= 0 || outData.size < timeStep * classNum) return RecRes("", 0f)
            return decodeCtc(outData, timeStep, classNum)
        } finally {
            result.close()
            tensor.close()
        }
    }

    private fun preprocessRec(bmp: Bitmap, targetH: Int, targetW: Int): PrepResult {
        val h = bmp.height
        val w = bmp.width
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
        val thresh = 0.15f
        val total = h * w
        val mask = BooleanArray(total)
        for (i in 0 until total) mask[i] = data[i] > thresh

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
        for (t in 0 until steps) {
            var maxVal = -10000f
            var maxIdx = 0
            val offset = t * classes
            if (offset + classes <= data.size) {
                for (c in 0 until classes) {
                    val v = data[offset + c]
                    if (v > maxVal) {
                        maxVal = v
                        maxIdx = c
                    }
                }
            }
            if (maxIdx != 0 && maxIdx != lastIdx) {
                val dictIdx = maxIdx - 1
                if (dictIdx >= 0 && dictIdx < dict.size) {
                    sb.append(dict[dictIdx])
                    totalConf += maxVal
                    count++
                }
            }
            lastIdx = maxIdx
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
        } catch (_e: Throwable) {
            null
        }
    }

    private fun flattenFloats(value: Any?): FloatArray {
        val out = ArrayList<Float>()
        fun walk(v: Any?) {
            if (v == null) return
            when (v) {
                is FloatArray -> for (x in v) out.add(x)
                is DoubleArray -> for (x in v) out.add(x.toFloat())
                is IntArray -> for (x in v) out.add(x.toFloat())
                is LongArray -> for (x in v) out.add(x.toFloat())
                is Number -> out.add(v.toFloat())
                else -> {
                    val cls = v.javaClass
                    if (cls.isArray) {
                        val len = java.lang.reflect.Array.getLength(v)
                        for (i in 0 until len) walk(java.lang.reflect.Array.get(v, i))
                    }
                }
            }
        }
        walk(value)
        return out.toFloatArray()
    }

    private fun directFloatBuffer(data: FloatArray): FloatBuffer {
        val buffer = ByteBuffer
            .allocateDirect(data.size * java.lang.Float.BYTES)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
        buffer.put(data)
        buffer.rewind()
        return buffer
    }

    private fun readShape(value: Any?): IntArray {
        val dims = ArrayList<Int>()
        var current = value
        while (current != null && current.javaClass.isArray) {
            val len = java.lang.reflect.Array.getLength(current)
            dims.add(len)
            current = if (len > 0) java.lang.reflect.Array.get(current, 0) else null
        }
        return dims.toIntArray()
    }
}
