package src

import android.content.Context
import android.util.Log
import com.k2fsa.sherpa.onnx.*
import java.io.*
import java.util.ArrayList
import java.util.concurrent.ConcurrentHashMap
import kotlin.math.sqrt

class AudioResult(val samples: FloatArray, val volume: Int)

class SherpaWrapper {
	    // 增加同步锁对象
	    private val lock = Any()
    private var voiceDetector: Vad? = null
    private var keywordSpotter: KeywordSpotter? = null
    private var recognizer: OnlineRecognizer? = null
    private var speakerExtractor: SpeakerEmbeddingExtractor? = null

    private var kwsStream: OnlineStream? = null
    private var asrStream: OnlineStream? = null

    private var currentState = 0 
    private var isRegistering = false
    private var currentRegisterName = ""
    private val registerEmbeddings = ArrayList<FloatArray>()
    private val speakerDatabase = ConcurrentHashMap<String, FloatArray>()
    private val logBuffer = StringBuilder()

    private val circleBuffer = FloatArray(16000 * 2)
    private var bufferIdx = 0
		// 增加一个初始化状态位
	private var isInitialized = false
    companion object {
        const val TAG = "ZeMo-Sherpa"
        const val SAMPLE_RATE = 16000
    }

    private fun writeLog(msg: String) {
        Log.i(TAG, msg)
        logBuffer.insert(0, "[$msg]\n")
        if (logBuffer.length > 2000) logBuffer.setLength(2000)
    }

    fun getLatestLogs(): String = logBuffer.toString()

        fun init(context: Context): Boolean = synchronized(lock) {
            val am = context.assets
            return try {
				   writeLog("开始初始化...")
                // --- 1. VAD 初始化 ---
                val sileroConfig = SileroVadModelConfig(
                    "model-vad/silero_vad.onnx", 0.5f, 0.01f, 0.2f, 512, 4.0f
                )
                val vadConfig = VadModelConfig(
                    sileroVadModelConfig = sileroConfig,
                    tenVadModelConfig = TenVadModelConfig(),
                    sampleRate = SAMPLE_RATE,
                    numThreads = 1,
                    provider = "cpu",
                    debug = false
                )
                this.voiceDetector = Vad(am, vadConfig)
    
                // --- 2. KWS 初始化 ---
                val kwsConfig = KeywordSpotterConfig().apply {
                    modelConfig = OnlineModelConfig().apply {
                        transducer = OnlineTransducerModelConfig().apply {
                            encoder = "model-kws/encoder-epoch-99-avg-1-chunk-16-left-64.int8.onnx"
                            decoder = "model-kws/decoder-epoch-99-avg-1-chunk-16-left-64.int8.onnx"
                            joiner = "model-kws/joiner-epoch-99-avg-1-chunk-16-left-64.int8.onnx"
                        }
                        tokens = "model-kws/tokens.txt"
                        // 【关键修复】设为空字符串，让库自动检测模型类型，避免 attention_dims 报错
                        modelType = "transducer" 
                        numThreads =2
                    }
                    keywordsFile = "model-kws/keywords.txt"
                    featConfig = FeatureConfig().apply { sampleRate = SAMPLE_RATE; featureDim = 80 }
                }
                this.keywordSpotter = KeywordSpotter(am, kwsConfig)
    
                // --- 3. ASR 初始化 ---
                val asrConfig = OnlineRecognizerConfig().apply {
                    modelConfig = OnlineModelConfig().apply {
                        transducer = OnlineTransducerModelConfig().apply {
                            encoder = "model-asr/encoder-epoch-99-avg-1.int8.onnx"
                            decoder = "model-asr/decoder-epoch-99-avg-1.int8.onnx"
                            joiner = "model-asr/joiner-epoch-99-avg-1.int8.onnx"
                        }
                        tokens = "model-asr/tokens.txt"
                        // 【关键修复】同上，设为空字符串
                        modelType = "" 
                        numThreads = 2
                    }
                    featConfig = FeatureConfig().apply { sampleRate = SAMPLE_RATE; featureDim = 80 }
                    enableEndpoint = true 
                }
                this.recognizer = OnlineRecognizer(am, asrConfig)
    
                // --- 4. 声纹 ---
                this.speakerExtractor = SpeakerEmbeddingExtractor(am, SpeakerEmbeddingExtractorConfig(
                    "model-speaker/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx",
                    1, false, "cpu"
                ))
    
			  // --- 增加这一行 ---
			    loadAllSpeakers(context) 
			    
			    Log.i("ZeMo", "✅ 初始化成功，已加载声纹数: ${speakerDatabase.size}")
			    return true
            } catch (e: Exception) {
                Log.e("ZeMo", "❌ 初始化失败: ${e.message}")
                // 发生异常时，主动清理，防止垃圾回收时 Mutex 报错
                // release()
                false
            }
        }

		// 完善加载逻辑
		private fun loadAllSpeakers(context: Context) {
			val dir = File(context.filesDir, "speakers")
			if (!dir.exists()) return
			
			speakerDatabase.clear() // 先清空，防止重复
			dir.listFiles()?.forEach { file ->
				if (file.extension == "bin") {
					try {
						DataInputStream(FileInputStream(file)).use { dis ->
							val size = dis.readInt()
							val emb = FloatArray(size)
							for (i in 0 until size) emb[i] = dis.readFloat()
							speakerDatabase[file.nameWithoutExtension] = emb
							Log.i("ZeMo", "已从硬盘载入声纹: ${file.nameWithoutExtension}")
						}
					} catch (e: Exception) {
						Log.e("ZeMo", "加载声纹文件失败: ${file.name}", e)
					}
				}
			}
		}
			fun processAudio(buffer: ShortArray, size: Int): AudioResult {
				val samples = FloatArray(size)
				var sum = 0.0
				for (i in 0 until size) {
					val f = buffer[i] / 32768.0f
					samples[i] = f
					sum += if (f > 0) f else -f
					circleBuffer[bufferIdx] = f
					bufferIdx = (bufferIdx + 1) % circleBuffer.size
				}
				return AudioResult(samples, (sum / size * 1000).toInt())
			}

			// 所有涉及 Native 指针的方法都加上 synchronized
		 fun decode(samples: FloatArray, context: Context): String = synchronized(lock) {
				 // 1. 检查对象是否存在
				 if (voiceDetector == null) {
					 Log.e(TAG, "❌ 错误: VAD 没初始化成功，请检查 asset 路径")
					 return ""
				 }
			voiceDetector!!.acceptWaveform(samples)
			 val vd = voiceDetector ?: return ""
			 // --- 第一层：VAD (低功耗人声检测) ---
			 vd.acceptWaveform(samples)
			 if (!vd.isSpeechDetected()) {
				 // 如果持续安静，不打印日志以免刷屏
				 return "" 
			 }
			 // 只有检测到人声，才会往下走
			 Log.d(TAG, "[Layer 1: VAD] 检测到活动人声...") 
				
			 if (isRegistering) {
				  Log.d(TAG, "[Layer 11: Register] 注册模式...") 
				 // 注册模式略过后续，直接采集
				 return processRegister(samples)
			 }
		 
			 if (currentState == 0) {
				 // --- 第二层：KWS (唤醒词识别) ---
				 val spotter = keywordSpotter ?: return ""
				 if (kwsStream == null) kwsStream = spotter.createStream()
				 val stream = kwsStream!!
				 stream.acceptWaveform(samples, SAMPLE_RATE)
				 
				 while (spotter.isReady(stream)) {
					 spotter.decode(stream)
					 val res = spotter.getResult(stream)
					 if (res.keyword.isNotBlank()) {
						 writeLog("🔥 [Layer 2: KWS] 匹配到关键词: ${res.keyword} (得分: ${res.tokens.size})")
						 
						 // --- 第三层：SV (声纹识别) ---
						 writeLog("⏳ [Layer 3: SV] 开始进行主人声纹比对...")
						 val user = identifySpeaker()
						 
						 if (user != null) {
							 writeLog("✅ [Layer 3: SV] 验证通过！确认身份: $user")
							 spotter.reset(stream)
							 currentState = 1 // 只有到这一步，才进入高功耗 ASR 模式
							 vd.reset()
							 return "WAKEUP_SUCCESS:$user"
						 } else {
							 writeLog("❌ [Layer 3: SV] 验证失败: 声纹不匹配 (陌生人)")
							 spotter.reset(stream)
							 return "WAKEUP_STRANGER"
						 }
					 }
				 }
			 } else {
				 // --- 第四层：ASR (高功耗命令识别) ---
				 val rec = recognizer ?: return ""
				 if (asrStream == null) asrStream = rec.createStream()
				 val stream = asrStream!!
				 stream.acceptWaveform(samples, SAMPLE_RATE)
				 
				 while (rec.isReady(stream)) rec.decode(stream)
				 val res = rec.getResult(stream)
				 
				 if (rec.isEndpoint(stream) || !vd.isSpeechDetected()) {
					 val text = res.text.trim()
					 writeLog("🎤 [Layer 4: ASR] 命令结束识别: $text")
					 rec.reset(stream)
					 currentState = 0 // 回到低功耗唤醒模式
					 vd.reset()
					 return if (text.isNotBlank()) "COMMAND_FINAL:$text" else ""
				 } else if (res.text.isNotBlank()) {
					 return "COMMAND_PARTIAL:${res.text}"
				 }
			 }
			 return ""
		 }
		private fun processRegister(samples: FloatArray): String {
			val extractor = speakerExtractor ?: return ""
			var spkStream: OnlineStream? = null
			try {
				spkStream = extractor.createStream()
				spkStream.acceptWaveform(samples, SAMPLE_RATE)
				val emb = extractor.compute(spkStream)
				
				synchronized(registerEmbeddings) {
					  registerEmbeddings.add(emb)
					  // 简化返回格式，确保前端好解析
					  return "REGISTERING:${registerEmbeddings.size}"
				  }
			} catch (e: Exception) {
				Log.e(TAG, "采集异常: ${e.message}")
			} finally {
				spkStream?.release()
			}
			return ""
		}
		 // 细化声纹比对日志
		 private fun identifySpeaker(): String? {
			 val extractor = speakerExtractor ?: return null
			 var spkStream: OnlineStream? = null
			 try {
				 spkStream = extractor.createStream()
				 // 获取最近 2 秒的音频缓冲区（circleBuffer）
				 val lastAudio = FloatArray(circleBuffer.size)
				 val firstPart = circleBuffer.size - bufferIdx
				 System.arraycopy(circleBuffer, bufferIdx, lastAudio, 0, firstPart)
				 System.arraycopy(circleBuffer, 0, lastAudio, firstPart, bufferIdx)
				 
				 spkStream.acceptWaveform(lastAudio, SAMPLE_RATE)
				 val currentEmb = extractor.compute(spkStream)
				 
				 if (speakerDatabase.isEmpty()) {
					 writeLog("⚠️ [SV 警告] 数据库为空，请先在设置中录制声纹")
					 return null
				 }
		 
				 var maxSim = -1f
				 var bestUser: String? = null
				 
				 for ((name, target) in speakerDatabase) {
					 val sim = cosineSimilarity(currentEmb, target)
					 // 重要：在控制台看这个分数值！
					 Log.i("ZeMo-SV", "对比主人[$name], 当前相似度得分: $sim") 
					 if (sim > maxSim) {
						 maxSim = sim
						 bestUser = name
					 }
				 }
				 
				 // 【关键修改】将原来的 0.4 或更高的值，下调到 0.22 左右测试
				 // 0.22 左右在吵闹环境下比较灵敏，0.35 比较严格
				 return if (maxSim > 0.4f) bestUser else null
			 } finally {
				 spkStream?.release()
			 }
		 }
			 // 3. 新增：完成注册并保存到本地
			 fun finishRegistration(context: Context): String? {
				 if (registerEmbeddings.isEmpty()) return null
				 
				 // 计算平均特征值 (简易平均)
				 val dim = registerEmbeddings[0].size
				 val avgEmb = FloatArray(dim)
				 for (emb in registerEmbeddings) {
					 for (i in 0 until dim) avgEmb[i] += emb[i]
				 }
				 for (i in 0 until dim) avgEmb[i] /= registerEmbeddings.size.toFloat()
			 
				 // 保存到内存
				 speakerDatabase[currentRegisterName] = avgEmb
				 
				 // 持久化到文件
				 val dir = File(context.filesDir, "speakers")
				 if (!dir.exists()) dir.mkdirs()
				 val file = File(dir, "${currentRegisterName}.bin")
				 try {
					 DataOutputStream(FileOutputStream(file)).use { dos ->
						 dos.writeInt(dim)
						 for (f in avgEmb) dos.writeFloat(f)
					 }
					 isRegistering = false
					 registerEmbeddings.clear()
					 return currentRegisterName
				 } catch (e: Exception) {
					 return null
				 }
			 }
			   // private fun identifySpeaker(): String? {
			   // val extractor = speakerExtractor ?: return null
			   //   var spkStream: OnlineStream? = null
			   //   try {
			   //       spkStream = extractor.createStream()
			   //       val lastAudio = FloatArray(circleBuffer.size)
			   //         val firstPart = circleBuffer.size - bufferIdx
			   //         System.arraycopy(circleBuffer, bufferIdx, lastAudio, 0, firstPart)
			   //         System.arraycopy(circleBuffer, 0, lastAudio, firstPart, bufferIdx)
			   //        System.arraycopy(circleBuffer, bufferIdx, lastAudio, 0, circleBuffer.size - bufferIdx)
			   //              System.arraycopy(circleBuffer, 0, lastAudio, circleBuffer.size - bufferIdx, bufferIdx)
					
			   //              spkStream.acceptWaveform(lastAudio, SAMPLE_RATE)
			   //              val currentEmb = extractor.compute(spkStream)
							
			   //              if (speakerDatabase.isEmpty()) {
			   //                  Log.w("ZeMo", "⚠️ 声纹库为空，请先录制！")
			   //                  return null
			   //              }
					
			   //              var maxSim = -1f
			   //              var bestUser: String? = null
					
			   //              for ((name, target) in speakerDatabase) {
			   //                  val sim = cosineSimilarity(currentEmb, target)
			   //                  // --- 核心调试日志：在控制台看这个分数 ---
			   //                  Log.i("ZeMo", "对比声纹 [$name], 相似度分数: $sim")
			   //                  if (sim > maxSim) {
			   //                      maxSim = sim
			   //                      bestUser = name
			   //                  }
			   //              }
					
			   //              // --- 关键调整：测试阶段把 0.65 改成 0.4 ---
			   //              // 如果 0.4 能唤醒，说明算法通了，只是你录制的环境噪音大或者离得远
			   //              return if (maxSim > 0.40f) bestUser else null 
			   //          } finally {
			   //              spkStream?.release()
			   //          }
			   // }

			fun setRegisterMode(en: Boolean, name: String) {
				this.isRegistering = en
				this.currentRegisterName = name
				this.registerEmbeddings.clear()
			}
			
			fun setWakeUpMode(en: Boolean) {
				this.currentState = 0
			}
			// 4. 获取所有已注册用户列表
			fun getRegisteredSpeakers(): List<String> {
				return speakerDatabase.keys().toList()
			}
			// 5. 删除声纹
			fun removeSpeaker(context: Context, name: String) {
				speakerDatabase.remove(name)
				val file = File(context.filesDir, "speakers/${name}.bin")
				if (file.exists()) file.delete()
			}
			private fun cosineSimilarity(v1: FloatArray, v2: FloatArray): Float {
				var dotProduct = 0.0f
				var nA = 0.0f
				var nB = 0.0f
				for (i in v1.indices) {
					dotProduct += v1[i] * v2[i]
					nA += v1[i] * v1[i]
					nB += v2[i] * v2[i]
				}
				return if (nA > 0 && nB > 0) dotProduct / (sqrt(nA.toDouble()) * sqrt(nB.toDouble())).toFloat() else 0.0f
			}

			// private fun loadAllSpeakers(context: Context) {
			//     val dir = File(context.filesDir, "speakers")
			//     if (!dir.exists()) dir.mkdirs()
			//     dir.listFiles()?.forEach { file ->
			//         try {
			//             val name = file.nameWithoutExtension
			//             DataInputStream(FileInputStream(file)).use { dis ->
			//                 val size = dis.readInt()
			//                 val emb = FloatArray(size)
			//                 for (i in 0 until size) emb[i] = dis.readFloat()
			//                 speakerDatabase[name] = emb
			//             }
			//         } catch (e: Exception) {
			//             e.printStackTrace()
			//         }
			//     }
			// }

		// fun release() = synchronized(lock) {
		//     writeLog("正在执行资源安全释放...")
		//     try {
		//         // 1. 先释放所有的 Stream (这是最容易引起 Mutex 崩溃的地方)
		//         kwsStream?.release()
		//         kwsStream = null
		//         asrStream?.release()
		//         asrStream = null
				
		//         // 2. 释放各个引擎
		//         voiceDetector?.release()
		//         voiceDetector = null
				
		//         keywordSpotter?.release()
		//         keywordSpotter = null
				
		//         recognizer?.release()
		//         recognizer = null
				
		//         speakerExtractor?.release()
		//         speakerExtractor = null

		//         isInitialized = false
		//         writeLog("✅ 资源已完全回收")
		//     } catch (e: Exception) {
		//         writeLog("❌ 释放资源时报错: ${e.message}")
		//     }
		// }
}