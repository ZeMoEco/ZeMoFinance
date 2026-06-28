import { image } from '@kit.ImageKit'
import fs from '@ohos.file.fs'
import { mindSporeLite } from '@kit.MindSporeLiteKit'

interface OcrBox {
  left: number
  top: number
  right: number
  bottom: number
}

interface RecResult {
  text: string
  confidence: number
}

interface PixelInfo {
  width: number
  height: number
  pixels: Uint8Array
  pixelFormat: image.PixelMapFormat
}

interface PrepResult {
  data: Float32Array
  width: number
  height: number
  scaleX: number
  scaleY: number
}

interface PpOcrV6Item {
  text: string
  confidence: number
  box: OcrBox
}

interface PpOcrV6Line {
  t: string
  c: number
  b: number[]
}

interface PpOcrV6Payload {
  lines: PpOcrV6Line[]
}

let detModel: mindSporeLite.Model | null = null
let recModel: mindSporeLite.Model | null = null
let recDict: string[] = []
let loadedKey = ''
const DET_TARGET_WIDTH = 768
const DET_TARGET_HEIGHT = 1088
const REC_MODEL_BATCH = 8
const REC_MODEL_C = 3
const REC_MODEL_H = 48
const REC_MODEL_W = 320
const REC_CONFIDENCE_THRESHOLD = 0.45
const DB_MASK_HORIZONTAL_DILATE = 4
const LINE_MERGE_ROW_CENTER_RATIO = 0.9
const LINE_MERGE_ROW_CENTER_MIN = 18
const LINE_MERGE_COMPACT_GAP_RATIO = 2.0
const LINE_MERGE_CLOSE_GAP_RATIO = 4.0
const LINE_MERGE_CLOSE_GAP_MIN = 28
const LINE_MERGE_CLOSE_GAP_MAX = 84
const LINE_MERGE_CLOSE_GAP_WIDTH_RATIO = 0.075
const LINE_MERGE_WIDE_BOX_RATIO = 8.0
const LINE_MERGE_WIDE_BOX_MIN_WIDTH = 92
const LINE_MERGE_MAX_ROW_WIDTH_RATIO = 0.62

function normalizeFsPath(path: string): string {
  let value = (path ?? '').toString().replace(/\\/g, '/')
  if (value.startsWith('file://')) value = value.substring(7)
  return value
}

function readTextFile(path: string): string {
  try {
    return fs.readTextSync(normalizeFsPath(path), { encoding: 'utf-8' })
  } catch (_e) {
    return ''
  }
}

function loadDict(dictPath: string): string[] {
  const text = readTextFile(dictPath)
  const yamlDict = parseYamlCharacterDict(text)
  if (yamlDict.length == 0) {
    throw new Error(`官方 inference.yml 未找到 character_dict：${dictPath}`)
  }
  return appendSpaceChar(yamlDict)
}

function appendSpaceChar(dict: string[]): string[] {
  for (let i = 0; i < dict.length; i++) {
    if (dict[i] == ' ') return dict
  }
  dict.push(' ')
  return dict
}

function parseYamlCharacterDict(text: string): string[] {
  if (text.indexOf('character_dict:') < 0) return []
  const lines = text.replace(/\r\n/g, '\n').replace(/\r/g, '\n').split('\n')
  const dict: string[] = []
  let inDict = false
  let dictIndent = -1
  for (let i = 0; i < lines.length; i++) {
    const rawLine = lines[i]
    const line = trimYamlSyntax(rawLine)
    if (!inDict) {
      if (line == 'character_dict:') {
        inDict = true
        dictIndent = countLeadingSpaces(rawLine)
      }
      continue
    }
    if (line == '' || line.startsWith('#')) continue
    if (line.startsWith('-')) {
      dict.push(cleanYamlScalar(yamlListItemValue(line)))
      continue
    }
    const indent = countLeadingSpaces(rawLine)
    if (indent <= dictIndent) break
  }
  return dict
}

function yamlListItemValue(line: string): string {
  let value = line.length > 1 ? line.substring(1) : ''
  if (value.startsWith(' ') || value.startsWith('\t')) value = value.substring(1)
  return trimYamlSyntax(value)
}

function cleanYamlScalar(value: string): string {
  let result = trimYamlSyntax(value)
  if (result.length >= 2 && result.startsWith("'") && result.endsWith("'")) {
    result = result.substring(1, result.length - 1)
    while (result.indexOf("''") >= 0) result = result.replace("''", "'")
    return result
  }
  if (result.length >= 2 && result.startsWith('"') && result.endsWith('"')) {
    result = result.substring(1, result.length - 1)
    result = result.replace('\\"', '"')
    result = result.replace('\\\\', '\\')
    return result
  }
  return result
}

function trimYamlSyntax(value: string): string {
  let start = 0
  let end = value.length
  while (start < end) {
    const c = value.substring(start, start + 1)
    if (c != ' ' && c != '\t') break
    start += 1
  }
  while (end > start) {
    const c = value.substring(end - 1, end)
    if (c != ' ' && c != '\t') break
    end -= 1
  }
  return value.substring(start, end)
}

function countLeadingSpaces(value: string): number {
  let count = 0
  for (let i = 0; i < value.length; i++) {
    if (value.substring(i, i + 1) != ' ') break
    count += 1
  }
  return count
}

async function loadModel(modelPath: string): Promise<mindSporeLite.Model> {
  let context: mindSporeLite.Context = {}
  context.target = ['cpu']
  context.cpu = {}
  context.cpu.threadNum = 2
  context.cpu.threadAffinityMode = 1
  context.cpu.precisionMode = 'enforce_fp32'
  const model: mindSporeLite.Model = await mindSporeLite.loadModelFromFile(normalizeFsPath(modelPath), context)
  if (model == null) throw new Error(`MindSpore Lite 加载模型失败：${modelPath}`)
  return model
}

async function ensureLoaded(detPath: string, recPath: string, dictPath: string): Promise<void> {
  const key = `${detPath}|${fileSizeSignature(detPath)}\n${recPath}|${fileSizeSignature(recPath)}\n${dictPath}|${fileSizeSignature(dictPath)}`
  if (detModel != null && recModel != null && loadedKey == key) return
  detModel = await loadModel(detPath)
  recModel = await loadModel(recPath)
  recDict = loadDict(dictPath)
  loadedKey = key
  console.log(`[ZemoOcr][MindSpore] model inputs det=${modelInputShapeText(detModel)}, rec=${modelInputShapeText(recModel)}`)
}

function fileSizeSignature(path: string): string {
  try {
    const stat = fs.statSync(normalizeFsPath(path))
    if (stat != null) return stat.size.toString()
  } catch (_e) {}
  return 'missing'
}

async function readPixelInfo(pixelMap: image.PixelMap): Promise<PixelInfo> {
  const info = await pixelMap.getImageInfo()
  const width = parseInt(info.size.width.toString())
  const height = parseInt(info.size.height.toString())
  const buffer = new ArrayBuffer(pixelMap.getPixelBytesNumber())
  await pixelMap.readPixelsToBuffer(buffer)
  return { width, height, pixels: new Uint8Array(buffer), pixelFormat: info.pixelFormat }
}

function pixelOffset(src: PixelInfo, x: number, y: number): number {
  return (y * src.width + x) * 4
}

function pixelBlue(src: PixelInfo, off: number): number {
  return src.pixelFormat == image.PixelMapFormat.RGBA_8888 ? src.pixels[off + 2] : src.pixels[off]
}

function pixelGreen(src: PixelInfo, off: number): number {
  return src.pixels[off + 1]
}

function pixelRed(src: PixelInfo, off: number): number {
  return src.pixelFormat == image.PixelMapFormat.RGBA_8888 ? src.pixels[off] : src.pixels[off + 2]
}

function preprocessDet(src: PixelInfo): PrepResult {
  const targetW = DET_TARGET_WIDTH
  const targetH = DET_TARGET_HEIGHT
  const ratio = Math.min(targetW / Math.max(1, src.width), targetH / Math.max(1, src.height))
  const rw = Math.max(32, Math.min(targetW, Math.round((src.width * ratio) / 32.0) * 32))
  const rh = Math.max(32, Math.min(targetH, Math.round((src.height * ratio) / 32.0) * 32))
  const scaleX = src.width / rw
  const scaleY = src.height / rh
  const count = targetW * targetH
  const data = new Float32Array(3 * count)
  const mean = [0.485, 0.456, 0.406]
  const std = [0.229, 0.224, 0.225]
  for (let y = 0; y < rh; y++) {
    const sy = Math.min(src.height - 1, Math.floor(y * scaleY))
    for (let x = 0; x < rw; x++) {
      const sx = Math.min(src.width - 1, Math.floor(x * scaleX))
      const off = pixelOffset(src, sx, sy)
      const dst = y * targetW + x
      const b = pixelBlue(src, off) / 255.0
      const g = pixelGreen(src, off) / 255.0
      const r = pixelRed(src, off) / 255.0
      data[dst] = (b - mean[0]) / std[0]
      data[dst + count] = (g - mean[1]) / std[1]
      data[dst + 2 * count] = (r - mean[2]) / std[2]
    }
  }
  return { data, width: targetW, height: targetH, scaleX, scaleY }
}

function cropPixelInfo(src: PixelInfo, box: OcrBox): PixelInfo | null {
  const left = Math.max(0, Math.floor(box.left))
  const top = Math.max(0, Math.floor(box.top))
  const right = Math.min(src.width, Math.floor(box.right))
  const bottom = Math.min(src.height, Math.floor(box.bottom))
  const width = right - left
  const height = bottom - top
  if (width <= 0 || height <= 0) return null
  const pixels = new Uint8Array(width * height * 4)
  for (let y = 0; y < height; y++) {
    const srcBase = pixelOffset(src, left, top + y)
    const dstBase = y * width * 4
    for (let x = 0; x < width * 4; x++) {
      pixels[dstBase + x] = src.pixels[srcBase + x]
    }
  }
  return { width, height, pixels, pixelFormat: src.pixelFormat }
}

function preprocessRec(src: PixelInfo, targetH: number = 48, targetW: number = 320): PrepResult {
  const ratio = targetH / Math.max(1, src.height)
  const newW = Math.max(1, Math.min(targetW, Math.round(src.width * ratio)))
  const count = targetH * targetW
  const data = new Float32Array(3 * count)
  for (let y = 0; y < targetH; y++) {
    const sy = Math.min(src.height - 1, Math.floor(y * src.height / targetH))
    for (let x = 0; x < newW; x++) {
      const sx = Math.min(src.width - 1, Math.floor(x * src.width / newW))
      const off = pixelOffset(src, sx, sy)
      const dst = y * targetW + x
      const b = pixelBlue(src, off) / 255.0
      const g = pixelGreen(src, off) / 255.0
      const r = pixelRed(src, off) / 255.0
      data[dst] = (b - 0.5) / 0.5
      data[dst + count] = (g - 0.5) / 0.5
      data[dst + 2 * count] = (r - 0.5) / 0.5
    }
  }
  return { data, width: targetW, height: targetH, scaleX: 1, scaleY: 1 }
}

function tensorDataToFloats(value: mindSporeLite.MSTensor): Float32Array {
  try {
    return new Float32Array(value.getData())
  } catch (_e) {}
  return new Float32Array(0)
}

function tensorShape(value: mindSporeLite.MSTensor): number[] {
  try {
    if (Array.isArray(value.shape)) return value.shape.map((v: number): number => Number(v))
  } catch (_e) {}
  return []
}

function modelInputShapeText(model: mindSporeLite.Model | null): string {
  if (model == null) return '[]'
  try {
    const inputs: mindSporeLite.MSTensor[] = model.getInputs()
    if (inputs == null || inputs.length == 0) return '[]'
    return shapeText(tensorShape(inputs[0]))
  } catch (_e) {}
  return '[]'
}

function shapeText(shape: number[]): string {
  return '[' + shape.join(',') + ']'
}

function hasDynamicShape(shape: number[]): boolean {
  for (let i = 0; i < shape.length; i++) {
    if (shape[i] < 0) return true
  }
  return false
}

function errorText(value: Object | null): string {
  if (value == null) return 'null'
  try {
    const err = value as Error
    const name = err.name != null ? err.name.toString() : ''
    const message = err.message != null ? err.message.toString() : ''
    const stack = err.stack != null ? err.stack.toString() : ''
    if (name != '' || message != '' || stack != '') {
      let text = ''
      if (name != '') text += name
      if (message != '') text += (text == '' ? '' : ': ') + message
      if (stack != '') text += (text == '' ? '' : '\n') + stack
      return text
    }
  } catch (_e0) {}
  try {
    const json = JSON.stringify(value)
    if (json != '{}') return json
  } catch (_e1) {}
  try {
    return value.toString()
  } catch (_e2) {}
  return '[unknown error]'
}

function toInputArrayBuffer(inputData: Float32Array): ArrayBuffer {
  const buffer = new ArrayBuffer(inputData.byteLength)
  const copied = new Float32Array(buffer)
  copied.set(inputData)
  return buffer
}

function usesNhwcInput(inputShape: number[]): boolean {
  return inputShape.length == 4 && inputShape[3] == 3
}

function resolveInputShape(nchwShape: number[], inputShape: number[]): number[] {
  if (usesNhwcInput(inputShape) && nchwShape.length == 4) {
    return [nchwShape[0], nchwShape[2], nchwShape[3], nchwShape[1]]
  }
  return nchwShape
}

function nchwToNhwc(inputData: Float32Array, nchwShape: number[]): Float32Array {
  if (nchwShape.length != 4) return inputData
  const n = nchwShape[0]
  const c = nchwShape[1]
  const h = nchwShape[2]
  const w = nchwShape[3]
  if (n <= 0 || c != 3) return inputData
  const out = new Float32Array(inputData.length)
  const plane = h * w
  for (let b = 0; b < n; b++) {
    const srcBatch = b * c * plane
    const dstBatch = b * plane * c
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const srcBase = y * w + x
        const dstBase = (y * w + x) * c
        for (let ch = 0; ch < c; ch++) {
          out[dstBatch + dstBase + ch] = inputData[srcBatch + ch * plane + srcBase]
        }
      }
    }
  }
  return out
}

function shouldLogModelRun(label: string): boolean {
  return label == 'det' || label == 'det#0' || label == 'rec#0' || label == 'recBatch#0'
}

async function runModel(model: mindSporeLite.Model | null, inputData: Float32Array, nchwShape: number[], label: string): Promise<mindSporeLite.MSTensor> {
  if (model == null) throw new Error('MindSpore Lite 模型未加载')
  let inputs: mindSporeLite.MSTensor[] = model.getInputs()
  if (inputs == null || inputs.length == 0) throw new Error('MindSpore Lite 模型输入为空')
  const inputShapeBefore = tensorShape(inputs[0])
  const targetShape = resolveInputShape(nchwShape, inputShapeBefore)
  const modelInputData = usesNhwcInput(inputShapeBefore) ? nchwToNhwc(inputData, nchwShape) : inputData
  try {
    await model.resize(inputs, [targetShape])
    inputs = model.getInputs()
  } catch (e) {
    console.warn(`[ZemoOcr][MindSpore] ${label} resize failed: target=${shapeText(targetShape)}, nchw=${shapeText(nchwShape)}, inputBefore=${shapeText(inputShapeBefore)}, err=${errorText(e as Object)}`)
  }
  if (inputs == null || inputs.length == 0) throw new Error(`${label} resize 后模型输入为空`)
  const input = inputs[0]
  const inputShapeAfter = tensorShape(input)
  if (hasDynamicShape(inputShapeAfter)) {
    throw new Error(`${label} 输入仍是动态 shape：before=${shapeText(inputShapeBefore)}, target=${shapeText(targetShape)}, after=${shapeText(inputShapeAfter)}。当前 ArkTS MindSpore Lite 路径未能将动态 shape 固定为目标尺寸，请导入固定输入尺寸的 .ms 模型，或改用 Native 侧 OH_AI_ModelResize。`)
  }
  const inputBuffer: ArrayBuffer = toInputArrayBuffer(modelInputData)
  const logRun = shouldLogModelRun(label)
  try {
    if (logRun) {
      console.log(`[ZemoOcr][MindSpore] ${label} setData target=${shapeText(targetShape)}, nchw=${shapeText(nchwShape)}, inputBefore=${shapeText(inputShapeBefore)}, inputAfter=${shapeText(inputShapeAfter)}, layout=${usesNhwcInput(inputShapeBefore) ? 'NHWC' : 'NCHW'}, floats=${modelInputData.length}, bytes=${inputBuffer.byteLength}`)
    }
    input.setData(inputBuffer)
  } catch (e) {
    throw new Error(`${label} setData failed: ${errorText(e as Object)}, target=${shapeText(targetShape)}, inputAfter=${shapeText(inputShapeAfter)}, bytes=${inputBuffer.byteLength}`)
  }
  let outputs: mindSporeLite.MSTensor[] = []
  let predictReturnLen = -1
  try {
    const predicted: mindSporeLite.MSTensor[] = await model.predict(inputs)
    if (predicted != null) {
      predictReturnLen = predicted.length
      outputs = predicted
    }
  } catch (e) {
    throw new Error(`${label} predict failed: ${errorText(e as Object)}, target=${shapeText(targetShape)}, inputAfter=${shapeText(inputShapeAfter)}`)
  }
  if (outputs == null || outputs.length == 0) {
    throw new Error(`MindSpore Lite 模型输出为空：predictReturnLen=${predictReturnLen}`)
  }
  if (logRun) console.log(`[ZemoOcr][MindSpore] ${label} outputShape=${shapeText(tensorShape(outputs[0]))}`)
  return outputs[0]
}

function dilateHorizontalMask(source: boolean[], h: number, w: number): boolean[] {
  if (DB_MASK_HORIZONTAL_DILATE <= 0) return source
  const out = new Array<boolean>(source.length)
  for (let i = 0; i < source.length; i++) out[i] = false
  for (let y = 0; y < h; y++) {
    const rowBase = y * w
    for (let x = 0; x < w; x++) {
      const idx = rowBase + x
      if (!source[idx]) continue
      const left = Math.max(0, x - DB_MASK_HORIZONTAL_DILATE)
      const right = Math.min(w - 1, x + DB_MASK_HORIZONTAL_DILATE)
      for (let dx = left; dx <= right; dx++) out[rowBase + dx] = true
    }
  }
  return out
}

function postprocessDB(data: Float32Array, h: number, w: number, sx: number, sy: number, sourceWidth: number): OcrBox[] {
  const boxes: OcrBox[] = []
  const total = h * w
  const baseMask = new Array<boolean>(total)
  const visited = new Array<boolean>(total)
  const thresh = 0.10
  for (let i = 0; i < total; i++) {
    baseMask[i] = data[i] > thresh
    visited[i] = false
  }
  const mask = dilateHorizontalMask(baseMask, h, w)
  for (let i = 0; i < total; i++) {
    if (!mask[i] || visited[i]) continue
    let minX = i % w
    let maxX = minX
    let minY = Math.floor(i / w)
    let maxY = minY
    const queue: number[] = [i]
    visited[i] = true
    let head = 0
    while (head < queue.length) {
      const curr = queue[head++]
      const cx = curr % w
      const cy = Math.floor(curr / w)
      minX = Math.min(minX, cx)
      maxX = Math.max(maxX, cx)
      minY = Math.min(minY, cy)
      maxY = Math.max(maxY, cy)
      const neighbors = [curr - 1, curr + 1, curr - w, curr + w]
      for (let n = 0; n < neighbors.length; n++) {
        const nb = neighbors[n]
        if (nb < 0 || nb >= total || !mask[nb] || visited[nb]) continue
        const nx = nb % w
        if (Math.abs(nx - cx) > 1) continue
        visited[nb] = true
        queue.push(nb)
      }
    }
    if (queue.length < 8) continue
    const pad = 3
    boxes.push({
      left: Math.floor(Math.max(0, minX - pad) * sx),
      top: Math.floor(Math.max(0, minY - pad) * sy),
      right: Math.floor(Math.min(w, maxX + pad) * sx),
      bottom: Math.floor(Math.min(h, maxY + pad) * sy)
    })
  }
  const merged = mergeCloseLineBoxes(boxes, sourceWidth)
  if (boxes.length != merged.length) console.log(`[ZemoOcr][MindSpore] postprocess merge boxes=${boxes.length}->${merged.length}, sourceWidth=${sourceWidth}`)
  return merged
}

function clampNumber(value: number, min: number, max: number): number {
  if (value < min) return min
  if (value > max) return max
  return value
}

function dbOutputHeight(shape: number[], fallback: number): number {
  if (shape.length >= 4 && shape[1] == 1) return shape[2]
  if (shape.length >= 4 && shape[3] == 1) return shape[1]
  return fallback
}

function dbOutputWidth(shape: number[], fallback: number): number {
  if (shape.length >= 4 && shape[1] == 1) return shape[3]
  if (shape.length >= 4 && shape[3] == 1) return shape[2]
  return fallback
}

function offsetAndClampBox(box: OcrBox, offsetLeft: number, offsetTop: number, src: PixelInfo): OcrBox | null {
  const left = Math.floor(clampNumber(box.left + offsetLeft, 0, src.width))
  const right = Math.floor(clampNumber(box.right + offsetLeft, 0, src.width))
  const top = Math.floor(clampNumber(box.top + offsetTop, 0, src.height))
  const bottom = Math.floor(clampNumber(box.bottom + offsetTop, 0, src.height))
  if (right <= left || bottom <= top) return null
  return { left, top, right, bottom }
}

async function detectSingleImageBoxes(src: PixelInfo, label: string, offsetLeft: number, offsetTop: number, fullSource: PixelInfo): Promise<OcrBox[]> {
  const detPrep = preprocessDet(src)
  const detOutput = await runModel(detModel, detPrep.data, [1, 3, detPrep.height, detPrep.width], label)
  const detData = tensorDataToFloats(detOutput)
  const detShape = tensorShape(detOutput)
  const outH = dbOutputHeight(detShape, detPrep.height)
  const outW = dbOutputWidth(detShape, detPrep.width)
  const localBoxes = postprocessDB(detData, outH, outW, detPrep.scaleX, detPrep.scaleY, fullSource.width)
  const boxes: OcrBox[] = []
  for (let i = 0; i < localBoxes.length; i++) {
    const box = offsetAndClampBox(localBoxes[i], offsetLeft, offsetTop, fullSource)
    if (box != null) boxes.push(box)
  }
  console.log(`[ZemoOcr][MindSpore] ${label} data=${detData.length}, shape=${shapeText(detShape)}, offsetX=${offsetLeft}, offsetY=${offsetTop}, boxes=${boxes.length}`)
  return boxes
}

function mergeBox(a: OcrBox, b: OcrBox): OcrBox {
  return {
    left: Math.min(a.left, b.left),
    top: Math.min(a.top, b.top),
    right: Math.max(a.right, b.right),
    bottom: Math.max(a.bottom, b.bottom)
  }
}

function sortOcrBoxes(boxes: OcrBox[]): OcrBox[] {
  const sorted: OcrBox[] = boxes.slice()
  sorted.sort((a: OcrBox, b: OcrBox): number => {
    const dy = Math.abs(a.top - b.top)
    return dy < 20 ? a.left - b.left : a.top - b.top
  })
  return sorted
}

function verticalOverlapRatio(a: OcrBox, b: OcrBox): number {
  const top = Math.max(a.top, b.top)
  const bottom = Math.min(a.bottom, b.bottom)
  const overlap = Math.max(0, bottom - top)
  const minHeight = Math.min(Math.max(1, a.bottom - a.top), Math.max(1, b.bottom - b.top))
  return overlap / minHeight
}

function horizontalGap(a: OcrBox, b: OcrBox): number {
  if (a.right < b.left) return b.left - a.right
  if (b.right < a.left) return a.left - b.right
  return 0
}

function boxWidth(box: OcrBox): number {
  return Math.max(1, box.right - box.left)
}

function boxHeight(box: OcrBox): number {
  return Math.max(1, box.bottom - box.top)
}

function boxCenterY(box: OcrBox): number {
  return (box.top + box.bottom) / 2.0
}

function isWideLinePart(box: OcrBox): boolean {
  const height = boxHeight(box)
  return boxWidth(box) >= Math.max(LINE_MERGE_WIDE_BOX_MIN_WIDTH, height * LINE_MERGE_WIDE_BOX_RATIO)
}

function mergedBoxWidth(a: OcrBox, b: OcrBox): number {
  return Math.max(a.right, b.right) - Math.min(a.left, b.left)
}

function sameTextRow(a: OcrBox, b: OcrBox): boolean {
  const minHeight = Math.min(Math.max(1, a.bottom - a.top), Math.max(1, b.bottom - b.top))
  if (verticalOverlapRatio(a, b) >= 0.28) return true
  const centerDiff = Math.abs(boxCenterY(a) - boxCenterY(b))
  return centerDiff <= Math.max(LINE_MERGE_ROW_CENTER_MIN, minHeight * LINE_MERGE_ROW_CENTER_RATIO)
}

function closeGapLimit(minHeight: number, sourceWidth: number): number {
  const byHeight = Math.max(LINE_MERGE_CLOSE_GAP_MIN, minHeight * LINE_MERGE_CLOSE_GAP_RATIO)
  const byWidth = sourceWidth > 0 ? sourceWidth * LINE_MERGE_CLOSE_GAP_WIDTH_RATIO : LINE_MERGE_CLOSE_GAP_MAX
  return Math.min(LINE_MERGE_CLOSE_GAP_MAX, Math.max(byHeight, byWidth))
}

function shouldMergeLineBox(a: OcrBox, b: OcrBox, sourceWidth: number): boolean {
  if (!sameTextRow(a, b)) return false
  const minHeight = Math.min(boxHeight(a), boxHeight(b))
  const gap = horizontalGap(a, b)
  const compactGap = Math.max(12, minHeight * LINE_MERGE_COMPACT_GAP_RATIO)
  if (gap <= compactGap) return true
  if (gap > closeGapLimit(minHeight, sourceWidth)) return false
  if ((isWideLinePart(a) || isWideLinePart(b)) && gap > Math.max(20, minHeight * 2.2)) return false
  if (sourceWidth > 0 && mergedBoxWidth(a, b) > sourceWidth * LINE_MERGE_MAX_ROW_WIDTH_RATIO) return false
  return true
}

function mergeCloseLineBoxes(boxes: OcrBox[], sourceWidth: number): OcrBox[] {
  const sorted = sortOcrBoxes(boxes)
  const rows: OcrBox[][] = []
  for (let i = 0; i < sorted.length; i++) {
    const box = sorted[i]
    let inserted = false
    for (let j = 0; j < rows.length; j++) {
      const row = rows[j]
      const anchor = row.length > 0 ? row[row.length - 1] : box
      if (sameTextRow(anchor, box)) {
        row.push(box)
        inserted = true
        break
      }
    }
    if (!inserted) rows.push([box])
  }
  const merged: OcrBox[] = []
  for (let r = 0; r < rows.length; r++) {
    const row = rows[r]
    row.sort((a: OcrBox, b: OcrBox): number => a.left - b.left)
    for (let i = 0; i < row.length; i++) {
      const box = row[i]
      if (merged.length == 0 || !shouldMergeLineBox(merged[merged.length - 1], box, sourceWidth)) {
        merged.push(box)
      } else {
        merged[merged.length - 1] = mergeBox(merged[merged.length - 1], box)
      }
    }
  }
  return sortOcrBoxes(merged)
}

async function detectTextBoxes(source: PixelInfo): Promise<OcrBox[]> {
  return await detectSingleImageBoxes(source, 'det', 0, 0, source)
}

function expandBoxForRecognition(box: OcrBox, src: PixelInfo): OcrBox {
  const width = Math.max(1, box.right - box.left)
  const height = Math.max(1, box.bottom - box.top)
  const padX = Math.max(4, Math.floor(Math.max(height * 0.35, width * 0.03)))
  const padY = Math.max(2, Math.floor(height * 0.20))
  return {
    left: Math.floor(clampNumber(box.left - padX, 0, src.width)),
    top: Math.floor(clampNumber(box.top - padY, 0, src.height)),
    right: Math.floor(clampNumber(box.right + padX, 0, src.width)),
    bottom: Math.floor(clampNumber(box.bottom + padY, 0, src.height))
  }
}

function boxText(box: OcrBox): string {
  return `${Math.round(box.left)},${Math.round(box.top)},${Math.round(Math.max(0, box.right - box.left))},${Math.round(Math.max(0, box.bottom - box.top))}`
}

function oneLineText(text: string): string {
  return (text ?? '').toString().replace(/\r\n/g, ' ').replace(/\n/g, ' ')
}

function decodeCtc(data: Float32Array, steps: number, classes: number): RecResult {
  let lastIdx = -1
  let totalConf = 0
  let count = 0
  const chars: string[] = []
  for (let t = 0; t < steps; t++) {
    let maxVal = -10000
    let maxIdx = 0
    const offset = t * classes
    for (let c = 0; c < classes; c++) {
      const value = data[offset + c]
      if (value > maxVal) {
        maxVal = value
        maxIdx = c
      }
    }
    if (maxIdx != 0 && maxIdx != lastIdx) {
      const dictIdx = maxIdx - 1
      if (dictIdx >= 0 && dictIdx < recDict.length) {
        chars.push(recDict[dictIdx])
        totalConf += maxVal
        count += 1
      }
    }
    lastIdx = maxIdx
  }
  return { text: chars.join(''), confidence: count > 0 ? totalConf / count : 0 }
}

function validateRecDictionary(classes: number, dictPath: string): void {
  const expectedDictSize = classes - 1
  if (expectedDictSize <= 0 || recDict.length == expectedDictSize) return
  throw new Error(`rec 字典与模型不匹配：outputClasses=${classes}, dictSize=${recDict.length}, expectedDictSize=${expectedDictSize}, dictPath=${dictPath}, dictBytes=${fileSizeSignature(dictPath)}。PP-OCRv6 small rec 应加载官方 inference.yml。`)
}

function sortPpOcrItems(items: PpOcrV6Item[]): void {
  items.sort((a: PpOcrV6Item, b: PpOcrV6Item): number => {
    const ab = a.box
    const bb = b.box
    const ah = Math.max(1, ab.bottom - ab.top)
    const bh = Math.max(1, bb.bottom - bb.top)
    const avgH = Math.max(1, (ah + bh) * 0.5)
    const acy = (ab.top + ab.bottom) * 0.5
    const bcy = (bb.top + bb.bottom) * 0.5
    const dy = Math.abs(acy - bcy)
    return dy <= Math.max(16, avgH * 0.6) ? ab.left - bb.left : ab.top - bb.top
  })
}

function buildCompactPayload(items: PpOcrV6Item[]): PpOcrV6Payload {
  const lines: PpOcrV6Line[] = []
  for (let i = 0; i < items.length; i++) {
    const item = items[i]
    const box = item.box
    lines.push({
      t: item.text,
      c: item.confidence,
      b: [
        Math.round(box.left),
        Math.round(box.top),
        Math.round(Math.max(0, box.right - box.left)),
        Math.round(Math.max(0, box.bottom - box.top))
      ]
    })
  }
  return { lines }
}

export async function recognizePpOcrV6MindSpore(
  pixelMap: image.PixelMap,
  detPath: string,
  recPath: string,
  dictPath: string
): Promise<string> {
  try {
    await ensureLoaded(detPath, recPath, dictPath)
  } catch (e) {
    throw new Error(`ensureLoaded failed: ${errorText(e as Object)}`)
  }
  console.log(`[ZemoOcr][MindSpore] loaded models, dictSize=${recDict.length}, dictPath=${dictPath}, dictBytes=${fileSizeSignature(dictPath)}`)
  const source = await readPixelInfo(pixelMap)
  console.log(`[ZemoOcr][MindSpore] source width=${source.width}, height=${source.height}, pixelFormat=${source.pixelFormat}`)
  const boxes = await detectTextBoxes(source)
  console.log(`[ZemoOcr][MindSpore] rec start boxes=${boxes.length}`)
  const items: PpOcrV6Item[] = []

  const singleInputLength = REC_MODEL_C * REC_MODEL_H * REC_MODEL_W
  let checkedClasses = -1
  let totalBatchRuns = 0
  let skippedEmptyCrop = 0
  let skippedInvalidSize = 0
  let skippedInvalidShape = 0
  let skippedEmptyText = 0
  let skippedLowConfidence = 0
  let boxIndex = 0

  while (boxIndex < boxes.length) {
    const batchInput = new Float32Array(REC_MODEL_BATCH * singleInputLength)
    const batchBoxes: OcrBox[] = []
    const batchStartIndex = boxIndex

    while (boxIndex < boxes.length && batchBoxes.length < REC_MODEL_BATCH) {
      const rawBox = boxes[boxIndex]
      boxIndex++
      const crop = cropPixelInfo(source, expandBoxForRecognition(rawBox, source))
      if (crop == null) {
        skippedEmptyCrop++
        continue
      }
      const recPrep = preprocessRec(crop, REC_MODEL_H, REC_MODEL_W)
      if (recPrep.height != REC_MODEL_H || recPrep.width != REC_MODEL_W || recPrep.data.length != singleInputLength) {
        skippedInvalidSize++
        continue
      }
      batchInput.set(recPrep.data as Float32Array, batchBoxes.length * singleInputLength)
      batchBoxes.push(rawBox)
    }

    if (batchBoxes.length == 0) continue
    totalBatchRuns++

    const recOutput = await runModel(recModel, batchInput, [REC_MODEL_BATCH, REC_MODEL_C, REC_MODEL_H, REC_MODEL_W], `recBatch#${batchStartIndex}`)
    const recData = tensorDataToFloats(recOutput)
    const shape = tensorShape(recOutput)
    if (shape.length != 3 || shape[0] != REC_MODEL_BATCH || shape[1] <= 0 || shape[2] <= 0) {
      skippedInvalidShape += batchBoxes.length
      console.log(`[ZemoOcr][MindSpore] rec batch invalid output shape=${shapeText(shape)}, realBatch=${batchBoxes.length}`)
      continue
    }

    const steps = shape[1]
    const classes = shape[2]
    if (checkedClasses != classes) {
      validateRecDictionary(classes, dictPath)
      checkedClasses = classes
    }
    const singleOutputLength = steps * classes
    for (let b = 0; b < batchBoxes.length; b++) {
      const oneData = recData.subarray(b * singleOutputLength, (b + 1) * singleOutputLength)
      const rec = decodeCtc(oneData, steps, classes)
      console.log(`[ZemoOcr][MindSpore] recBatch#${batchStartIndex}.${b}/${boxes.length} box=${boxText(batchBoxes[b])}, confidence=${rec.confidence}, text=${oneLineText(rec.text)}`)
      if (rec.text == '') {
        skippedEmptyText++
        continue
      }
      if (rec.confidence < REC_CONFIDENCE_THRESHOLD) {
        skippedLowConfidence++
        continue
      }
      const item: PpOcrV6Item = {
        text: rec.text,
        confidence: rec.confidence,
        box: batchBoxes[b]
      }
      items.push(item)
    }
  }

  sortPpOcrItems(items)
  console.log(`[ZemoOcr][MindSpore] batch done, boxes=${boxes.length}, items=${items.length}, batch=${REC_MODEL_BATCH}, batchRuns=${totalBatchRuns}, skippedEmptyCrop=${skippedEmptyCrop}, skippedInvalidSize=${skippedInvalidSize}, skippedInvalidShape=${skippedInvalidShape}, skippedEmptyText=${skippedEmptyText}, skippedLowConfidence=${skippedLowConfidence}`)
  const payload = buildCompactPayload(items)
  return JSON.stringify(payload)
}
