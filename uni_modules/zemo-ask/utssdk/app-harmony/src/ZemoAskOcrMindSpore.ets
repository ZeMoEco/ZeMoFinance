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

// function decodeCtc(data: Float32Array, steps: number, classes: number): RecResult {
//   let lastIdx = -1
//   let totalConf = 0
//   let count = 0
//   const chars: string[] = []
//   for (let t = 0; t < steps; t++) {
//     let maxVal = -10000
//     let maxIdx = 0
//     const offset = t * classes
//     for (let c = 0; c < classes; c++) {
//       const value = data[offset + c]
//       if (value > maxVal) {
//         maxVal = value
//         maxIdx = c
//       }
//     }
//     if (maxIdx != 0 && maxIdx != lastIdx) {
//       const dictIdx = maxIdx - 1
//       if (dictIdx >= 0 && dictIdx < recDict.length) {
//         chars.push(recDict[dictIdx])
//         totalConf += maxVal
//         count += 1
//       }
//     }
//     lastIdx = maxIdx
//   }
//   return { text: chars.join(''), confidence: count > 0 ? totalConf / count : 0 }
// }
interface CtcDecodeResult {
  text: string
  confidence: number
}

function zemoOcrFastConfidence(v: number): number {
  // 如果模型输出已经是概率，直接用
  if (v >= 0 && v <= 1) {
    return v
  }

  // 如果模型输出是 logit，只对最终选中的字符做 sigmoid
  // 不要对全部 classes 做 softmax
  if (v >= 20) {
    return 1
  }

  if (v <= -20) {
    return 0
  }

  return 1 / (1 + Math.exp(-v))
}

/**
 * PP-OCR CTC 快速贪心解码
 *
 * data shape: [steps, classes]
 * class 0: blank
 * class 1..N: dict[0..N-1]
 */
function decodeCtcFast(
  data: Float32Array,
  steps: number,
  classes: number,
  dict: string[]
): CtcDecodeResult {
  const chars: string[] = []

  let lastIndex = -1
  let confSum = 0
  let confCount = 0

  for (let t = 0; t < steps; t++) {
    const base = t * classes

    let maxIndex = 0
    let maxValue = data[base]

    for (let c = 1; c < classes; c++) {
      const v = data[base + c]

      if (v > maxValue) {
        maxValue = v
        maxIndex = c
      }
    }

    // CTC blank
    if (maxIndex == 0) {
      lastIndex = maxIndex
      continue
    }

    // CTC 去重：连续相同字符只保留一次
    if (maxIndex == lastIndex) {
      continue
    }

    const dictIndex = maxIndex - 1

    if (dictIndex < 0 || dictIndex >= dict.length) {
      lastIndex = maxIndex
      continue
    }

    const ch = dict[dictIndex]

    if (ch == null || ch == '') {
      lastIndex = maxIndex
      continue
    }

    chars.push(ch)
    confSum += zemoOcrFastConfidence(maxValue)
    confCount++

    lastIndex = maxIndex
  }

  if (chars.length <= 0) {
    return {
      text: '',
      confidence: 0
    }
  }

  return {
    text: chars.join(''),
    confidence: confCount > 0 ? confSum / confCount : 0
  }
}


function validateRecDictionary(classes: number, dictPath: string): void {
  const expectedDictSize = classes - 1
  if (expectedDictSize <= 0 || recDict.length == expectedDictSize) return
  throw new Error(`rec 字典与模型不匹配：outputClasses=${classes}, dictSize=${recDict.length}, expectedDictSize=${expectedDictSize}, dictPath=${dictPath}, dictBytes=${fileSizeSignature(dictPath)}。PP-OCRv6 small rec 应加载官方 inference.yml。`)
}

// export async function recognizePpOcrV6MindSpore(
//   pixelMap: image.PixelMap,
//   detPath: string,
//   recPath: string,
//   dictPath: string
// ): Promise<string> {
//   try {
//     await ensureLoaded(detPath, recPath, dictPath)
//   } catch (e) {
//     throw new Error(`ensureLoaded failed: ${errorText(e as Object)}`)
//   }
//   console.log(`[ZemoOcr][MindSpore] loaded models, dictSize=${recDict.length}, dictPath=${dictPath}, dictBytes=${fileSizeSignature(dictPath)}`)
//   const source = await readPixelInfo(pixelMap)
//   console.log(`[ZemoOcr][MindSpore] source width=${source.width}, height=${source.height}, pixelFormat=${source.pixelFormat}`)
//   const boxes = await detectTextBoxes(source)
//   console.log(`[ZemoOcr][MindSpore] rec start boxes=${boxes.length}`)
//   const items: PpOcrV6Item[] = []
//   for (let i = 0; i < boxes.length; i++) {
//     const crop = cropPixelInfo(source, expandBoxForRecognition(boxes[i], source))
//     if (crop == null) {
//       console.log(`[ZemoOcr][MindSpore] rec#${i} skipped empty crop, box=${boxText(boxes[i])}`)
//       continue
//     }
//     const recPrep = preprocessRec(crop)
//     const recOutput = await runModel(recModel, recPrep.data, [1, 3, recPrep.height, recPrep.width], `rec#${i}`)
//     const recData = tensorDataToFloats(recOutput)
//     const shape = tensorShape(recOutput)
//     let steps = 0
//     let classes = 0
//     if (shape.length == 3) {
//       steps = shape[1]
//       classes = shape[2]
//     } else if (shape.length == 2) {
//       steps = shape[0]
//       classes = shape[1]
//     }
//     if (steps <= 0 || classes <= 0) continue
//     validateRecDictionary(classes, dictPath)
//     const rec = decodeCtc(recData, steps, classes)
//     console.log(`[ZemoOcr][MindSpore] rec#${i}/${boxes.length} box=${boxText(boxes[i])}, confidence=${rec.confidence}, text=${oneLineText(rec.text)}`)
//     if (rec.text != '') {
//       const item: PpOcrV6Item = {
//         text: rec.text,
//         confidence: rec.confidence,
//         box: boxes[i]
//       }
//       items.push(item)
//     }
//   }
//   items.sort((a: PpOcrV6Item, b: PpOcrV6Item): number => {
//     const ab = a.box
//     const bbx = b.box
//     const dy = Math.abs(ab.top - bbx.top)
//     return dy < 20 ? ab.left - bbx.left : ab.top - bbx.top
//   })
//   const texts: string[] = []
//   for (let i = 0; i < items.length; i++) {
//     const item = items[i]
//     texts.push(item.text)
//   }
//   const payload: PpOcrV6Payload = { items, fullText: texts.join('\n') }
//   return JSON.stringify(payload)
// }
// const REC_CONFIDENCE_THRESHOLD = 0.45

// function isTinyOcrBox(box: OcrBox): boolean {
//   const w = Math.max(0, box.right - box.left)
//   const h = Math.max(0, box.bottom - box.top)

//   if (w < 6) return true
//   if (h < 6) return true
//   if (w * h < 60) return true

//   return false
// }

// export async function recognizePpOcrV6MindSpore(
//   pixelMap: image.PixelMap,
//   detPath: string,
//   recPath: string,
//   dictPath: string
// ): Promise<string> {
//   const totalStart = Date.now()

//   try {
//     await ensureLoaded(detPath, recPath, dictPath)
//   } catch (e) {
//     throw new Error(`ensureLoaded failed: ${errorText(e as Object)}`)
//   }

//   const source = await readPixelInfo(pixelMap)

//   const detStart = Date.now()
  // const boxes = await detectTextBoxes(source)
//   const detCost = Date.now() - detStart

//   const items: PpOcrV6Item[] = []

//   let checkedClasses = -1
//   let skippedTiny = 0
//   let skippedEmptyCrop = 0
//   let skippedInvalidShape = 0
//   let skippedEmptyText = 0
//   let skippedLowConfidence = 0

//   const recStart = Date.now()

//   for (let i = 0; i < boxes.length; i++) {
//     const rawBox = boxes[i]
//     const recBox = expandBoxForRecognition(rawBox, source) as OcrBox

//     if (isTinyOcrBox(recBox)) {
//       skippedTiny++
//       continue
//     }

//     const crop = cropPixelInfo(source, recBox)

//     if (crop == null) {
//       skippedEmptyCrop++
//       continue
//     }

//     const recPrep = preprocessRec(crop)

//     const recOutput = await runModel(
//       recModel,
//       recPrep.data,
//       [3, 3, recPrep.height, recPrep.width],
//       `rec#${i}`
//     )

//     const recData = tensorDataToFloats(recOutput)
//     const shape = tensorShape(recOutput)

//     let steps = 0
//     let classes = 0

//     if (shape.length == 3) {
//       steps = shape[1]
//       classes = shape[2]
//     } else if (shape.length == 2) {
//       steps = shape[0]
//       classes = shape[1]
//     }

//     if (steps <= 0 || classes <= 0) {
//       skippedInvalidShape++
//       continue
//     }

//     if (checkedClasses != classes) {
//       validateRecDictionary(classes, dictPath)
//       checkedClasses = classes
//     }

//     const rec = decodeCtc(recData, steps, classes)

//     if (rec.text == '') {
//       skippedEmptyText++
//       continue
//     }

//     if (rec.confidence < REC_CONFIDENCE_THRESHOLD) {
//       skippedLowConfidence++
//       continue
//     }

//     const item: PpOcrV6Item = {
//       text: rec.text,
//       confidence: rec.confidence,
//       box: rawBox
//     }

//     items.push(item)
//   }

//   const recCost = Date.now() - recStart

//   items.sort((a: PpOcrV6Item, b: PpOcrV6Item): number => {
//     const ab = a.box
//     const bbx = b.box
//     const dy = Math.abs(ab.top - bbx.top)

//     return dy < 20 ? ab.left - bbx.left : ab.top - bbx.top
//   })

//   const texts: string[] = []

//   for (let i = 0; i < items.length; i++) {
//     texts.push(items[i].text)
//   }

//   const payload: PpOcrV6Payload = {
//     items,
//     fullText: texts.join('\n')
//   }

//   const result = JSON.stringify(payload)
//   const totalCost = Date.now() - totalStart

//   console.log(
//     `[ZemoOcr][MindSpore] done, source=${source.width}x${source.height}, rawBoxes=${boxes.length}, items=${items.length}, detCost=${detCost}ms, recCost=${recCost}ms, totalCost=${totalCost}ms, skippedTiny=${skippedTiny}, skippedEmptyCrop=${skippedEmptyCrop}, skippedInvalidShape=${skippedInvalidShape}, skippedEmptyText=${skippedEmptyText}, skippedLowConfidence=${skippedLowConfidence}`
//   )

//   return result
// }

// const REC_MODEL_BATCH = 8
// const REC_MODEL_C = 3
// const REC_MODEL_H = 48
// const REC_MODEL_W = 320
// const REC_CONFIDENCE_THRESHOLD = 0.45

// function zemoOcrIsTinyBox(box: OcrBox): boolean {
//   const w = Math.max(0, box.right - box.left)
//   const h = Math.max(0, box.bottom - box.top)

//   if (w < 6) return true
//   if (h < 6) return true
//   if (w * h < 60) return true

//   return false
// }

// function zemoOcrCopyFloatData(
//   dst: Float32Array,
//   dstOffset: number,
//   src: Float32Array
// ): void {
//   for (let i = 0; i < src.length; i++) {
//     dst[dstOffset + i] = src[i]
//   }
// }

// function zemoOcrSliceBatchOutput(
//   data: Float32Array,
//   batchIndex: number,
//   steps: number,
//   classes: number
// ): Float32Array {
//   const singleLength = steps * classes
//   const offset = batchIndex * singleLength
//   const result = new Float32Array(singleLength)

//   for (let i = 0; i < singleLength; i++) {
//     result[i] = data[offset + i]
//   }

//   return result
// }

// export async function recognizePpOcrV6MindSpore(
//   pixelMap: image.PixelMap,
//   detPath: string,
//   recPath: string,
//   dictPath: string
// ): Promise<string> {
//   const totalStart = Date.now()

//   try {
//     await ensureLoaded(detPath, recPath, dictPath)
//   } catch (e) {
//     throw new Error(`ensureLoaded failed: ${errorText(e as Object)}`)
//   }

//   const source = await readPixelInfo(pixelMap)

//   const detStart = Date.now()
//   const boxes = await detectTextBoxes(source)
//   const detCost = Date.now() - detStart

//   const items: PpOcrV6Item[] = []

//   let checkedClasses = -1

//   let skippedTiny = 0
//   let skippedEmptyCrop = 0
//   let skippedInvalidSize = 0
//   let skippedInvalidShape = 0
//   let skippedEmptyText = 0
//   let skippedLowConfidence = 0

//   const singleInputLength = REC_MODEL_C * REC_MODEL_H * REC_MODEL_W
//   const recStart = Date.now()

//   for (let start = 0; start < boxes.length; start += REC_MODEL_BATCH) {
//     const batchInput = new Float32Array(REC_MODEL_BATCH * singleInputLength)
//     const batchBoxes: OcrBox[] = []

//     let realBatch = 0

//     for (let b = 0; b < REC_MODEL_BATCH; b++) {
//       const index = start + b

//       if (index >= boxes.length) {
//         break
//       }

//       const rawBox = boxes[index] as OcrBox
//       const recBox = expandBoxForRecognition(rawBox, source) as OcrBox

//       if (zemoOcrIsTinyBox(recBox)) {
//         skippedTiny++
//         continue
//       }

//       const crop = cropPixelInfo(source, recBox)

//       if (crop == null) {
//         skippedEmptyCrop++
//         continue
//       }

//       const recPrep = preprocessRec(crop)

//       if (
//         recPrep.height != REC_MODEL_H ||
//         recPrep.width != REC_MODEL_W ||
//         recPrep.data.length != singleInputLength
//       ) {
//         skippedInvalidSize++

//         console.log(
//           `[ZemoOcr][MindSpore] rec batch skip size mismatch, index=${index}, h=${recPrep.height}, w=${recPrep.width}, dataLen=${recPrep.data.length}, expected=${singleInputLength}`
//         )

//         continue
//       }

//       const dstOffset = realBatch * singleInputLength

//       zemoOcrCopyFloatData(
//         batchInput,
//         dstOffset,
//         recPrep.data as Float32Array
//       )

//       batchBoxes.push(rawBox)
//       realBatch++
//     }

//     if (realBatch <= 0) {
//       continue
//     }

//     // 注意：
//     // 你的 rec 模型是固定 batch=3，所以这里永远传 [3,3,48,320]。
//     // 如果最后一组只有 1 或 2 个文字框，batchInput 后面的数据保持 0 padding。
//     const recOutput = await runModel(
//       recModel,
//       batchInput,
//       [REC_MODEL_BATCH, REC_MODEL_C, REC_MODEL_H, REC_MODEL_W],
//       `recBatch#${start}`
//     )

//     const recData = tensorDataToFloats(recOutput) as Float32Array
//     const shape = tensorShape(recOutput)

//     let outBatch = 0
//     let steps = 0
//     let classes = 0

//     if (shape.length == 3) {
//       outBatch = shape[0]
//       steps = shape[1]
//       classes = shape[2]
//     } else {
//       skippedInvalidShape += realBatch

//       console.log(
//         `[ZemoOcr][MindSpore] rec batch invalid output shape=${JSON.stringify(shape)}`
//       )

//       continue
//     }

//     if (outBatch != REC_MODEL_BATCH || steps <= 0 || classes <= 0) {
//       skippedInvalidShape += realBatch

//       console.log(
//         `[ZemoOcr][MindSpore] rec batch invalid output, shape=${JSON.stringify(shape)}, expectedBatch=${REC_MODEL_BATCH}`
//       )

//       continue
//     }

//     if (checkedClasses != classes) {
//       validateRecDictionary(classes, dictPath)
//       checkedClasses = classes
//     }

//     // 只 decode 真实文字框数量，padding 的位置不要 decode。
//     for (let b = 0; b < realBatch; b++) {
//       const oneData = zemoOcrSliceBatchOutput(recData, b, steps, classes)
//       const rec = decodeCtc(oneData, steps, classes)

//       if (rec.text == '') {
//         skippedEmptyText++
//         continue
//       }

//       if (rec.confidence < REC_CONFIDENCE_THRESHOLD) {
//         skippedLowConfidence++
//         continue
//       }

//       const item: PpOcrV6Item = {
//         text: rec.text,
//         confidence: rec.confidence,
//         box: batchBoxes[b]
//       }

//       items.push(item)
//     }
//   }

//   const recCost = Date.now() - recStart

//   items.sort((a: PpOcrV6Item, b: PpOcrV6Item): number => {
//     const ab = a.box
//     const bbx = b.box
//     const dy = Math.abs(ab.top - bbx.top)
//     return dy < 20 ? ab.left - bbx.left : ab.top - bbx.top
//   })

//   const texts: string[] = []

//   for (let i = 0; i < items.length; i++) {
//     texts.push(items[i].text)
//   }

//   const payload: PpOcrV6Payload = {
//     items,
//     fullText: texts.join('\n')
//   }

//   const result = JSON.stringify(payload)
//   const totalCost = Date.now() - totalStart

//   console.log(
//     `[ZemoOcr][MindSpore] batch done, source=${source.width}x${source.height}, rawBoxes=${boxes.length}, items=${items.length}, modelBatch=${REC_MODEL_BATCH}, detCost=${detCost}ms, recCost=${recCost}ms, totalCost=${totalCost}ms, skippedTiny=${skippedTiny}, skippedEmptyCrop=${skippedEmptyCrop}, skippedInvalidSize=${skippedInvalidSize}, skippedInvalidShape=${skippedInvalidShape}, skippedEmptyText=${skippedEmptyText}, skippedLowConfidence=${skippedLowConfidence}`
//   )

//   return result
// }


// const REC_MODEL_BATCH = 8
// const REC_MODEL_C = 3
// const REC_MODEL_H = 48
// const REC_MODEL_W = 320
// const REC_CONFIDENCE_THRESHOLD = 0.45

// function zemoOcrIsTinyBox(box: OcrBox): boolean {
//   const w = Math.max(0, box.right - box.left)
//   const h = Math.max(0, box.bottom - box.top)

//   if (w < 6) return true
//   if (h < 6) return true
//   if (w * h < 60) return true

//   return false
// }

// function zemoNow(): number {
//   return Date.now()
// }

// function zemoCost(start: number): number {
//   return Date.now() - start
// }

// function zemoOcrSortItems(items: PpOcrV6Item[]): void {
//   items.sort((a: PpOcrV6Item, b: PpOcrV6Item): number => {
//     const ab = a.box
//     const bb = b.box

//     const dy = Math.abs(ab.top - bb.top)

//     if (dy < 20) {
//       return ab.left - bb.left
//     }

//     return ab.top - bb.top
//   })
// }

// function zemoOcrJoinTexts(items: PpOcrV6Item[]): string {
//   const texts: string[] = []

//   for (let i = 0; i < items.length; i++) {
//     texts.push(items[i].text)
//   }

//   return texts.join('\n')
// }

// function zemoOcrCheckRecOutputShape(
//   shape: number[],
//   expectedBatch: number
// ): boolean {
//   if (shape.length != 3) {
//     return false
//   }

//   if (shape[0] != expectedBatch) {
//     return false
//   }

//   if (shape[1] <= 0) {
//     return false
//   }

//   if (shape[2] <= 0) {
//     return false
//   }

//   return true
// }

// export async function recognizePpOcrV6MindSpore(
//   pixelMap: image.PixelMap,
//   detPath: string,
//   recPath: string,
//   dictPath: string
// ): Promise<string> {
//   const totalStart = zemoNow()

//   let ensureCost = 0
//   let readPixelCost = 0
//   let detCost = 0
//   let recTotalCost = 0
//   let recCropCost = 0
//   let recPreprocessCost = 0
//   let recCopyCost = 0
//   let recRunCost = 0
//   let recDecodeCost = 0
//   let sortCost = 0
//   let jsonCost = 0

//   try {
//     const ensureStart = zemoNow()
//     await ensureLoaded(detPath, recPath, dictPath)
//     ensureCost = zemoCost(ensureStart)
//   } catch (e) {
//     throw new Error(`ensureLoaded failed: ${errorText(e as Object)}`)
//   }

//   const readStart = zemoNow()
//   const source = await readPixelInfo(pixelMap)
//   readPixelCost = zemoCost(readStart)

//   const detStart = zemoNow()
//   const boxes = await detectTextBoxes(source)
//   detCost = zemoCost(detStart)

//   const items: PpOcrV6Item[] = []

//   let checkedClasses = -1

//   let skippedTiny = 0
//   let skippedEmptyCrop = 0
//   let skippedInvalidSize = 0
//   let skippedInvalidShape = 0
//   let skippedEmptyText = 0
//   let skippedLowConfidence = 0

//   let totalValidForRec = 0
//   let totalRecBatchRuns = 0
//   let totalPaddingSlots = 0

//   const singleInputLength = REC_MODEL_C * REC_MODEL_H * REC_MODEL_W

//   /**
//    * 关键优化：
//    * batchInput 放到循环外复用。
//    * 不要每一批 new Float32Array。
//    */
//   const batchInput = new Float32Array(REC_MODEL_BATCH * singleInputLength)

//   const recStart = zemoNow()

//   let boxIndex = 0

//   /**
//    * 关键优化：
//    * 不再使用：
//    *
//    * for (let start = 0; start < boxes.length; start += REC_MODEL_BATCH)
//    *
//    * 因为中间有 skippedTiny / skippedInvalidSize 等 continue，
//    * 会导致 batch 经常没填满，但是固定 batch 模型仍然会计算 padding。
//    *
//    * 现在改成：
//    * 一直往 batchInput 塞有效框，塞满 REC_MODEL_BATCH 再跑。
//    */
//   while (boxIndex < boxes.length) {
//     batchInput.fill(0)

//     const batchBoxes: OcrBox[] = []

//     let realBatch = 0
//     const batchCollectStartBoxIndex = boxIndex

//     while (boxIndex < boxes.length && realBatch < REC_MODEL_BATCH) {
//       const rawBox = boxes[boxIndex] as OcrBox
//       boxIndex++

//       const recBox = expandBoxForRecognition(rawBox, source) as OcrBox

//       if (zemoOcrIsTinyBox(recBox)) {
//         skippedTiny++
//         continue
//       }

//       const cropStart = zemoNow()
//       const crop = cropPixelInfo(source, recBox)
//       recCropCost += zemoCost(cropStart)

//       if (crop == null) {
//         skippedEmptyCrop++
//         continue
//       }

//       const preprocessStart = zemoNow()
//       const recPrep = preprocessRec(crop)
//       recPreprocessCost += zemoCost(preprocessStart)

//       if (
//         recPrep.height != REC_MODEL_H ||
//         recPrep.width != REC_MODEL_W ||
//         recPrep.data.length != singleInputLength
//       ) {
//         skippedInvalidSize++

//         console.log(
//           `[ZemoOcr][MindSpore] rec batch skip size mismatch, index=${boxIndex - 1}, h=${recPrep.height}, w=${recPrep.width}, dataLen=${recPrep.data.length}, expected=${singleInputLength}`
//         )

//         continue
//       }

//       const dstOffset = realBatch * singleInputLength

//       /**
//        * 关键优化：
//        * 用 Float32Array.set() 替代手写 for 循环复制。
//        */
//       const copyStart = zemoNow()
//       batchInput.set(recPrep.data as Float32Array, dstOffset)
//       recCopyCost += zemoCost(copyStart)

//       batchBoxes.push(rawBox)
//       realBatch++
//       totalValidForRec++
//     }

//     if (realBatch <= 0) {
//       continue
//     }

//     totalRecBatchRuns++

//     const paddingSlots = REC_MODEL_BATCH - realBatch
//     totalPaddingSlots += paddingSlots

//     const runStart = zemoNow()

//     const recOutput = await runModel(
//       recModel,
//       batchInput,
//       [REC_MODEL_BATCH, REC_MODEL_C, REC_MODEL_H, REC_MODEL_W],
//       `recBatch#${batchCollectStartBoxIndex}`
//     )

//     recRunCost += zemoCost(runStart)

//     const recData = tensorDataToFloats(recOutput) as Float32Array
//     const shape = tensorShape(recOutput)

//     if (!zemoOcrCheckRecOutputShape(shape, REC_MODEL_BATCH)) {
//       skippedInvalidShape += realBatch

//       console.log(
//         `[ZemoOcr][MindSpore] rec batch invalid output shape=${JSON.stringify(shape)}, expected=[${REC_MODEL_BATCH},steps,classes], realBatch=${realBatch}`
//       )

//       continue
//     }

//     const steps = shape[1]
//     const classes = shape[2]

//     if (checkedClasses != classes) {
//       validateRecDictionary(classes, dictPath)
//       checkedClasses = classes
//     }

//     const singleOutputLength = steps * classes

//     /**
//      * 只 decode 真实样本。
//      * padding 的位置不要 decode。
//      */
//     for (let b = 0; b < realBatch; b++) {
//       const offset = b * singleOutputLength

//       /**
//        * 关键优化：
//        * subarray 是视图，不复制数据。
//        * 不要再 new Float32Array + for copy。
//        */
//       // const oneData = recData.subarray(offset, offset + singleOutputLength)

//       // const decodeStart = zemoNow()
//       // const rec = decodeCtc(oneData, steps, classes)
//       // recDecodeCost += zemoCost(decodeStart)
	  
// 	  const oneData = recData.subarray(offset, offset + singleOutputLength)
	  
// 	  const decodeStart = zemoNow()
// 	  const rec = decodeCtcFast(oneData, steps, classes, recDict)
// 	  recDecodeCost += zemoCost(decodeStart)

//       if (rec.text == '') {
//         skippedEmptyText++
//         continue
//       }

//       if (rec.confidence < REC_CONFIDENCE_THRESHOLD) {
//         skippedLowConfidence++
//         continue
//       }

//       const item: PpOcrV6Item = {
//         text: rec.text,
//         confidence: rec.confidence,
//         box: batchBoxes[b] as OcrBox
//       }

//       items.push(item)
//     }
//   }

//   recTotalCost = zemoCost(recStart)

//   const sortStart = zemoNow()
//   zemoOcrSortItems(items)
//   sortCost = zemoCost(sortStart)

//   const jsonStart = zemoNow()

//   const payload: PpOcrV6Payload = {
//     items,
//     fullText: zemoOcrJoinTexts(items)
//   }

//   const result = JSON.stringify(payload)
//   jsonCost = zemoCost(jsonStart)

//   const totalCost = zemoCost(totalStart)

//   console.log(
//     `[ZemoOcr][MindSpore] optimized batch done, ` +
//       `source=${source.width}x${source.height}, ` +
//       `rawBoxes=${boxes.length}, ` +
//       `validForRec=${totalValidForRec}, ` +
//       `items=${items.length}, ` +
//       `modelBatch=${REC_MODEL_BATCH}, ` +
//       `batchRuns=${totalRecBatchRuns}, ` +
//       `paddingSlots=${totalPaddingSlots}, ` +
//       `ensureCost=${ensureCost}ms, ` +
//       `readPixelCost=${readPixelCost}ms, ` +
//       `detCost=${detCost}ms, ` +
//       `recTotalCost=${recTotalCost}ms, ` +
//       `recCropCost=${recCropCost}ms, ` +
//       `recPreprocessCost=${recPreprocessCost}ms, ` +
//       `recCopyCost=${recCopyCost}ms, ` +
//       `recRunCost=${recRunCost}ms, ` +
//       `recDecodeCost=${recDecodeCost}ms, ` +
//       `sortCost=${sortCost}ms, ` +
//       `jsonCost=${jsonCost}ms, ` +
//       `totalCost=${totalCost}ms, ` +
//       `skippedTiny=${skippedTiny}, ` +
//       `skippedEmptyCrop=${skippedEmptyCrop}, ` +
//       `skippedInvalidSize=${skippedInvalidSize}, ` +
//       `skippedInvalidShape=${skippedInvalidShape}, ` +
//       `skippedEmptyText=${skippedEmptyText}, ` +
//       `skippedLowConfidence=${skippedLowConfidence}`
//   )

//   return result
// }
const REC_MODEL_BATCH = 8
const REC_MODEL_C = 3
const REC_MODEL_H = 48
const REC_MODEL_W = 320
const REC_CONFIDENCE_THRESHOLD = 0.45

/**
 * 横向行合并参数。
 * 不丢框，只把同一行的 det 框合成横向 rec 框。
 */
const OCR_LINE_VERTICAL_OVERLAP_RATIO = 0.34
const OCR_LINE_CENTER_RATIO = 0.85
const OCR_LINE_MIN_CENTER_GAP = 22
/**
 * 单个横向 chunk 的最大宽度。
 * 太大时会把一行拆成多个 chunk，避免 rec 输入 320x48 后文字被压扁。
 */
const OCR_CHUNK_MAX_WIDTH_RATIO = 0.98
const OCR_CHUNK_MAX_ASPECT_RATIO = 28.0

class ZemoOcrMergeResult {
  boxes: OcrBox[] = []
  rawValidBoxes: number = 0
  lineCount: number = 0
  skippedTiny: number = 0
}

function zemoNow(): number {
  return Date.now()
}

function zemoCost(start: number): number {
  return Date.now() - start
}

function zemoOcrBoxWidth(box: OcrBox): number {
  return Math.max(0, box.right - box.left)
}

function zemoOcrBoxHeight(box: OcrBox): number {
  return Math.max(0, box.bottom - box.top)
}

function zemoOcrCenterY(box: OcrBox): number {
  return (box.top + box.bottom) * 0.5
}

function zemoOcrMakeBox(
  left: number,
  top: number,
  right: number,
  bottom: number
): OcrBox {
  const box: OcrBox = {
    left: left,
    top: top,
    right: right,
    bottom: bottom
  }

  return box
}

function zemoOcrClampBox(box: OcrBox, source: PixelInfo): OcrBox {
  const left = Math.max(0, Math.min(source.width - 1, Math.floor(box.left)))
  const top = Math.max(0, Math.min(source.height - 1, Math.floor(box.top)))
  const right = Math.max(left + 1, Math.min(source.width, Math.ceil(box.right)))
  const bottom = Math.max(top + 1, Math.min(source.height, Math.ceil(box.bottom)))

  return zemoOcrMakeBox(left, top, right, bottom)
}

function zemoOcrUnionBox(a: OcrBox, b: OcrBox, source: PixelInfo): OcrBox {
  const left = Math.min(a.left, b.left)
  const top = Math.min(a.top, b.top)
  const right = Math.max(a.right, b.right)
  const bottom = Math.max(a.bottom, b.bottom)

  return zemoOcrClampBox(
    zemoOcrMakeBox(left, top, right, bottom),
    source
  )
}

function zemoOcrVerticalOverlap(a: OcrBox, b: OcrBox): number {
  const top = Math.max(a.top, b.top)
  const bottom = Math.min(a.bottom, b.bottom)
  return Math.max(0, bottom - top)
}

function zemoOcrIsTinyBox(box: OcrBox): boolean {
  const w = zemoOcrBoxWidth(box)
  const h = zemoOcrBoxHeight(box)

  if (w < 6) return true
  if (h < 6) return true
  if (w * h < 60) return true

  return false
}

function zemoOcrIsSameLine(a: OcrBox, b: OcrBox): boolean {
  const ah = zemoOcrBoxHeight(a)
  const bh = zemoOcrBoxHeight(b)
  const minH = Math.max(1, Math.min(ah, bh))

  const overlap = zemoOcrVerticalOverlap(a, b)
  const overlapRatio = overlap / minH

  if (overlapRatio >= OCR_LINE_VERTICAL_OVERLAP_RATIO) {
    return true
  }

  const centerDiff = Math.abs(zemoOcrCenterY(a) - zemoOcrCenterY(b))
  const centerLimit = Math.max(
    OCR_LINE_MIN_CENTER_GAP,
    minH * OCR_LINE_CENTER_RATIO
  )

  return centerDiff <= centerLimit
}

function zemoOcrSortBoxes(boxes: OcrBox[]): void {
  boxes.sort((a: OcrBox, b: OcrBox): number => {
    const ah = zemoOcrBoxHeight(a)
    const bh = zemoOcrBoxHeight(b)
    const avgH = Math.max(1, (ah + bh) * 0.5)

    const dy = Math.abs(zemoOcrCenterY(a) - zemoOcrCenterY(b))

    if (dy <= Math.max(12, avgH * 0.55)) {
      return a.left - b.left
    }

    return a.top - b.top
  })
}

function zemoOcrSortItems(items: PpOcrV6Item[]): void {
  items.sort((a: PpOcrV6Item, b: PpOcrV6Item): number => {
    const ab = a.box
    const bb = b.box

    const ah = zemoOcrBoxHeight(ab)
    const bh = zemoOcrBoxHeight(bb)
    const avgH = Math.max(1, (ah + bh) * 0.5)

    const dy = Math.abs(zemoOcrCenterY(ab) - zemoOcrCenterY(bb))

    if (dy <= Math.max(16, avgH * 0.6)) {
      return ab.left - bb.left
    }

    return ab.top - bb.top
  })
}

function zemoOcrJoinTexts(items: PpOcrV6Item[]): string {
  const texts: string[] = []

  for (let i = 0; i < items.length; i++) {
    texts.push(items[i].text)
  }

  return texts.join('\n')
}

function zemoOcrCompactLines(items: PpOcrV6Item[]): PpOcrV6Line[] {
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
  return lines
}

function zemoOcrCheckRecOutputShape(
  shape: number[],
  expectedBatch: number
): boolean {
  if (shape.length != 3) {
    return false
  }

  if (shape[0] != expectedBatch) {
    return false
  }

  if (shape[1] <= 0) {
    return false
  }

  if (shape[2] <= 0) {
    return false
  }

  return true
}

/**
 * 把 det 框分成多行。
 */
function zemoOcrSplitToLines(boxes: OcrBox[], source: PixelInfo): OcrBox[][] {
  const sorted = boxes.slice()

  sorted.sort((a: OcrBox, b: OcrBox): number => {
    const dy = zemoOcrCenterY(a) - zemoOcrCenterY(b)

    if (Math.abs(dy) < 6) {
      return a.left - b.left
    }

    return dy
  })

  const lines: OcrBox[][] = []
  const lineBounds: OcrBox[] = []

  for (let i = 0; i < sorted.length; i++) {
    const box = sorted[i]

    let targetLine = -1
    let bestDiff = 1.0e30

    for (let j = 0; j < lineBounds.length; j++) {
      const lineBox = lineBounds[j]

      if (!zemoOcrIsSameLine(lineBox, box)) {
        continue
      }

      const diff = Math.abs(zemoOcrCenterY(lineBox) - zemoOcrCenterY(box))

      if (diff < bestDiff) {
        bestDiff = diff
        targetLine = j
      }
    }

    if (targetLine < 0) {
      const line: OcrBox[] = []
      line.push(box)
      lines.push(line)
      lineBounds.push(box)
    } else {
      lines[targetLine].push(box)
      lineBounds[targetLine] = zemoOcrUnionBox(lineBounds[targetLine], box, source)
    }
  }

  return lines
}

/**
 * 判断当前横向 chunk 加入 next 后会不会太长。
 */
function zemoOcrIsChunkTooLong(chunk: OcrBox, source: PixelInfo): boolean {
  const w = zemoOcrBoxWidth(chunk)
  const h = zemoOcrBoxHeight(chunk)
  const aspect = w / Math.max(1, h)

  if (w > source.width * OCR_CHUNK_MAX_WIDTH_RATIO) {
    return true
  }

  if (aspect > OCR_CHUNK_MAX_ASPECT_RATIO) {
    return true
  }

  return false
}

/**
 * 把一行合成一个或多个横向 chunk。
 * 不丢任何框。
 */
function zemoOcrMergeLineToChunks(line: OcrBox[], source: PixelInfo): OcrBox[] {
  const chunks: OcrBox[] = []

  if (line.length <= 0) {
    return chunks
  }

  line.sort((a: OcrBox, b: OcrBox): number => {
    return a.left - b.left
  })

  let current = line[0]

  for (let i = 1; i < line.length; i++) {
    const next = line[i]
    const merged = zemoOcrUnionBox(current, next, source)

    if (zemoOcrIsChunkTooLong(merged, source)) {
      chunks.push(current)
      current = next
    } else {
      current = merged
    }
  }

  chunks.push(current)

  return chunks
}

/**
 * det 后构建 rec 框。
 * 只做：
 * 1. expand
 * 2. tiny 过滤
 * 3. 按行横向合并
 *
 * 不限制数量，不丢内容。
 */
function zemoOcrBuildRecBoxes(
  rawBoxes: OcrBox[],
  source: PixelInfo
): ZemoOcrMergeResult {
  const result = new ZemoOcrMergeResult()
  const valid: OcrBox[] = []

  let skippedTiny = 0

  for (let i = 0; i < rawBoxes.length; i++) {
    const rawBox = rawBoxes[i]
    const expanded = expandBoxForRecognition(rawBox, source)
    const box = zemoOcrClampBox(expanded, source)

    if (zemoOcrIsTinyBox(box)) {
      skippedTiny++
      continue
    }

    valid.push(box)
  }

  zemoOcrSortBoxes(valid)

  const lines = zemoOcrSplitToLines(valid, source)
  const finalBoxes: OcrBox[] = []

  for (let i = 0; i < lines.length; i++) {
    const chunks = zemoOcrMergeLineToChunks(lines[i], source)

    for (let j = 0; j < chunks.length; j++) {
      finalBoxes.push(chunks[j])
    }
  }

  zemoOcrSortBoxes(finalBoxes)

  result.boxes = finalBoxes
  result.rawValidBoxes = valid.length
  result.lineCount = lines.length
  result.skippedTiny = skippedTiny

  return result
}

export async function recognizePpOcrV6MindSpore(
  pixelMap: image.PixelMap,
  detPath: string,
  recPath: string,
  dictPath: string
): Promise<string> {
  const totalStart = zemoNow()

  let ensureCost = 0
  let readPixelCost = 0
  let detCost = 0
  let mergeCost = 0
  let recTotalCost = 0
  let recCropCost = 0
  let recPreprocessCost = 0
  let recCopyCost = 0
  let recRunCost = 0
  let recDecodeCost = 0
  let sortCost = 0
  let jsonCost = 0

  try {
    const ensureStart = zemoNow()
    await ensureLoaded(detPath, recPath, dictPath)
    ensureCost = zemoCost(ensureStart)
  } catch (e) {
    throw new Error(`ensureLoaded failed: ${errorText(e as Object)}`)
  }

  const readStart = zemoNow()
  const source: PixelInfo = await readPixelInfo(pixelMap)
  readPixelCost = zemoCost(readStart)

  const detStart = zemoNow()
  const rawBoxes: OcrBox[] = await detectTextBoxes(source)
  detCost = zemoCost(detStart)

  const mergeStart = zemoNow()
  const mergeResult = zemoOcrBuildRecBoxes(rawBoxes, source)
  const boxes = mergeResult.boxes
  mergeCost = zemoCost(mergeStart)

  const items: PpOcrV6Item[] = []

  let checkedClasses = -1

  let skippedTiny = mergeResult.skippedTiny
  let skippedEmptyCrop = 0
  let skippedInvalidSize = 0
  let skippedInvalidShape = 0
  let skippedEmptyText = 0
  let skippedLowConfidence = 0

  let totalValidForRec = 0
  let totalRecBatchRuns = 0
  let totalPaddingSlots = 0

  const singleInputLength = REC_MODEL_C * REC_MODEL_H * REC_MODEL_W
  const batchInput = new Float32Array(REC_MODEL_BATCH * singleInputLength)

  const recStart = zemoNow()

  let boxIndex = 0

  while (boxIndex < boxes.length) {
    const batchBoxes: OcrBox[] = []

    let realBatch = 0
    const batchCollectStartBoxIndex = boxIndex

    while (boxIndex < boxes.length && realBatch < REC_MODEL_BATCH) {
      const recBox = boxes[boxIndex]
      boxIndex++

      if (zemoOcrIsTinyBox(recBox)) {
        skippedTiny++
        continue
      }

      const cropStart = zemoNow()
      const crop = cropPixelInfo(source, recBox)
      recCropCost += zemoCost(cropStart)

      if (crop == null) {
        skippedEmptyCrop++
        continue
      }

      const preprocessStart = zemoNow()
      const recPrep = preprocessRec(crop)
      recPreprocessCost += zemoCost(preprocessStart)

      if (
        recPrep.height != REC_MODEL_H ||
        recPrep.width != REC_MODEL_W ||
        recPrep.data.length != singleInputLength
      ) {
        skippedInvalidSize++

        console.log(
          `[ZemoOcr][MindSpore] rec batch skip size mismatch, index=${boxIndex - 1}, ` +
          `h=${recPrep.height}, w=${recPrep.width}, dataLen=${recPrep.data.length}, ` +
          `expected=${singleInputLength}`
        )

        continue
      }

      const dstOffset = realBatch * singleInputLength

      const copyStart = zemoNow()
      batchInput.set(recPrep.data as Float32Array, dstOffset)
      recCopyCost += zemoCost(copyStart)

      batchBoxes.push(recBox)
      realBatch++
      totalValidForRec++
    }

    if (realBatch <= 0) {
      continue
    }

    /**
     * 只有不足 batch 时清空 padding。
     */
    if (realBatch < REC_MODEL_BATCH) {
      batchInput.fill(0, realBatch * singleInputLength)
    }

    totalRecBatchRuns++

    const paddingSlots = REC_MODEL_BATCH - realBatch
    totalPaddingSlots += paddingSlots

    const runStart = zemoNow()

    const recOutput = await runModel(
      recModel,
      batchInput,
      [REC_MODEL_BATCH, REC_MODEL_C, REC_MODEL_H, REC_MODEL_W],
      `recBatch#${batchCollectStartBoxIndex}`
    )

    recRunCost += zemoCost(runStart)

    const recData = tensorDataToFloats(recOutput) as Float32Array
    const shape = tensorShape(recOutput)

    if (!zemoOcrCheckRecOutputShape(shape, REC_MODEL_BATCH)) {
      skippedInvalidShape += realBatch

      console.log(
        `[ZemoOcr][MindSpore] rec batch invalid output shape=${JSON.stringify(shape)}, ` +
        `expected=[${REC_MODEL_BATCH},steps,classes], realBatch=${realBatch}`
      )

      continue
    }

    const steps = shape[1]
    const classes = shape[2]

    if (checkedClasses != classes) {
      validateRecDictionary(classes, dictPath)
      checkedClasses = classes
    }

    const singleOutputLength = steps * classes

    for (let b = 0; b < realBatch; b++) {
      const offset = b * singleOutputLength
      const oneData = recData.subarray(offset, offset + singleOutputLength)

      const decodeStart = zemoNow()
      const rec = decodeCtcFast(oneData, steps, classes, recDict)
      recDecodeCost += zemoCost(decodeStart)

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

  recTotalCost = zemoCost(recStart)

  const sortStart = zemoNow()
  zemoOcrSortItems(items)
  sortCost = zemoCost(sortStart)

  const jsonStart = zemoNow()

  const payload: PpOcrV6Payload = {
    lines: zemoOcrCompactLines(items)
  }

  const result = JSON.stringify(payload)
  jsonCost = zemoCost(jsonStart)

  const totalCost = zemoCost(totalStart)

  console.log(
    `[ZemoOcr][MindSpore] simple line merge batch done, ` +
    `source=${source.width}x${source.height}, ` +
    `rawBoxes=${rawBoxes.length}, ` +
    `rawValidBoxes=${mergeResult.rawValidBoxes}, ` +
    `lineCount=${mergeResult.lineCount}, ` +
    `recBoxes=${boxes.length}, ` +
    `validForRec=${totalValidForRec}, ` +
    `items=${items.length}, ` +
    `modelBatch=${REC_MODEL_BATCH}, ` +
    `batchRuns=${totalRecBatchRuns}, ` +
    `paddingSlots=${totalPaddingSlots}, ` +
    `ensureCost=${ensureCost}ms, ` +
    `readPixelCost=${readPixelCost}ms, ` +
    `detCost=${detCost}ms, ` +
    `mergeCost=${mergeCost}ms, ` +
    `recTotalCost=${recTotalCost}ms, ` +
    `recCropCost=${recCropCost}ms, ` +
    `recPreprocessCost=${recPreprocessCost}ms, ` +
    `recCopyCost=${recCopyCost}ms, ` +
    `recRunCost=${recRunCost}ms, ` +
    `recDecodeCost=${recDecodeCost}ms, ` +
    `sortCost=${sortCost}ms, ` +
    `jsonCost=${jsonCost}ms, ` +
    `totalCost=${totalCost}ms, ` +
    `skippedTiny=${skippedTiny}, ` +
    `skippedEmptyCrop=${skippedEmptyCrop}, ` +
    `skippedInvalidSize=${skippedInvalidSize}, ` +
    `skippedInvalidShape=${skippedInvalidShape}, ` +
    `skippedEmptyText=${skippedEmptyText}, ` +
    `skippedLowConfidence=${skippedLowConfidence}`
  )

  return result
}
