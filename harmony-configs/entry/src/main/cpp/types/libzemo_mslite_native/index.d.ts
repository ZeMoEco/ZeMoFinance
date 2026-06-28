export interface NativeLoadResult {
  ok: boolean
  inputCount: number
  message: string
  diag: string
}

declare const zemoMindSporeNative: {
  loadModel(modelPath: string): string
  predict(inputIds: Int32Array, pixelValues: Float32Array, imageSize: number): Float32Array
  inputCount(): number
  inputDiag(): string
  reset(): void
}

export default zemoMindSporeNative
