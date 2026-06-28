function removedResult(message: string, method: string): string {
  return JSON.stringify({
    ok: false,
    message,
    method
  })
}

export async function classifyFinanceFieldsMindSpore(
  _requestsJson: string,
  _modelPath: string,
  _vocabPath: string,
  _labelsPath: string
): Promise<string> {
  return removedResult('金融字段 MindSpore kit 模型已移除，请使用屏幕视觉模型/规则解析', 'finance_field_ms_removed')
}

export async function classifyFinanceLayoutFieldsMindSpore(
  _requestsJson: string,
  _modelPath: string,
  _vocabPath: string,
  _labelsPath: string,
  _configPath: string
): Promise<string> {
  return removedResult('金融字段 Layout MindSpore kit 模型已移除，请使用屏幕视觉模型/规则解析', 'finance_layout_ms_removed')
}

export async function classifyScreenIntentMindSpore(
  _text: string,
  _modelPath: string,
  _labelsPath: string
): Promise<string> {
  return removedResult('ScreenOmni MindSpore kit 模型已移除，请使用 Native C++ MiniMind-O 全模态模型', 'screen_omni_ms_removed')
}

export async function classifyScreenImageIntentMindSpore(
  _imagePath: string,
  _modelPath: string,
  _labelsPath: string
): Promise<string> {
  return removedResult('ScreenOmni 图片 MindSpore kit 模型已移除，请使用 Native C++ MiniMind-O 全模态模型', 'screen_omni_vision_ms_removed')
}
