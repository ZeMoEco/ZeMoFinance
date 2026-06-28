# ZeMo-O Real OCR SFT Data

目标：把真实 OCR v2 输出和对应图片整理为 MiniMind3O/MiniMind-O 可读的图文 SFT 数据。

默认输入：

```powershell
E:\CamXAll\ZEMO\Data\Model\Omni\zemo-o\dataset\img\_ocr_out_cpp_v2
```

默认输出：

```powershell
E:\CamXAll\ZEMO\Data\Model\Omni\zemo-o\dataset\sft\zemo_o_real_ocr_sft
```

生成命令：

```powershell
python scripts\export-zemo-o-real-ocr-sft.py
```

输出文件：

- `images/`: 整理后的训练图片副本。
- `zemo_o_real_ocr_sft.jsonl`: 可审计样本，包含 OCR、答案、对话和来源路径。
- `zemo_o_real_ocr_sft.parquet`: MiniMind3O/MiniMind-O 训练 parquet，核心列为 `image_bytes` 和 `conversations`。
- `zemo_o_real_ocr_sft.report.json`: 统计、样例和风险标记。

隐私选项：

```powershell
# 默认：文本脱敏，图片仍保留原始视觉内容
python scripts\export-zemo-o-real-ocr-sft.py --mask-sensitive

# 本机原样训练，不建议外传
python scripts\export-zemo-o-real-ocr-sft.py --no-mask-sensitive

# 图片中敏感 OCR 行也涂黑，需要 pillow
python scripts\export-zemo-o-real-ocr-sft.py --redact-sensitive-images
```

训练接入：

```powershell
Copy-Item `
  E:\CamXAll\ZEMO\Data\Model\Omni\zemo-o\dataset\sft\zemo_o_real_ocr_sft\zemo_o_real_ocr_sft.parquet `
  E:\CamXAll\ZEMO\uniappx\ZeMo-finance\model\minimind-o\dataset\zemo_o_real_ocr_sft.parquet
```

MiniMind-O 训练时使用：

```powershell
cd E:\CamXAll\ZEMO\uniappx\ZeMo-finance\model\minimind-o\trainer
python train_sft_omni.py `
  --data_path ..\dataset\zemo_o_real_ocr_sft.parquet `
  --from_weight sft_omni `
  --save_weight zemo_o_real_ocr_sft `
  --max_seq_len 768 `
  --mode vision_proj `
  --vision_dir ..\model\siglip2-base-p32-256-ve
```

说明：

- `conversations` 的 user 内容包含 `<image>`，MiniMind-O 的 `OmniDataset` 会把它替换为连续的 `<|image_pad|>`。
- parquet 内写入真实 `image_bytes`；训练时如果 `vision_dir` 指向 SigLIP2，模型会走真实图片视觉输入。
- 端侧推理是否真正看图，取决于导出的 `.ms` 图是否有 `pixel_values` 输入，以及运行时是否传入图片像素。text-only 图不会真正看图。
