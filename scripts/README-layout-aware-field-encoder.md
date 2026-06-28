# Layout-Aware Encoder 制作教程

本教程说明如何从 OCR 数据开始，逐步制作一个面向手机金融页面的 Layout-Aware Encoder，用来识别基金、账单、银行流水、支付详情中的字段。

禁止使用或修改 `unpackage`。以下脚本都在 `scripts` 目录。

## 目标

输入一页 OCR 结果：

```json
{
  "page_type": "investment_holding",
  "width": 1080,
  "height": 2400,
  "items": [
    {"text": "纳斯达克", "bbox": [80, 420, 250, 456], "label": "asset_name"},
    {"text": "513300", "bbox": [820, 420, 960, 456], "label": "asset_code"}
  ]
}
```

输出每个 OCR 框的字段类别：

```text
纳斯达克 -> asset_name
513300 -> asset_code
2752.00 -> market_value
0.204% -> profit_rate
```

## 脚本地图

| 文件 | 作用 |
| --- | --- |
| `prepare-finance-layout-data.py` | 生成或整理 layout 训练数据 |
| `pretrain-layout-aware-base-encoder-scratch.py` | 从零自监督预训练 Layout Base Encoder |
| `train-layout-aware-field-encoder-from-base.py` | 加载基础模型并微调字段分类器 |
| `train-layout-aware-field-encoder-scratch.py` | 不依赖基础模型，从零训练 Layout-Aware Encoder |
| `probe-layout-aware-field-encoder-scratch.py` | 加载 scratch 模型，本地查看预测效果 |
| `export-layout-aware-field-encoder-scratch-onnx.py` | 把 scratch 模型导出为 ONNX |
| `export-layout-aware-field-encoder-scratch-onnx-lite.py` | 导出端侧更稳的固定 shape ONNX |
| `package-layout-field-model-runtime.py` | 打包成 `finance_layout_field_cls.ms/vocab.txt/label_map.json/model_config.json` |
| `train-layout-aware-field-encoder.py` | 可选，使用 RoBERTa 作为文本编码器 |
| `export-chinese-roberta-field-cls-onnx.py` | 旧文本分类器导出脚本 |

## 制作顺序

### 1. 准备训练数据

先用合成数据跑通流程：

```powershell
python scripts\prepare-finance-layout-data.py --samples-per-scene 120 --output E:\CamXAll\ZEMO\Data\model\finance_layout_train.jsonl
```

这个命令会生成以下场景：

- 支付详情
- 支付列表
- 银行账单
- 理财持仓
- 理财交易列表

真实 OCR 文本也可以导入。当前工程 OCR 文本格式是：

```text
[[ocr:left,top,width,height]]文本
```

人工标注时，在行尾加标签：

```text
[[ocr:80,420,180,36]]纳斯达克	label=asset_name
[[ocr:820,420,140,36]]513300	label=asset_code
[[ocr:430,420,130,36]]2752.00	label=market_value
```

转换真实 OCR：

```powershell
python scripts\prepare-finance-layout-data.py --input E:\your_ocr.txt --output E:\CamXAll\ZEMO\Data\model\finance_layout_train.jsonl
```

没有人工标签时可以临时用弱规则：

```powershell
python scripts\prepare-finance-layout-data.py --input E:\your_ocr.txt --weak-label --output E:\CamXAll\ZEMO\Data\model\finance_layout_train.jsonl
```

弱标签只能冷启动，最终必须人工抽样校对。

### 2. 预训练基础模型

推荐先制作基础模型，再做字段分类微调：

```text
无标签/有标签 OCR layout 数据 -> MLM 预训练 base encoder -> 字段分类微调
```

执行：

```powershell
python scripts\pretrain-layout-aware-base-encoder-scratch.py --train-file E:\CamXAll\ZEMO\Data\model\finance_layout_train.jsonl --epochs 30 --batch-size 8
```

默认输出目录：

```text
E:\CamXAll\ZEMO\Data\model\finance_layout_base_encoder_scratch
```

输出文件：

| 文件 | 说明 |
| --- | --- |
| `base_encoder.pt` | 可复用基础 encoder 权重 |
| `pretrain_mlm_model.pt` | base encoder + MLM 头的完整预训练权重 |
| `char_vocab.json` | 字符词表 |
| `model_config.json` | 基础模型结构参数 |

预训练任务是 MLM：随机遮住 OCR 文本里的字符，让模型根据剩余字符和页面布局猜回原字符。

```text
输入：余额 [MASK]4884.70 + bbox/行/列
目标：猜出被遮住的字符
```

这个阶段不需要字段标签，所以真实 OCR 即使没标注也能用于预训练。

### 3. 从基础模型微调字段分类器

```powershell
python scripts\train-layout-aware-field-encoder-from-base.py --base-dir E:\CamXAll\ZEMO\Data\model\finance_layout_base_encoder_scratch --train-file E:\CamXAll\ZEMO\Data\model\finance_layout_train.jsonl --epochs 20 --batch-size 8
```

默认输出目录：

```text
E:\CamXAll\ZEMO\Data\model\finance_layout_field_encoder_from_base
```

这个目录会生成和 scratch 分类模型兼容的文件：

```text
layout_field_encoder_scratch.pt
char_vocab.json
label_map.json
model_config.json
```

因此后面的 probe 和 ONNX 导出脚本可以直接复用。

如果数据很少，可以先冻结基础模型，只训练分类头：

```powershell
python scripts\train-layout-aware-field-encoder-from-base.py --freeze-base
```

如果数据较多，默认不冻结基础模型，效果通常更好。

### 4. 直接从零训练分类模型

不需要 `chinese_roberta_L-4_H-256`：

```powershell
python scripts\train-layout-aware-field-encoder-scratch.py --train-file E:\CamXAll\ZEMO\Data\model\finance_layout_train.jsonl --epochs 20 --batch-size 8
```

默认输出目录：

```text
E:\CamXAll\ZEMO\Data\model\finance_layout_field_encoder_scratch
```

输出文件：

| 文件 | 说明 |
| --- | --- |
| `layout_field_encoder_scratch.pt` | PyTorch 权重 |
| `char_vocab.json` | 字符词表 |
| `label_map.json` | 字段标签表 |
| `model_config.json` | 模型结构参数 |

常用参数：

```powershell
--max-items 96          # 每页最多 OCR 框数量
--max-item-len 32       # 每个 OCR 文本最多字符数
--hidden-size 192       # 向量维度，越大越强，也越慢
--text-layers 2         # 字符级 Transformer 层数
--page-layers 2         # 页面级 Transformer 层数
--heads 4               # attention 头数
```

数据少时先用小模型：

```powershell
python scripts\train-layout-aware-field-encoder-scratch.py --hidden-size 96 --text-layers 1 --page-layers 1 --epochs 10
```

这条路线更简单，但没有预训练基础模型。推荐用于快速验证，不推荐作为最终路线。

### 5. 查看预测效果

```powershell
python scripts\probe-layout-aware-field-encoder-scratch.py --model-dir E:\CamXAll\ZEMO\Data\model\finance_layout_field_encoder_from_base --layout-file E:\CamXAll\ZEMO\Data\model\finance_layout_train.jsonl --page-index 0 --top-k 3
```

输出示例：

```json
{"text":"纳斯达克","gold":"asset_name","pred":"asset_name","top":[{"label":"asset_name","score":0.92}]}
```

如果 `pred` 和 `gold` 经常不一致，优先检查：

- 标签是否标错
- OCR 框坐标是否错位
- 同一字段样本是否太少
- `other` 是否过多
- 合成数据和真实截图差异是否太大

### 6. 导出 ONNX

```powershell
python scripts\export-layout-aware-field-encoder-scratch-onnx.py --model-dir E:\CamXAll\ZEMO\Data\model\finance_layout_field_encoder_from_base --output E:\CamXAll\ZEMO\Data\model\finance_layout_field_encoder_from_base\finance_layout_field_encoder_from_base.onnx
```

默认输出：

```text
E:\CamXAll\ZEMO\Data\model\finance_layout_field_encoder_scratch\finance_layout_field_encoder_scratch.onnx
```

ONNX 输入：

| 输入 | shape | 类型 | 含义 |
| --- | --- | --- | --- |
| `input_ids` | `[1,max_items,max_item_len]` | int64 | 字符 ID |
| `text_mask` | `[1,max_items,max_item_len]` | bool | 字符是否有效 |
| `bbox` | `[1,max_items,4]` | int64 | 归一化坐标，0 到 1000 |
| `row_ids` | `[1,max_items]` | int64 | 行号 |
| `col_ids` | `[1,max_items]` | int64 | 列号 |
| `item_mask` | `[1,max_items]` | bool | OCR 框是否有效 |

ONNX 输出：

| 输出 | shape | 含义 |
| --- | --- | --- |
| `logits` | `[1,max_items,num_labels]` | 每个 OCR 框的分类分数 |

### 7. 转端侧模型

Harmony 端通常需要 MindSpore Lite `.ms`。流程是：

```text
PyTorch .pt -> ONNX -> MindSpore Lite .ms
```

转换命令按本机 MindSpore Lite 工具版本为准，常见形式：

```powershell
converter_lite --fmk=ONNX --modelFile=finance_layout_field_encoder_scratch.onnx --outputFile=finance_layout_field_encoder_scratch
```

如果转换器不支持 bool 输入，需要把 `text_mask`、`item_mask` 改成 int32 或 float32 wrapper 后再导出。这是端侧接入时最常见的兼容点。

如果普通 ONNX 转出的 `.ms` 在 benchmark 中出现 `Shape ... Custom` 加载失败，使用 Lite-friendly 导出脚本：

```powershell
python scripts\export-layout-aware-field-encoder-scratch-onnx-lite.py --model-dir E:\CamXAll\ZEMO\Data\model\finance_layout_field_encoder_from_base --output E:\CamXAll\ZEMO\Data\model\finance_layout_field_encoder_from_base\finance_layout_field_encoder_from_base_lite.onnx
```

再转换：

```powershell
converter_lite --fmk=ONNX --modelFile=finance_layout_field_encoder_from_base_lite.onnx --outputFile=finance_layout_field_encoder_from_base_lite --optimize=general --infer=false
```

### 7.1 新格式运行包

推荐把 layout 模型打包成新格式：

```powershell
python scripts\package-layout-field-model-runtime.py --model-dir E:\CamXAll\ZEMO\Data\model\finance_layout_field_encoder_from_base --output-dir E:\CamXAll\ZEMO\Data\model\finance_layout_field_encoder_runtime
```

输出：

```text
finance_layout_field_cls.ms
vocab.txt
label_map.json
model_config.json
manifest.json
```

`vocab.txt` 格式和旧模型一致：一行一个 token，行号就是 token id。

`model_config.json` 建议保留。它不是模型权重的一部分，但用于端侧预处理参数校验：

```text
max_items = 96
max_item_len = 32
pad_id = 0
unk_id = 1
cls_id = 2
mask_id = 3
bbox_range = 0..1000
inputs = input_ids,text_mask,bbox,row_ids,col_ids,item_mask
```

只有为了兼容旧导入校验时才使用旧命名：

```powershell
python scripts\package-layout-field-model-runtime.py --model-dir E:\CamXAll\ZEMO\Data\model\finance_layout_field_encoder_from_base --output-dir E:\CamXAll\ZEMO\Data\model\finance_layout_field_encoder_runtime_legacy --legacy-names --no-model-config
```

注意：即使文件名叫 `finance_field_cls.ms`，它也不是旧的文本单输入模型。它需要 6 个 layout 输入，不能直接复用旧的 `classifyFinanceFields` 文本推理 runner。

### 8. 端侧接入要做的事

端侧推理必须和 Python 预处理完全一致：

1. 读取 `char_vocab.json`
2. 每个 OCR 文本按字符转 ID
3. 每个 OCR 框坐标归一化到 0 到 1000
4. 按 y、x 排序 OCR 框
5. 聚类生成 `row_ids` 和 `col_ids`
6. 构造 `input_ids/text_mask/bbox/row_ids/col_ids/item_mask`
7. 调用模型得到 `logits`
8. softmax 后取 top-k 标签

如果端侧预处理和训练时不一致，模型会明显掉效果。

## 模型原理

### 字符词表

从训练集 OCR 文本中统计字符：

```text
纳 -> 15
斯 -> 42
达 -> 87
克 -> 91
```

每个 OCR 文本会变成 ID 序列：

```text
[CLS] 纳 斯 达 克 -> [2,15,42,87,91]
```

这一步就是 tokenizer 的最简形式。

### Embedding

ID 本身没有语义，模型先把 ID 映射为向量：

```text
15 -> [0.12, -0.03, 0.88, ...]
```

这就是 embedding。大模型里的 token embedding 也是这个原理。

### 字符级 Transformer

每个 OCR 文本内部做 self-attention：

```text
余额 84884.70
```

模型会学习“余额”和后面数字一起出现时更像 `balance`。

### Layout Embedding

每个 OCR 框有坐标：

```text
x0,y0,x1,y1,row_id,col_id
```

模型把这些也变成向量：

```text
layout_vec = x0_embed + y0_embed + x1_embed + y1_embed + row_embed + col_embed
```

这就是 Layout-Aware 的关键。模型不只看文字，还看它在页面哪里。

### 页面级 Transformer

一页上所有 OCR 框一起进入 Transformer：

```text
名称  市值  盈亏  持仓  代码
纳斯达克 2752.00 5.61 1000 513300
```

模型可以通过 self-attention 学到：

- `纳斯达克` 在名称列，所以是 `asset_name`
- `513300` 在代码列，所以是 `asset_code`
- `0.204%` 带百分号且在盈亏率位置，所以是 `profit_rate`

### 分类头

最后一层是线性分类：

```text
hidden_vec -> logits -> softmax -> label
```

训练时用 CrossEntropyLoss，让正确标签分数变高。

## 从零模型和预训练模型的区别

### scratch 版

优点：

- 不需要任何基础模型
- 结构透明，适合学习
- 体积小，端侧更容易部署

缺点：

- 需要更多真实 OCR 样本
- 泛化能力弱于预训练模型
- 遇到没见过的 App 样式更容易错

### RoBERTa 版

优点：

- 借用已有中文语义能力
- 小数据下更稳
- 对商户名、交易描述更友好

缺点：

- 需要基础模型目录
- 体积更大
- 端侧转换和推理更重

## 真正自制基础模型怎么做

现在有两层模型：

```text
pretrain-layout-aware-base-encoder-scratch.py     -> 基础模型
train-layout-aware-field-encoder-from-base.py     -> 字段分类模型
```

这个基础模型是面向 OCR layout 的小型基础模型，不是通用大语言模型。它已经具备基础 encoder 的制作流程：词表、embedding、self-attention、自监督预训练、下游微调。

如果要进一步做成更通用的基础模型，需要扩大预训练数据和任务：

1. 收集大量无标签 OCR 文本和 layout 页面
2. 建词表
3. 做自监督任务，例如 Masked Language Modeling
4. 训练基础 encoder
5. 再用金融字段标签微调

流程：

```text
大量无标签 OCR -> 预训练 Layout Encoder -> 金融字段有标签数据 -> 微调分类头
```

这就是 BERT/LayoutLM 类模型的制作思路。区别是它们用了海量数据和更大模型。

## 数据建议

第一阶段可以这样做：

| 阶段 | 数据量 | 目标 |
| --- | --- | --- |
| 跑通 | 100 到 500 页合成数据 | 验证训练链路 |
| 可用 | 300 到 1000 页真实 OCR | 覆盖常见页面 |
| 稳定 | 2000 页以上真实 OCR | 覆盖不同 App 和截图样式 |

每页不要只标目标字段，也要保留标题、按钮、tab、说明文字，并标成 `other`。否则模型上线后会把页面控件误判成金额或资产字段。

## 常见问题

### 训练 loss 是 nan

通常是空 OCR 框或 mask 错误。当前脚本已处理 padding 空框。如果仍出现，检查 `bbox`、`items` 是否为空。

### 某些标签一直识别不准

检查该标签样本数。样本太少时，模型会偏向常见标签。

### 合成数据准确，真实截图不准

说明合成页面和真实页面分布差异太大。需要把真实 OCR 加进训练集。

### `other` 太强

`other` 样本过多会压制其它标签。可以减少无关控件，或给少数类补样本。

### ONNX 转 `.ms` 失败

优先检查：

- opset 是否支持
- bool 输入是否支持
- Transformer 算子是否支持
- 是否需要固定 shape

必要时把模型简化：降低层数，降低 hidden size，把 bool mask 改成 int32 或 float32。
