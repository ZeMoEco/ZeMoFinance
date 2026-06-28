# zemo-ocs (ZeMo OCR Service)

ZeMo OCR Service Plugin. Used for recognizing text from images.

## Install

Import the plugin in your project.

## Usage

### Import

```typescript
import ZemoOcr from "@/uni_modules/zemo-ocr"
import { OcrResult, OcrItem } from "@/uni_modules/zemo-ocr"
```

### API

#### `recognize(options: RecognizeOptions): void`

Recognizes text from a single image.

```typescript
type RecognizeOptions = {
    imagePath: string
    engine?: 'native' | 'ppocrv6'
    success?: (res: OcrSuccess) => void
    fail?: (err: UniError) => void
    complete?: (res: any) => void
}

type OcrSuccess = {
    items: OcrItem[]
    fullText: string
}

type OcrItem = {
    text: string
    confidence: number
    box: OcrBox
}
```

**Example:**

```typescript
import { recognize } from "@/uni_modules/zemo-ocr"

recognize({
    imagePath: "/storage/emulated/0/DCIM/Camera/test.jpg",
    success: (res) => {
        console.log("Full Text:", res.fullText);
        res.items.forEach(item => {
            console.log(`Found "${item.text}"`);
        });
    },
    fail: (err) => {
        console.error("OCR Failed:", err.errCode, err.errMsg);
    }
});
```

### PP-OCRv6 offline model package

The app setting page can import/update a local PP-OCRv6 package. Android supports ONNX Runtime and Paddle Lite packages. HarmonyOS supports MindSpore Lite `.ms` packages with the official `inference.yml`.

You can import a zip package, an extracted directory, or directly select multiple model files at once.

- Harmony detection model: path/name should contain `det`, for example `det/model.ms`
- Harmony recognition model: path/name should contain `rec`, for example `rec/model.ms`
- ONNX detection model: path/name should contain `det`, for example `det/model.onnx`
- ONNX recognition model: path/name should contain `rec`, for example `rec/model.onnx`
- Paddle Lite detection model: path/name should contain `det`, for example `det/model.nb`
- Paddle Lite recognition model: path/name should contain `rec`, for example `rec/model.nb`
- Android dictionary: optional `dict.txt`, `ppocr_keys*.txt`, `keys*.txt`, `vocab*.txt`, or official `inference.yml` / `inference.yaml` with `PostProcess.character_dict`
- Harmony dictionary: official `inference.yml` / `inference.yaml` with `PostProcess.character_dict`

On Android, if the dictionary is omitted, the bundled `ppocr_keys_v1.txt` is used. On HarmonyOS, `inference.yml` is required and loaded directly. Server inference files such as `inference.pdmodel` and `inference.pdiparams` are not loaded directly.

#### `recognizeTextFromImages(imagePaths: string[]): Promise<any>`

Batched recognition (Compatibility / Legacy API).

- **imagePaths**: Array of absolute paths.
- **Returns**: Promise with combined result.

```typescript
const res = await ZemoOcr.recognizeTextFromImages(["/path/to/img1.jpg", "/path/to/img2.jpg"]);
// Returns object with consolidated text or success status.
```

## Platform Support

- Android: Supported (native plugin, Paddle Lite local OCR; PP-OCRv6 can be selected after importing `.onnx` or `.nb` models)
- iOS: Stub (Pending implementation)
- HarmonyOS: Platform native OCR is supported via CoreVisionKit. PP-OCRv6 can be selected after importing MindSpore Lite `.ms` models.
