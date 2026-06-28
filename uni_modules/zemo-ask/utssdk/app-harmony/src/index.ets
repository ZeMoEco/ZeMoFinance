// HarmonyOS implementation for Zemo Ask
import window from '@ohos.window';
import display from '@ohos.display';
import promptAction from '@ohos.promptAction';
import { BuilderNode, floatingBall, FrameNode, NodeController, PiPWindow, screenshot, UIContext } from '@kit.ArkUI';
import { textRecognition } from '@kit.CoreVisionKit';
import { image } from '@kit.ImageKit';
import fs from '@ohos.file.fs';
import { UTSHarmony, uni } from '@dcloudio/uni-app-x-runtime';
import { BusinessError } from '@ohos.base';
import { common, Want } from '@kit.AbilityKit';

const DESK_PET_FLOAT_WINDOW_NAME = "ZemoDeskPetFloatWindow";
const DESK_PET_LEGACY_SUB_WINDOW_NAME = "ZemoDeskPetSubWindow";
const DESK_PET_FLOAT_PAGE_PATH = "uni_modules/zemo-ask/pages/desk-pet-float/index";
const DESK_PET_PIP_MODE_KEY = "desktop_pet_pip_mode";
const SYSTEM_FLOAT_WINDOW_PERMISSION = "ohos.permission.SYSTEM_FLOAT_WINDOW";
const CUSTOM_SCREEN_CAPTURE_PERMISSION = "ohos.permission.CUSTOM_SCREEN_CAPTURE";
const DESK_PET_BILL_CAPTURE_EVENT = "zemo.desk_pet_bill_capture";
const DESK_PET_RECORD_OPEN_EVENT = "zemo.desk_pet_record_open";
const DESK_PET_PENDING_BILL_CAPTURE_KEY = "desktop_pet_pending_bill_capture";
const DESK_PET_PENDING_RECORD_OPEN_KEY = "desktop_pet_pending_record_open";
const DESK_PET_PENDING_RECORD_QUERY_KEY = "desktop_pet_pending_record_query";
const DESK_PET_PENDING_RECORD_AUTO_CONFIRM_KEY = "desktop_pet_pending_record_auto_confirm";
const DESK_PET_CAPTURED_BILL_TEXT_KEY = "desktop_pet_captured_bill_text";
const DESK_PET_CAPTURED_SCREEN_IMAGE_KEY = "desktop_pet_captured_screen_image";
const DESK_PET_BILL_DEBUG_TEXT_KEY = "desktop_pet_bill_debug_text";
const DESK_PET_BILL_DEBUG_STATUS_KEY = "desktop_pet_bill_debug_status";
const DESK_PET_QUICK_ACTIONS_KEY = "desktop_pet_quick_actions";
const DESK_PET_ENABLED_KEY = "desktop_pet_enabled";
const DESK_PET_AUTO_SIDE_KEY = "desktop_pet_auto_side";
const DESK_PET_LOTTIE_NAME_KEY = "desktop_pet_lottie_name";
const DESK_PET_PIP_METRIC_KEY = "desktop_pet_pip_metric";
const DESK_PET_PIP_SNAPSHOT_KEY = "desktop_pet_pip_snapshot";
const FINANCE_RECORD_DEEP_LINK = "zemo://finance/record";
const FINANCE_BUNDLE_NAME = "zenith.most.zemo.finance";
const FINANCE_ENTRY_ABILITY = "EntryAbility";
const DESK_PET_FULL_WINDOW_SIZE = 250;
const DESK_PET_PIP_CONTENT_WIDTH = 220;
const DESK_PET_PIP_CONTENT_HEIGHT = 148;
const PIP_MEMORY_IMAGE_TOKEN = "zemo://pip-memory-image";
const PIP_MINIMIND_IMAGE_SIZE = 256;
const DESK_PET_DOCK_LINE_WIDTH = 24;
const DESK_PET_DOCK_LINE_HEIGHT = 156;
const DESK_PET_BUBBLE_PANEL_WIDTH = 286;
const DESK_PET_BUBBLE_PANEL_HEIGHT = 236;
const DESK_PET_BUBBLE_SIZE = 72;
const DESK_PET_BUBBLE_GAP = 14;
const DESK_PET_BUBBLE_SIDE_GAP = 44;
const MAX_DESK_PET_QUICK_ACTIONS = 4;
const OCR_SCREEN_ENABLED_KEY = "screen_ocr_enabled";
const OCR_PPOCRV6_ENABLED_KEY = "ocr_ppocrv6_enabled";
const OCR_PPOCRV6_DET_PATH_KEY = "ocr_ppocrv6_det_model_path";
const OCR_PPOCRV6_REC_PATH_KEY = "ocr_ppocrv6_rec_model_path";
const OCR_PPOCRV6_DICT_PATH_KEY = "ocr_ppocrv6_dict_path";
const OCR_PPOCRV6_ROOT_DIR = uni.env.USER_DATA_PATH.replace(/\/+$/, "") + "/zemo/assets/ai/ppocrv6";
const OCR_PPOCRV6_DET_PATH = OCR_PPOCRV6_ROOT_DIR + "/det/model.ms";
const OCR_PPOCRV6_REC_PATH = OCR_PPOCRV6_ROOT_DIR + "/rec/model.ms";
const OCR_PPOCRV6_DICT_YAML_PATH = OCR_PPOCRV6_ROOT_DIR + "/inference.yml";

@Builder
function buildDeskPetPipContent() {
    Stack() {
        if (ZemoAsk.isPipCalendarMode()) {
            Column() {
                Row() {
                    Text(ZemoAsk.getPipCalendarTitle())
                        .fontSize(15)
                        .fontWeight(FontWeight.Bold)
                        .fontColor('#FFFFFF')
                        .maxLines(1)
                    Text(ZemoAsk.getPipCalendarTotalText())
                        .fontSize(13)
                        .fontWeight(FontWeight.Bold)
                        .fontColor(ZemoAsk.getPipCalendarTotalColor())
                        .maxLines(1)
                }
                .width('100%')
                .alignItems(VerticalAlign.Center)
                .justifyContent(FlexAlign.SpaceBetween)
                .margin({ bottom: 6 })
                Row() {
                    ForEach(['日', '一', '二', '三', '四', '五', '六'], (label: string) => {
                        Text(label)
                            .fontSize(8)
                            .fontColor('#91DCD7')
                            .textAlign(TextAlign.Center)
                            .layoutWeight(1)
                    })
                }
                .width('100%')
                .margin({ bottom: 3 })
                Column() {
                    ForEach([0, 1, 2, 3, 4, 5], (row: number) => {
                        Row() {
                            ForEach([0, 1, 2, 3, 4, 5, 6], (col: number) => {
                                Column() {
                                    Text(ZemoAsk.getPipCalendarCellDay(row * 7 + col))
                                        .fontSize(7)
                                        .fontColor(ZemoAsk.getPipCalendarCellDayColor(row * 7 + col))
                                        .fontWeight(ZemoAsk.isPipCalendarCellToday(row * 7 + col) ? FontWeight.Bold : FontWeight.Normal)
                                        .maxLines(1)
                                        .textAlign(TextAlign.Center)
                                    Text(ZemoAsk.getPipCalendarCellAmount(row * 7 + col))
                                        .fontSize(7)
                                        .fontWeight(FontWeight.Bold)
                                        .fontColor(ZemoAsk.getPipCalendarCellAmountColor(row * 7 + col))
                                        .maxLines(1)
                                        .textAlign(TextAlign.Center)
                                        .margin({ top: 1 })
                                }
                                .height(17)
                                .layoutWeight(1)
                                .alignItems(HorizontalAlign.Center)
                                .justifyContent(FlexAlign.Center)
                            })
                        }
                        .width('100%')
                        .height(17)
                    })
                }
                .width('100%')
                .layoutWeight(1)
            }
            .width('100%')
            .height('100%')
            .padding(11)
            .borderRadius(24)
            .backgroundColor('#0D5B63')
        } else {
            Column() {
                Row() {
                    Column() {
                        Text(ZemoAsk.getPipAssetMetric().title)
                            .fontSize(18)
                            .fontColor('#DDF7F6')
                            .maxLines(1)
                        Text('泽墨账')
                            .fontSize(24)
                            .fontWeight(FontWeight.Bold)
                            .fontColor('#FFFFFF')
                            .maxLines(1)
                            .margin({ top: 6 })
                    }
                    .layoutWeight(1)
                    .alignItems(HorizontalAlign.Start)
                    Text(ZemoAsk.getPipAssetMetric().badge)
                        .fontSize(14)
                        .fontWeight(FontWeight.Bold)
                        .fontColor('#083F45')
                        .padding({ left: 12, right: 12, top: 5, bottom: 5 })
                        .borderRadius(16)
                        .backgroundColor('#BBF7D0')
                }
                .width('100%')
                .alignItems(VerticalAlign.Center)
                .justifyContent(FlexAlign.SpaceBetween)
                Text(ZemoAsk.getPipAssetMetricAmountText())
                    .fontSize(ZemoAsk.isPipBillScreenshotInProgress() ? 34 : 44)
                    .fontWeight(FontWeight.Bold)
                    .fontColor('#FFFFFF')
                    .maxLines(1)
                    .margin({ top: 14 })
                Row() {
                    Text(ZemoAsk.getPipAssetMetric().desc)
                        .fontSize(15)
                        .fontColor('#B7EBE8')
                        .maxLines(1)
                    Text(ZemoAsk.getPipAssetMetric().tail)
                        .fontSize(14)
                        .fontWeight(FontWeight.Bold)
                        .fontColor('#E8FFF8')
                        .maxLines(1)
                }
                .width('100%')
                .alignItems(VerticalAlign.Center)
                .justifyContent(FlexAlign.SpaceBetween)
                .margin({ top: 12 })
            }
            .width('100%')
            .height('100%')
            .padding(26)
            .borderRadius(30)
            .backgroundColor('#0D5B63')
        }
        if (ZemoAsk.isPipBillConfirmVisible()) {
            Column() {
                Row() {
                    Text(ZemoAsk.getPipBillPreviewAccountSummary())
                        .fontSize(12)
                        .fontWeight(FontWeight.Bold)
                        .fontColor('#DDF7F6')
                        .maxLines(1)
                        .layoutWeight(1)
                    Text(ZemoAsk.getPipBillPreviewModeLabel())
                        .fontSize(12)
                        .fontWeight(FontWeight.Bold)
                        .fontColor('#083F45')
                        .padding({ left: 9, right: 9, top: 3, bottom: 3 })
                        .borderRadius(13)
                        .backgroundColor(ZemoAsk.getPipBillPreviewModeColor())
                        .margin({ left: 8 })
                }
                .width('100%')
                .alignItems(VerticalAlign.Center)
                .justifyContent(FlexAlign.SpaceBetween)
                Text(ZemoAsk.getPipBillPreviewAmount())
                    .fontSize(34)
                    .fontWeight(FontWeight.Bold)
                    .fontColor('#FFFFFF')
                    .maxLines(1)
                    .width('100%')
                    .textAlign(TextAlign.Center)
                    .margin({ top: 4, bottom: 4 })
                Row() {
                    Text(ZemoAsk.getPipBillPreviewNote())
                        .fontSize(13)
                        .fontWeight(FontWeight.Bold)
                        .fontColor('#DDF7F6')
                        .maxLines(1)
                        .layoutWeight(1)
                    Text(ZemoAsk.getPipBillPreviewCategoryLabel())
                        .fontSize(11)
                        .fontColor('#91DCD7')
                        .maxLines(1)
                        .margin({ left: 8 })
                }
                .width('100%')
                .alignItems(VerticalAlign.Center)
                .justifyContent(FlexAlign.SpaceBetween)
                .margin({ top: 11 })
                Text(ZemoAsk.getPipBillPreviewHint())
                    .fontSize(11)
                    .fontColor('#91DCD7')
                    .maxLines(1)
                    .margin({ top: 4 })
            }
            .width('100%')
            .height('100%')
            .padding(14)
            .alignItems(HorizontalAlign.Start)
            .justifyContent(FlexAlign.SpaceBetween)
            .borderRadius(30)
            .backgroundColor('#0A3F45')
        }
    }
    .width('100%')
    .height('100%')
    .backgroundColor('#0D5B63')
}

class ZemoDeskPetPipNodeController extends NodeController {
    private contentNode: BuilderNode<[]> | null = null;

    makeNode(context: UIContext): FrameNode | null {
        this.contentNode = new BuilderNode(context);
        this.contentNode.build(wrapBuilder(buildDeskPetPipContent));
        return this.contentNode.getFrameNode();
    }

    refresh(): void {
        if (this.contentNode != null) {
            this.contentNode.build(wrapBuilder(buildDeskPetPipContent));
        }
    }
}

export class DeskPetQuickAction {
    id: string = "";
    title: string = "";
    link: string = "";

    constructor(id: string = "", title: string = "", link: string = "") {
        this.id = id;
        this.title = title;
        this.link = link;
    }
}

export class DeskPetConfig {
    enabled: boolean = false;
    pipMode: boolean = false;
    autoSide: boolean = true;
    lottieName: string = "lottie.json";
    pipMetric: string = "net_worth";
    quickActionsJson: string = "";

    constructor(enabled: boolean = false, pipMode: boolean = false, autoSide: boolean = true, lottieName: string = "lottie.json", pipMetric: string = "net_worth", quickActionsJson: string = "") {
        this.enabled = enabled;
        this.pipMode = pipMode;
        this.autoSide = autoSide;
        this.lottieName = lottieName;
        this.pipMetric = pipMetric;
        this.quickActionsJson = quickActionsJson;
    }
}

export class DeskPetAssetMetric {
    mode: string = "net_worth";
    title: string = "我的资产";
    amount: number = 0.0;
    badge: string = "CNY";
    desc: string = "净资产总览";
    tail: string = "ASSET";
    holder: string = "ASSET CARD";
    updatedAt: number = 0;

    constructor(mode: string = "net_worth", title: string = "我的资产", amount: number = 0.0, badge: string = "CNY", desc: string = "净资产总览", tail: string = "ASSET", holder: string = "ASSET CARD", updatedAt: number = 0) {
        this.mode = mode;
        this.title = title;
        this.amount = amount;
        this.badge = badge;
        this.desc = desc;
        this.tail = tail;
        this.holder = holder;
        this.updatedAt = updatedAt;
    }
}

class DeskPetQuickActionRect {
    x: number = 0;
    y: number = 0;
    width: number = 0;
    height: number = 0;

    constructor(x: number, y: number, width: number, height: number) {
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;
    }
}

class DeskPetOcrLineBox {
    x: number = 0;
    y: number = 0;
    width: number = 0;
    height: number = 0;

    constructor(x: number, y: number, width: number, height: number) {
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;
    }
}

class DeskPetPpOcrV6Status {
    enabled: boolean = false;
    ready: boolean = false;
    detPath: string = "";
    recPath: string = "";
    dictPath: string = "";
    reason: string = "";
    detExists: boolean = false;
    recExists: boolean = false;
    dictExists: boolean = false;
}

class DeskPetPpOcrV6Box {
    left: number | null = null;
    top: number | null = null;
    right: number | null = null;
    bottom: number | null = null;
}

class DeskPetPpOcrV6Item {
    text: string | null = null;
    confidence: number | null = null;
    box: DeskPetPpOcrV6Box | null = null;
}

class DeskPetPpOcrV6Line {
    t: string | null = null;
    c: number | null = null;
    b: Array<number> | null = null;
}

class DeskPetPpOcrV6Payload {
    lines: Array<DeskPetPpOcrV6Line> | null = null;
    items: Array<DeskPetPpOcrV6Item> | null = null;
    fullText: string | null = null;
}

interface ZemoAskOcrMindSporeModule {
    recognizePpOcrV6MindSpore(pixelMap: image.PixelMap, detPath: string, recPath: string, dictPath: string): Promise<string>;
}

class DeskPetOcrPointLike {
    x: number | null = null;
    y: number | null = null;
}

class DeskPetOcrRectLike {
    left: number | null = null;
    top: number | null = null;
    right: number | null = null;
    bottom: number | null = null;
    x: number | null = null;
    y: number | null = null;
    width: number | null = null;
    height: number | null = null;
    w: number | null = null;
    h: number | null = null;
}

class DeskPetOcrLineLike {
    textRect: DeskPetOcrRectLike | null = null;
    boundingBox: DeskPetOcrRectLike | null = null;
    rect: DeskPetOcrRectLike | null = null;
    box: DeskPetOcrRectLike | null = null;
    cornerPoints: Array<DeskPetOcrPointLike> | null = null;
    points: Array<DeskPetOcrPointLike> | null = null;
    vertexes: Array<DeskPetOcrPointLike> | null = null;
}

class DeskPetBillPreview {
    amount: string = "";
    note: string = "";
    mode: string = "expense";
    category: string = "expense_shopping";
    sourceApp: string = "";
    payMethod: string = "";
    cardTail: string = "";
    accountType: string = "";
    status: string = "识别中...";
}

class DeskPetBillOcrLine {
    raw: string = "";
    text: string = "";
    x: number = 0;
    y: number = 0;
    w: number = 0;
    h: number = 0;
    index: number = 0;
}

export class ZemoAsk {
    public static subWindow: window.Window | null = null;
    public static pipController: PiPWindow.PiPController | null = null;
    public static floatingBallController: floatingBall.FloatingBallController | null = null;
    public static floatingBallClickRegistered: boolean = false;
    public static isShowing: boolean = false;
    public static isInitializing: boolean = false;
    public static usePipMode: boolean = false;
    private static pipXComponentController: XComponentController = new XComponentController();
    private static pipNodeController: ZemoDeskPetPipNodeController = new ZemoDeskPetPipNodeController();
    private static currentX: number = 20; // Logical coordinates (vp)
    private static currentY: number = 350; // Logical coordinates (vp)
    private static winWidth: number = DESK_PET_FULL_WINDOW_SIZE; // Logical width (vp)
    private static winHeight: number = DESK_PET_FULL_WINDOW_SIZE; // Logical height (vp)
    private static dockSide: string = "left";
    private static lastBillCaptureTriggerAt: number = 0;
    private static pipBillConfirmVisible: boolean = false;
    private static pipBillConfirmTimer: number = -1;
    private static pipBillCaptureInFlight: boolean = false;
    private static pipBillScreenshotInProgress: boolean = false;
    private static pipBillPreview: DeskPetBillPreview = new DeskPetBillPreview();
    private static pipMiniMindImageValues: Float32Array | null = null;
    private static ppOcrMindSporeModulePromise: Promise<ZemoAskOcrMindSporeModule> | null = null;

    static init(context: Object): void {
        console.log("ZemoAsk init in HarmonyOS");
    }

    static getIsShowing(): boolean {
        return ZemoAsk.isShowing;
    }

    static getWindowPosition(): number[] {
        return [ZemoAsk.currentX, ZemoAsk.currentY];
    }

    static getWindowSize(): number[] {
        return [ZemoAsk.winWidth, ZemoAsk.winHeight];
    }

    static getDockSide(): string {
        return ZemoAsk.dockSide;
    }

    static isPipBillConfirmVisible(): boolean {
        return ZemoAsk.pipBillConfirmVisible;
    }

    static isPipBillScreenshotInProgress(): boolean {
        return ZemoAsk.pipBillScreenshotInProgress;
    }

    static getPipBillPreviewTitle(): string {
        if (ZemoAsk.pipBillScreenshotInProgress) return "记账中";
        const note = ZemoAsk.pipBillPreview.note.trim();
        if (note != "") return note;
        return ZemoAsk.pipBillPreview.status;
    }

    static getPipBillPreviewModeLabel(): string {
        if (ZemoAsk.pipBillScreenshotInProgress || ZemoAsk.pipBillCaptureInFlight) return "识别中";
        if (ZemoAsk.pipBillPreview.mode == "investment") return "理财";
        if (ZemoAsk.pipBillPreview.mode == "income") return "收入";
        if (ZemoAsk.pipBillPreview.mode == "transfer") return "转账";
        if (ZemoAsk.pipBillPreview.mode == "refund") return "退款";
        return "支出";
    }

    static getPipBillPreviewModeColor(): string {
        if (ZemoAsk.pipBillPreview.mode == "investment") return "#BBF7D0";
        if (ZemoAsk.pipBillPreview.mode == "income") return "#BBF7D0";
        if (ZemoAsk.pipBillPreview.mode == "refund") return "#BBF7D0";
        if (ZemoAsk.pipBillPreview.mode == "transfer") return "#BFDBFE";
        return "#FDE68A";
    }

    static getPipBillPreviewCategoryLabel(): string {
        const category = ZemoAsk.pipBillPreview.category.trim();
        if (category == "") return "未分类";
        if (category == "income_salary") return "工资";
        if (category == "income_refund") return "退款";
        if (category == "income_transfer") return "转入";
        if (category == "expense_food") return "餐饮";
        if (category == "expense_transport") return "交通";
        if (category == "expense_gift") return "红包/礼金";
        if (category == "expense_shopping") return "购物";
        if (category == "transfer") return "转账";
        if (category == "investment_holding") return "持有";
        return category;
    }

    static getPipBillPreviewAmount(): string {
        if (ZemoAsk.pipBillScreenshotInProgress) return "记账中";
        if (ZemoAsk.pipBillPreview.mode == "investment") return ZemoAsk.pipBillPreview.amount.trim() == "" ? "理财录入中" : ZemoAsk.pipBillPreview.amount.trim();
        const amount = ZemoAsk.pipBillPreview.amount.trim();
        if (amount == "") return "请勿遮挡记账金额";
        if (ZemoAsk.pipBillPreview.mode == "transfer") return "¥" + amount;
        return (ZemoAsk.pipBillPreview.mode == "income" || ZemoAsk.pipBillPreview.mode == "refund" ? "+" : "-") + "¥" + amount;
    }

    static getPipBillPreviewNote(): string {
        if (ZemoAsk.pipBillCaptureInFlight) return "正在识别屏幕内容";
        const note = ZemoAsk.pipBillPreview.note.trim();
        if (note != "") return note;
        return ZemoAsk.pipBillPreview.status;
    }

    static getPipBillPreviewHint(): string {
        if (ZemoAsk.pipBillCaptureInFlight) return "正在识别屏幕内容";
        if (ZemoAsk.pipBillPreview.mode == "investment") return "正在打开 ZeMo 导入持仓";
        if (ZemoAsk.pipBillPreview.amount.trim() == "") return "请勿遮挡记账金额";
        return "再点暂停确认，5秒后自动记录";
    }

    static getPipBillPreviewAccountLabel(): string {
        if (ZemoAsk.pipBillCaptureInFlight) return "账户 识别中";
        const pay = ZemoAsk.pipBillPreview.payMethod.trim();
        const app = ZemoAsk.pipBillPreview.sourceApp.trim();
        const accountType = ZemoAsk.pipBillPreview.accountType.trim();
        const base = pay != "" ? pay : (app != "" ? app : "账户待匹配");
        return accountType == "" ? base : base + " · " + accountType;
    }

    static getPipBillPreviewCardLabel(): string {
        const tail = ZemoAsk.pipBillPreview.cardTail.trim();
        if (tail != "") return "尾号 " + tail;
        const type = ZemoAsk.pipBillPreview.accountType.trim();
        return type == "" ? "无尾号" : type;
    }

    static getPipBillPreviewAccountSummary(): string {
        const account = ZemoAsk.getPipBillPreviewAccountLabel();
        const tail = ZemoAsk.pipBillPreview.cardTail.trim();
        if (tail != "") return account + " · " + tail;
        return account;
    }

    static isPipModeEnabled(): boolean {
        const raw = ZemoAsk.readStoredText(DESK_PET_PIP_MODE_KEY);
        return raw == "1";
    }

    static isAutoSideEnabled(): boolean {
        const raw = ZemoAsk.readStoredText(DESK_PET_AUTO_SIDE_KEY);
        return raw == "" || raw == "1";
    }

    static getDeskPetConfig(): DeskPetConfig {
        const enabled = ZemoAsk.readStoredText(DESK_PET_ENABLED_KEY) == "1";
        const pipMode = ZemoAsk.readStoredText(DESK_PET_PIP_MODE_KEY) == "1";
        const autoRaw = ZemoAsk.readStoredText(DESK_PET_AUTO_SIDE_KEY);
        const autoSide = autoRaw == "" ? true : autoRaw == "1";
        const lottieNameRaw = ZemoAsk.readStoredText(DESK_PET_LOTTIE_NAME_KEY);
        const lottieName = lottieNameRaw == "" ? "lottie.json" : lottieNameRaw;
        const pipMetric = ZemoAsk.getPipMetricMode();
        const quickActionsJson = ZemoAsk.readStoredText(DESK_PET_QUICK_ACTIONS_KEY);
        return new DeskPetConfig(enabled, pipMode, autoSide, lottieName, pipMetric, quickActionsJson);
    }

    static setDeskPetConfig(config: DeskPetConfig): void {
        try {
            uni.setStorageSync(DESK_PET_ENABLED_KEY, config.enabled ? "1" : "0");
            uni.setStorageSync(DESK_PET_PIP_MODE_KEY, config.pipMode ? "1" : "0");
            uni.setStorageSync(DESK_PET_AUTO_SIDE_KEY, config.autoSide ? "1" : "0");
            if (config.lottieName != "") uni.setStorageSync(DESK_PET_LOTTIE_NAME_KEY, config.lottieName);
            if (config.pipMetric != "") uni.setStorageSync(DESK_PET_PIP_METRIC_KEY, ZemoAsk.normalizePipMetric(config.pipMetric));
            if (config.quickActionsJson != "") uni.setStorageSync(DESK_PET_QUICK_ACTIONS_KEY, config.quickActionsJson);
        } catch (err) {
            console.warn("Failed to save desk pet config: " + JSON.stringify(err));
        }
    }

    static getPipContentWidth(): number {
        return DESK_PET_PIP_CONTENT_WIDTH;
    }

    static getPipContentHeight(): number {
        return DESK_PET_PIP_CONTENT_HEIGHT;
    }

    static getPipMetricMode(): string {
        return ZemoAsk.normalizePipMetric(ZemoAsk.readStoredText(DESK_PET_PIP_METRIC_KEY));
    }

    static getPipAssetMetricAmountText(): string {
        if (ZemoAsk.pipBillScreenshotInProgress) return "记账中";
        return ZemoAsk.getPipAssetMetric().amount.toFixed(2);
    }

    static getPipAssetMetric(): DeskPetAssetMetric {
        const mode = ZemoAsk.getPipMetricMode();
        const raw = ZemoAsk.readStoredText(DESK_PET_PIP_SNAPSHOT_KEY);
        if (raw != "") {
            try {
                const itemRaw: string = ZemoAsk.readPipMetricJson(raw, mode);
                if (itemRaw != "") {
                    return ZemoAsk.mapPipMetric(mode, itemRaw, raw);
                }
            } catch (err) {
                console.warn("Failed to parse desk pet PiP snapshot: " + JSON.stringify(err));
            }
        }
        return ZemoAsk.defaultPipMetric(mode);
    }

    static isPipCalendarMode(): boolean {
        return ZemoAsk.getPipMetricMode() == "calendar_month";
    }

    static getPipCalendarTitle(): string {
        const item = ZemoAsk.getPipCalendarMetricRaw();
        if (item == "") return "日历金额";
        return ZemoAsk.readJsonText(item, "monthLabel", ZemoAsk.readJsonText(item, "title", "日历金额"));
    }

    static getPipCalendarTotalText(): string {
        if (ZemoAsk.pipBillScreenshotInProgress || ZemoAsk.pipBillCaptureInFlight) return "记账中";
        const amount = ZemoAsk.getPipAssetMetric().amount;
        if (amount == 0) return "0.00";
        return (amount > 0 ? "+" : "") + amount.toFixed(2);
    }

    static getPipCalendarTotalColor(): string {
        const amount = ZemoAsk.getPipAssetMetric().amount;
        if (amount > 0) return "#BBF7D0";
        if (amount < 0) return "#FDE68A";
        return "#B7EBE8";
    }

    static getPipCalendarCellDay(index: number): string {
        const item = ZemoAsk.getPipCalendarMetricRaw();
        if (item == "") return "";
        return ZemoAsk.readJsonText(item, "d" + index.toString(), "");
    }

    static getPipCalendarCellAmount(index: number): string {
        const amount = ZemoAsk.getPipCalendarCellAmountValue(index);
        if (amount == 0) return "";
        const abs = Math.abs(amount);
        let text = "";
        if (abs >= 10000) {
            text = (abs / 10000.0).toFixed(1) + "w";
        } else if (abs >= 1000) {
            text = Math.round(abs).toString();
        } else if (abs >= 100) {
            text = abs.toFixed(0);
        } else {
            text = abs.toFixed(abs >= 10 ? 0 : 1);
        }
        return (amount > 0 ? "+" : "-") + text;
    }

    static getPipCalendarCellAmountColor(index: number): string {
        const amount = ZemoAsk.getPipCalendarCellAmountValue(index);
        if (amount > 0) return "#BBF7D0";
        if (amount < 0) return "#FDE68A";
        return "#3B7C82";
    }

    static getPipCalendarCellDayColor(index: number): string {
        const day = ZemoAsk.getPipCalendarCellDay(index);
        if (day == "") return "#3B7C82";
        if (ZemoAsk.isPipCalendarCellToday(index)) return "#FFFFFF";
        return "#B7EBE8";
    }

    static isPipCalendarCellToday(index: number): boolean {
        const item = ZemoAsk.getPipCalendarMetricRaw();
        if (item == "") return false;
        const today = ZemoAsk.readJsonText(item, "today", "");
        if (today == "") return false;
        const key = ZemoAsk.readJsonText(item, "k" + index.toString(), "");
        return key != "" && key == today;
    }

    private static getPipCalendarCellAmountValue(index: number): number {
        const item = ZemoAsk.getPipCalendarMetricRaw();
        if (item == "") return 0.0;
        return ZemoAsk.readJsonNumber(item, "a" + index.toString(), 0.0);
    }

    private static getPipCalendarMetricRaw(): string {
        const raw = ZemoAsk.readStoredText(DESK_PET_PIP_SNAPSHOT_KEY);
        if (raw == "") return "";
        try {
            return ZemoAsk.readPipMetricJson(raw, "calendar_month");
        } catch (_err) {
            return "";
        }
    }

    static getQuickActions(): DeskPetQuickAction[] {
        const raw = ZemoAsk.readStoredText(DESK_PET_QUICK_ACTIONS_KEY);
        if (raw != "") {
            try {
                const parsed = JSON.parse(raw) as DeskPetQuickAction[];
                const normalized = [] as DeskPetQuickAction[];
                for (let i = 0; i < parsed.length && normalized.length < MAX_DESK_PET_QUICK_ACTIONS; i++) {
                    const action = ZemoAsk.normalizeQuickAction(parsed[i], normalized.length);
                    if (action != null) normalized.push(action);
                }
                if (normalized.length > 0) return normalized;
            } catch (err) {
                console.warn("Failed to parse desk pet quick actions: " + JSON.stringify(err));
            }
        }
        return ZemoAsk.defaultQuickActions();
    }

    static async show(text: string): Promise<void> {
        if (ZemoAsk.isShowing) {
            return;
        }
        if (ZemoAsk.isInitializing) {
            console.log("ZemoAsk: already initializing desk pet window, ignoring nested call.");
            return;
        }
        ZemoAsk.usePipMode = ZemoAsk.isPipModeEnabled();
        if (ZemoAsk.usePipMode) {
            const pipShown = await ZemoAsk.showPip();
            if (pipShown) return;
        }
        await ZemoAsk.showFloat(text);
    }

    private static async showPip(): Promise<boolean> {
        if (!PiPWindow.isPiPEnabled()) {
            ZemoAsk.showPipUnavailableToast();
            return false;
        }
        ZemoAsk.isInitializing = true;
        ZemoAsk.winWidth = DESK_PET_PIP_CONTENT_WIDTH;
        ZemoAsk.winHeight = DESK_PET_PIP_CONTENT_HEIGHT;
        try {
            const config: PiPWindow.PiPConfiguration = {
                context: UTSHarmony.getUIAbilityContext(),
                componentController: ZemoAsk.pipXComponentController,
                templateType: PiPWindow.PiPTemplateType.VIDEO_PLAY,
                contentWidth: DESK_PET_PIP_CONTENT_WIDTH,
                contentHeight: DESK_PET_PIP_CONTENT_HEIGHT,
                controlGroups: new Array<PiPWindow.PiPControlGroup>(),
                customUIController: ZemoAsk.pipNodeController
            };
            const controller = await PiPWindow.create(config);
            ZemoAsk.pipController = controller;
            controller.setAutoStartEnabled(true);
            controller.setPiPControlEnabled(PiPWindow.PiPControlType.VIDEO_PLAY_PAUSE, true);
            controller.on('controlEvent', (control) => {
                if (control.controlType != PiPWindow.PiPControlType.VIDEO_PLAY_PAUSE) return;
                ZemoAsk.handlePipBillCaptureControl();
                try {
                    controller.updatePiPControlStatus(
                        PiPWindow.PiPControlType.VIDEO_PLAY_PAUSE,
                        PiPWindow.PiPControlStatus.PLAY
                    );
                } catch (_statusErr) {}
            });
            controller.on('stateChange', (state: PiPWindow.PiPState, reason: string) => {
                if (state == PiPWindow.PiPState.STARTED) {
                    ZemoAsk.onPipStarted(controller);
                } else if (state == PiPWindow.PiPState.STOPPED) {
                    ZemoAsk.onPipStopped(reason);
                } else if (state == PiPWindow.PiPState.ERROR) {
                    ZemoAsk.onPipBootstrapFailed(reason == '' ? '画中画启动失败' : reason);
                }
            });
            await controller.startPiP();
            return true;
        } catch (exception) {
            const bErr = exception as BusinessError;
            ZemoAsk.isInitializing = false;
            ZemoAsk.usePipMode = false;
            console.warn(`Failed to start desk pet PiP. Code: ${bErr.code}, Message: ${bErr.message}`);
            ZemoAsk.showPipFallbackToast();
            return false;
        }
    }

    static onPipStarted(controller: PiPWindow.PiPController): void {
        ZemoAsk.pipController = controller;
        ZemoAsk.isShowing = true;
        ZemoAsk.isInitializing = false;
        ZemoAsk.hidePipBillConfirm();
        void ZemoAsk.destroyBootstrapWindow();
        console.log("ZemoAsk desk pet PiP started successfully on Harmony");
    }

    static onPipStopped(reason: string = ""): void {
        console.log("ZemoAsk desk pet PiP stopped on Harmony, reason=" + reason);
        ZemoAsk.hidePipBillConfirm();
        ZemoAsk.pipController = null;
        ZemoAsk.isShowing = false;
        ZemoAsk.isInitializing = false;
        void ZemoAsk.destroyBootstrapWindow();
    }

    static onPipBootstrapFailed(message: string): void {
        ZemoAsk.isInitializing = false;
        ZemoAsk.isShowing = false;
        ZemoAsk.pipController = null;
        void ZemoAsk.destroyBootstrapWindow();
        console.error("ZemoAsk desk pet PiP bootstrap failed: " + message);
        try {
            promptAction.showToast({
                message: message == '' ? '画中画桌宠启动失败' : message,
                duration: 3000
            });
        } catch (_toastErr) {}
    }

    private static async destroyBootstrapWindow(): Promise<void> {
        if (ZemoAsk.subWindow != null) {
            try {
                await ZemoAsk.subWindow.destroyWindow();
            } catch (_destroyErr) {}
            ZemoAsk.subWindow = null;
        }
    }

    private static showPipUnavailableToast(): void {
        try {
            promptAction.showToast({
                message: "当前系统未开启或不支持画中画，已切换普通悬浮窗",
                duration: 3000
            });
        } catch (_toastErr) {}
    }

    private static showPipFallbackToast(): void {
        try {
            promptAction.showToast({
                message: "画中画启动失败，已切换普通悬浮窗",
                duration: 3000
            });
        } catch (_toastErr) {}
    }

    private static async showFloat(text: string): Promise<void> {
        if (ZemoAsk.isShowing) {
            return;
        }
        if (ZemoAsk.isInitializing) {
            console.log("ZemoAsk: already initializing desk pet window, ignoring nested call.");
            return;
        }
        ZemoAsk.isInitializing = true;
        
        let retryCount = 0;
        const maxRetries = 5;
        const delayMs = 300;

        while (retryCount < maxRetries) {
            try {
                let win: window.Window | null = null;

                try {
                    win = window.findWindow(DESK_PET_FLOAT_WINDOW_NAME);
                    if (win != null) {
                        console.log("ZemoAsk desk pet: found existing global float window, reusing it.");
                    }
                } catch (findErr) {
                    // Window has not been created yet.
                }

                if (win == null) {
                    console.log(`ZemoAsk desk pet: trying to create global TYPE_FLOAT window (attempt ${retryCount + 1})...`);
                    const permissionReady = await ZemoAsk.requestFloatWindowPermission();
                    if (!permissionReady) {
                        throw new Error("System float window permission was not granted");
                    }
                    const config: window.Configuration = {
                        name: DESK_PET_FLOAT_WINDOW_NAME,
                        windowType: window.WindowType.TYPE_FLOAT,
                        ctx: UTSHarmony.getUIAbilityContext()
                    };
                    win = await window.createWindow(config);
                }

                if (win == null) {
                    throw new Error("Unable to create or obtain ZemoDeskPetFloatWindow");
                }
                
                ZemoAsk.subWindow = win;
                ZemoAsk.winWidth = DESK_PET_FULL_WINDOW_SIZE;
                ZemoAsk.winHeight = DESK_PET_FULL_WINDOW_SIZE;
                
                // Get screen and layout details using getDefaultDisplaySync
                const defaultDisplay = display.getDefaultDisplaySync();
                const density = defaultDisplay.densityPixels;

                // Move it to initial physical position
                const pxX = Math.round(ZemoAsk.currentX * density);
                const pxY = Math.round(ZemoAsk.currentY * density);
                try {
                    await win.moveWindowTo(pxX, pxY);
                } catch (moveErr) {
                    console.warn("Failed to move window: " + JSON.stringify(moveErr));
                }
                
                // Resize it to physical dimensions
                const pxWidth = Math.round(ZemoAsk.winWidth * density);
                const pxHeight = Math.round(ZemoAsk.winHeight * density);
                try {
                    await win.resize(pxWidth, pxHeight);
                } catch (resizeErr) {
                    console.warn("Failed to resize window: " + JSON.stringify(resizeErr));
                }

                // Configure window properties: prevent focus stealing, make touchable
                try {
                    await win.setWindowFocusable(false);
                } catch (focusErr) {
                    console.warn("Failed to set window focusable: " + JSON.stringify(focusErr));
                }

                try {
                    await win.setWindowTouchable(true);
                } catch (touchErr) {
                    console.warn("Failed to set window touchable: " + JSON.stringify(touchErr));
                }

                console.log("ZemoAsk: setUIContent native desk pet page start");
                await win.setUIContent(DESK_PET_FLOAT_PAGE_PATH);

                // Try API 11/12 recommended transparent background, fallback to legacy if needed
                try {
                    if (typeof win.setBackgroundColor === 'function') {
                        await win.setBackgroundColor("#00000000");
                    } else {
                        await win.setWindowBackgroundColor("#00000000");
                    }
                } catch (bgErr) {
                    console.warn("Failed to set window background color: " + JSON.stringify(bgErr));
                }
                
                // Show the window
                await win.showWindow();
                
                ZemoAsk.isShowing = true;
                ZemoAsk.isInitializing = false;
                console.log("ZemoAsk global desk pet float window shown successfully on Harmony");
                return; // Entire setup succeeded, exit!

            } catch (exception) {
                const bErr = exception as BusinessError;
                console.warn(`Attempt ${retryCount + 1} to setup desk pet global float window failed. Code: ${bErr.code}, Message: ${bErr.message}`);
                if (bErr.message == "System float window permission was not granted" || bErr.message.indexOf("permission") >= 0) {
                    ZemoAsk.showFloatPermissionToast();
                    break;
                }
                
                if (bErr.code == 1300002) {
                    // Stage/Window is in inactive/abnormal state, defer and retry the entire sequence
                    await new Promise<void>((resolve) => setTimeout(resolve, delayMs));
                    retryCount++;
                } else {
                    // For other errors, log and also retry or wait
                    console.warn(`Non-1300002 exception caught: ${JSON.stringify(exception)}`);
                    await new Promise<void>((resolve) => setTimeout(resolve, delayMs));
                    retryCount++;
                }
            }
        }

        // If we exhaust all retries, log failure
        ZemoAsk.isInitializing = false;
        console.error('Failed to create desk pet global float window.');
    }

    private static requestFloatWindowPermission(): Promise<boolean> {
        return new Promise<boolean>((resolve) => {
            try {
                UTSHarmony.requestSystemPermission([SYSTEM_FLOAT_WINDOW_PERMISSION],
                    (allRight: boolean, grantedList: string[]) => {
                        resolve(allRight || grantedList.indexOf(SYSTEM_FLOAT_WINDOW_PERMISSION) >= 0);
                    },
                    (_doNotAskAgain: boolean, grantedList: string[]) => {
                        resolve(grantedList.indexOf(SYSTEM_FLOAT_WINDOW_PERMISSION) >= 0);
                    }
                );
            } catch (permissionErr) {
                console.warn("Failed to request system float window permission: " + JSON.stringify(permissionErr));
                resolve(false);
            }
        });
    }

    private static showFloatPermissionToast(): void {
        try {
            promptAction.showToast({
                message: "请在系统设置中开启悬浮窗权限",
                duration: 3000
            });
        } catch (toastErr) {
            console.warn("Failed to show float permission toast: " + JSON.stringify(toastErr));
        }
    }

    static async hide(): Promise<void> {
        if (ZemoAsk.usePipMode || ZemoAsk.pipController != null) {
            await ZemoAsk.hidePip();
            return;
        }
        await ZemoAsk.hideFloat();
    }

    private static async hidePip(): Promise<void> {
        ZemoAsk.isInitializing = false;
        const controller = ZemoAsk.pipController;
        ZemoAsk.pipController = null;
        if (controller != null) {
            try {
                await controller.stopPiP();
            } catch (stopErr) {
                console.warn("Failed to stop desk pet PiP: " + JSON.stringify(stopErr));
            }
        }
        await ZemoAsk.destroyBootstrapWindow();
        ZemoAsk.isShowing = false;
        ZemoAsk.usePipMode = false;
    }

    private static async hideFloat(): Promise<void> {
        const destroySafe = async (winObj: window.Window | null): Promise<void> => {
            if (winObj != null) {
                try {
                    await winObj.destroyWindow();
                } catch (destroyErr) {
                    // Ignore already destroyed state errors
                }
            }
        };

        ZemoAsk.isInitializing = false;

        try {
            const existingWin = window.findWindow(DESK_PET_FLOAT_WINDOW_NAME);
            if (existingWin != null) {
                await destroySafe(existingWin);
                console.log("ZemoAsk global desk pet float window found and hidden/destroyed successfully on Harmony");
            }
        } catch (findErr) {
            // Not found is fine
        }
        try {
            const legacyWin = window.findWindow(DESK_PET_LEGACY_SUB_WINDOW_NAME);
            if (legacyWin != null) {
                await destroySafe(legacyWin);
            }
        } catch (legacyFindErr) {
            // Not found is fine
        }
        const win = ZemoAsk.subWindow;
        if (win != null) {
            await destroySafe(win);
        }
        ZemoAsk.subWindow = null;
        ZemoAsk.isShowing = false;
        ZemoAsk.usePipMode = false;
    }

    static moveWindow(x: number, y: number): void {
        const d = display.getDefaultDisplaySync();
        const density = d.densityPixels;
        
        const lsw = d.width / density;
        const lsh = d.height / density;
        const maxX = Math.max(0, lsw - ZemoAsk.winWidth);
        const maxY = Math.max(0, lsh - ZemoAsk.winHeight);
        
        let targetX = x;
        let targetY = y;
        
        if (targetX < 0) targetX = 0;
        if (targetX > maxX) targetX = maxX;
        if (targetY < 0) targetY = 0;
        if (targetY > maxY) targetY = maxY;
        
        ZemoAsk.currentX = targetX;
        ZemoAsk.currentY = targetY;
        
        if (ZemoAsk.usePipMode || ZemoAsk.pipController != null) {
            return;
        }
        const win = ZemoAsk.subWindow;
        if (win != null) {
            const pxX = Math.round(targetX * density);
            const pxY = Math.round(targetY * density);
            win.moveWindowTo(pxX, pxY).catch((err: BusinessError) => {
                console.error("Failed to move window to px " + pxX + ", " + pxY + ": " + JSON.stringify(err));
            });
        }
    }

    static moveWindowByScreenDelta(startX: number, startY: number, deltaScreenX: number, deltaScreenY: number): void {
        const d = display.getDefaultDisplaySync();
        const density = d.densityPixels;
        ZemoAsk.moveWindow(startX + deltaScreenX / density, startY + deltaScreenY / density);
    }

    static dockAsSideLine(): void {
        void ZemoAsk.resizeToSideLine();
    }

    static expandDockedQuickPanel(): void {
        void ZemoAsk.resizeToQuickPanel();
    }

    static collapseQuickPanelToSideLine(): void {
        void ZemoAsk.resizeToSideLine();
    }

    static resolveQuickActionAtScreenPoint(screenX: number, screenY: number): string {
        const actions = ZemoAsk.getQuickActions();
        if (actions.length == 0) return "";
        const d = display.getDefaultDisplaySync();
        const density = d.densityPixels;
        const localX = screenX / density - ZemoAsk.currentX;
        const localY = screenY / density - ZemoAsk.currentY;
        const count = Math.min(actions.length, MAX_DESK_PET_QUICK_ACTIONS);
        for (let i = 0; i < count; i++) {
            const rect = ZemoAsk.getQuickActionRect(i, count);
            if (localX >= rect.x && localX <= rect.x + rect.width && localY >= rect.y && localY <= rect.y + rect.height) {
                return actions[i].id;
            }
        }
        return "";
    }

    static triggerQuickAction(actionId: string): void {
        const actions = ZemoAsk.getQuickActions();
        for (let i = 0; i < actions.length; i++) {
            const action = actions[i];
            if (action.id == actionId) {
                ZemoAsk.executeQuickAction(action);
                return;
            }
        }
        try {
            promptAction.showToast({ message: "未找到快捷功能", duration: 1800 });
        } catch (_toastErr) {}
    }

    static handlePipBillCaptureControl(): void {
        const now = Date.now();
        if (now - ZemoAsk.lastBillCaptureTriggerAt < 800) return;
        ZemoAsk.lastBillCaptureTriggerAt = now;
        if (!ZemoAsk.pipBillConfirmVisible) {
            void ZemoAsk.startPipBillRecognitionPreview();
            return;
        }
        if (ZemoAsk.pipBillPreview.mode == "investment") {
            void ZemoAsk.startPipBillRecognitionPreview();
            return;
        }
        if (ZemoAsk.pipBillPreview.amount.trim() == "") {
            void ZemoAsk.startPipBillRecognitionPreview();
            return;
        }
        void ZemoAsk.confirmPipBillAutoRecord();
    }

    private static async startPipBillRecognitionPreview(): Promise<void> {
        if (ZemoAsk.pipBillCaptureInFlight) return;
        ZemoAsk.pipBillCaptureInFlight = true;
        ZemoAsk.pipBillConfirmVisible = false;
        ZemoAsk.pipBillPreview = new DeskPetBillPreview();
        ZemoAsk.refreshPipContent();
        if (ZemoAsk.pipBillConfirmTimer != -1) {
            clearTimeout(ZemoAsk.pipBillConfirmTimer as number);
            ZemoAsk.pipBillConfirmTimer = -1;
        }

        const captured = await ZemoAsk.captureCurrentScreenBillText();
        if (!captured) {
            ZemoAsk.pipBillPreview = new DeskPetBillPreview();
            ZemoAsk.pipBillPreview.status = "截图失败，请重试";
            ZemoAsk.pipBillPreview.note = "未获取到屏幕截图";
            ZemoAsk.pipBillCaptureInFlight = false;
            ZemoAsk.pipBillConfirmVisible = true;
            ZemoAsk.refreshPipContent();
            return;
        }
        const text = ZemoAsk.readStoredText(DESK_PET_CAPTURED_BILL_TEXT_KEY);
        if (ZemoAsk.looksLikeFundHoldingListText(text)) {
            ZemoAsk.pipBillPreview = new DeskPetBillPreview();
            ZemoAsk.pipBillPreview.mode = "investment";
            ZemoAsk.pipBillPreview.category = "investment_holding";
            ZemoAsk.pipBillPreview.amount = "理财录入中";
            ZemoAsk.pipBillPreview.status = "理财录入中";
            ZemoAsk.pipBillPreview.note = "理财持仓";
            ZemoAsk.pipBillPreview.sourceApp = ZemoAsk.detectFundHoldingSourceApp(text);
            ZemoAsk.pipBillPreview.accountType = "基金/证券持仓";
            ZemoAsk.pipBillCaptureInFlight = false;
            ZemoAsk.pipBillConfirmVisible = true;
            ZemoAsk.markPendingDeskPetBillCapture(false);
            ZemoAsk.emitDeskPetBillCapture();
            ZemoAsk.refreshPipContent();
            return;
        }
        ZemoAsk.pipBillPreview = ZemoAsk.parseBillPreview(text);
        ZemoAsk.pipBillCaptureInFlight = false;
        ZemoAsk.pipBillConfirmVisible = true;
        ZemoAsk.markPendingDeskPetBillCapture(false);
        ZemoAsk.emitDeskPetBillCapture();
        ZemoAsk.refreshPipContent();
    }

    private static hidePipBillConfirm(): void {
        if (ZemoAsk.pipBillConfirmTimer != -1) {
            clearTimeout(ZemoAsk.pipBillConfirmTimer as number);
            ZemoAsk.pipBillConfirmTimer = -1;
        }
        if (!ZemoAsk.pipBillConfirmVisible) return;
        ZemoAsk.pipBillConfirmVisible = false;
        ZemoAsk.pipBillCaptureInFlight = false;
        ZemoAsk.refreshPipContent();
    }

    private static async confirmPipBillAutoRecord(): Promise<void> {
        if (!ZemoAsk.pipBillConfirmVisible) return;
        ZemoAsk.hidePipBillConfirm();
        try {
            uni.removeStorageSync(DESK_PET_PENDING_RECORD_OPEN_KEY);
            uni.setStorageSync(DESK_PET_PENDING_RECORD_AUTO_CONFIRM_KEY, Date.now().toString());
        } catch (err) {
            console.warn("Failed to store pending auto confirm: " + JSON.stringify(err));
        }
        ZemoAsk.emitDeskPetRecordOpen();
        ZemoAsk.refreshPipContent();
    }

    static refreshPipContent(): void {
        try {
            ZemoAsk.pipNodeController.refresh();
        } catch (err) {
            console.warn("Failed to refresh desk pet PiP content: " + JSON.stringify(err));
        }
    }

    static triggerDeskPetBillCapture(): void {
        void ZemoAsk.captureBillTextThenOpenFinanceApp();
    }

    private static async captureBillTextThenOpenFinanceApp(): Promise<void> {
        const captured = await ZemoAsk.captureCurrentScreenBillText();
        if (!captured) {
            ZemoAsk.refreshPipContent();
            return;
        }
        ZemoAsk.markPendingDeskPetBillCapture(false);
        ZemoAsk.emitDeskPetBillCapture();
    }

    private static async captureCurrentScreenBillText(): Promise<boolean> {
        try {
            uni.removeStorageSync(DESK_PET_CAPTURED_BILL_TEXT_KEY);
            uni.removeStorageSync(DESK_PET_CAPTURED_SCREEN_IMAGE_KEY);
            uni.removeStorageSync(DESK_PET_BILL_DEBUG_TEXT_KEY);
            uni.removeStorageSync(DESK_PET_BILL_DEBUG_STATUS_KEY);
        } catch (_clearErr) {}

        const screenOcrEnabled = ZemoAsk.isScreenOcrEnabled();
        if (!screenOcrEnabled) {
            console.log("[截图] 屏幕 OCR 已关闭，截图只进入多模态模型");
            ZemoAsk.storeBillDebugStatus("screen-ocr-disabled-vision-only");
        }

        let pixelMap: image.PixelMap | null = null;
        try {
            const permissionReady = await ZemoAsk.requestScreenCapturePermission();
            console.log("[截图] 权限=" + (permissionReady ? "已授权" : "未授权"));
            if (!permissionReady) {
                console.warn("[截图] 未授权，跳过系统截图：" + CUSTOM_SCREEN_CAPTURE_PERMISSION);
                ZemoAsk.storeBillDebugStatus("screen-capture-permission-denied");
                return false;
            }
            ZemoAsk.storeBillDebugStatus("permissionReady=" + permissionReady);
            console.log("[截图] 调用系统截图");
            pixelMap = await ZemoAsk.takeScreenshotPixelMap();
            if (pixelMap == null) {
                console.warn("[截图] 系统截图返回空 PixelMap");
                ZemoAsk.storeBillDebugStatus("screenshot-empty");
                return false;
            }
            console.log("[截图] 截图成功：像素字节=" + pixelMap.getPixelBytesNumber());
            ZemoAsk.storeBillDebugStatus("screenshot-ok pixelBytes=" + pixelMap.getPixelBytesNumber());
            const currentPixelMap: image.PixelMap = pixelMap;
            const imageToken: string = await ZemoAsk.registerPipMiniMindImage(currentPixelMap);
            if (imageToken != "") {
                uni.setStorageSync(DESK_PET_CAPTURED_SCREEN_IMAGE_KEY, imageToken);
                console.log("[截图] 已交给多模态模型：内存图片");
                ZemoAsk.storeBillDebugStatus("memory-image-ready " + imageToken);
            }
            if (!screenOcrEnabled) {
                if (imageToken == "") {
                    ZemoAsk.storeBillDebugStatus("memory-image-empty");
                    return false;
                }
                return true;
            }
            ZemoAsk.storeBillDebugStatus("ocr-start");
            const text = await ZemoAsk.recognizeBillTextFromPixelMap(currentPixelMap);
            ZemoAsk.storeBillDebugStatus("ocr-ok textLength=" + text.length);
            try {
                uni.setStorageSync(DESK_PET_BILL_DEBUG_TEXT_KEY, text);
            } catch (debugStoreErr) {
                console.warn("[OCR] 写入调试文本失败：" + ZemoAsk.safeStringify(debugStoreErr));
                ZemoAsk.storeBillDebugStatus("debug-store-failed " + ZemoAsk.safeStringify(debugStoreErr));
            }
            if (text.trim() != "") {
                uni.setStorageSync(DESK_PET_CAPTURED_BILL_TEXT_KEY, text);
                return true;
            } else {
                ZemoAsk.storeBillDebugStatus("ocr-empty-text");
                return false;
            }
        } catch (err) {
            const message = ZemoAsk.safeStringify(err);
            console.warn("[截图] 截图或 OCR 失败：" + message);
            ZemoAsk.storeBillDebugStatus("capture-or-ocr-failed " + message);
            return false;
        } finally {
            if (pixelMap != null) {
                try { pixelMap.release(); } catch (_releaseErr) {}
            }
        }
    }

    static consumePipMiniMindImageValues(): Float32Array | null {
        const values = ZemoAsk.pipMiniMindImageValues;
        ZemoAsk.pipMiniMindImageValues = null;
        return values;
    }

    private static async registerPipMiniMindImage(pixelMap: image.PixelMap): Promise<string> {
        const values = await ZemoAsk.buildMiniMindImageValues(pixelMap, PIP_MINIMIND_IMAGE_SIZE);
        if (values.length == 0) return "";
        ZemoAsk.pipMiniMindImageValues = values;
        console.log("[MiniMind-O] 已准备内存截图张量：长度=" + values.length);
        return PIP_MEMORY_IMAGE_TOKEN;
    }

    private static async buildMiniMindImageValues(pixelMap: image.PixelMap, targetSize: number): Promise<Float32Array> {
        const info = await pixelMap.getImageInfo();
        const width = info.size.width;
        const height = info.size.height;
        if (width <= 0 || height <= 0) return new Float32Array(0);
        const buffer = new ArrayBuffer(pixelMap.getPixelBytesNumber());
        await pixelMap.readPixelsToBuffer(buffer);
        const pixels = new Uint8Array(buffer);
        if (pixels.length == 0) return new Float32Array(0);
        let bytesPerPixel = 4;
        const guessed = Math.floor(pixels.length / (width * height));
        if (guessed > 0) bytesPerPixel = guessed;
        const out = new Float32Array(3 * targetSize * targetSize);
        for (let y = 0; y < targetSize; y++) {
            const sy = Math.min(height - 1, Math.floor((y + 0.5) * height / targetSize));
            for (let x = 0; x < targetSize; x++) {
                const sx = Math.min(width - 1, Math.floor((x + 0.5) * width / targetSize));
                const off = (sy * width + sx) * bytesPerPixel;
                const idx = y * targetSize + x;
                const base = idx * 3;
                const r = ZemoAsk.pixelR(pixels, off, bytesPerPixel, info.pixelFormat);
                const g = ZemoAsk.pixelG(pixels, off, bytesPerPixel);
                const b = ZemoAsk.pixelB(pixels, off, bytesPerPixel, info.pixelFormat);
                out[base] = r / 127.5 - 1.0;
                out[base + 1] = g / 127.5 - 1.0;
                out[base + 2] = b / 127.5 - 1.0;
            }
        }
        return out;
    }

    private static pixelR(pixels: Uint8Array, off: number, bytesPerPixel: number, pixelFormat: image.PixelMapFormat): number {
        if (off < 0 || off >= pixels.length) return 0;
        if (bytesPerPixel >= 4 && pixelFormat == image.PixelMapFormat.RGBA_8888) return pixels[Math.min(pixels.length - 1, off + 2)];
        return pixels[off];
    }

    private static pixelG(pixels: Uint8Array, off: number, bytesPerPixel: number): number {
        if (off < 0 || off >= pixels.length) return 0;
        if (bytesPerPixel >= 2) return pixels[Math.min(pixels.length - 1, off + 1)];
        return pixels[off];
    }

    private static pixelB(pixels: Uint8Array, off: number, bytesPerPixel: number, pixelFormat: image.PixelMapFormat): number {
        if (off < 0 || off >= pixels.length) return 0;
        if (bytesPerPixel >= 4 && pixelFormat == image.PixelMapFormat.RGBA_8888) return pixels[off];
        if (bytesPerPixel >= 3) return pixels[Math.min(pixels.length - 1, off + 2)];
        return pixels[off];
    }

    private static async recognizeBillTextFromPixelMap(pixelMap: image.PixelMap): Promise<string> {
        const ppocrStatus = ZemoAsk.resolvePpOcrV6Status();
        if (ppocrStatus.enabled && ppocrStatus.ready) {
            try {
                const ppocr = await ZemoAsk.getPpOcrMindSporeModule();
                console.log("[OCR] 使用 PP-OCRv6：det=" + ppocrStatus.detPath + ", rec=" + ppocrStatus.recPath);
                const payload = await ppocr.recognizePpOcrV6MindSpore(pixelMap, ppocrStatus.detPath, ppocrStatus.recPath, ppocrStatus.dictPath);
                const text = ZemoAsk.normalizePpOcrV6Payload(payload);
                console.log("[OCR] PP-OCRv6 完成：长度=" + text.length + "，预览=" + ZemoAsk.formatDebugPreview(text));
                return text;
            } catch (err) {
                const firstError = ZemoAsk.safeStringify(err);
                console.warn("[OCR] PP-OCRv6 失败：" + firstError);
                const dynamicShapeError = firstError.indexOf("动态 shape") >= 0 || firstError.indexOf("ArkTS MindSpore Lite 不支持动态 shape") >= 0;
                const modelContractError = firstError.indexOf("字典与模型不匹配") >= 0 || firstError.indexOf("outputClasses=") >= 0 || firstError.indexOf("dictSize=") >= 0;
                if (!dynamicShapeError && !modelContractError && ppocrStatus.detPath != ppocrStatus.recPath) {
                    try {
                        const ppocr = await ZemoAsk.getPpOcrMindSporeModule();
                        console.log("[OCR] 交换 det/rec 后重试 PP-OCRv6");
                        const swappedPayload = await ppocr.recognizePpOcrV6MindSpore(pixelMap, ppocrStatus.recPath, ppocrStatus.detPath, ppocrStatus.dictPath);
                        const swappedText = ZemoAsk.normalizePpOcrV6Payload(swappedPayload);
                        console.log("[OCR] PP-OCRv6 重试完成：长度=" + swappedText.length + "，预览=" + ZemoAsk.formatDebugPreview(swappedText));
                        if (swappedText.trim() != "") {
                            try {
                                uni.setStorageSync(OCR_PPOCRV6_DET_PATH_KEY, ppocrStatus.recPath);
                                uni.setStorageSync(OCR_PPOCRV6_REC_PATH_KEY, ppocrStatus.detPath);
                            } catch (_storeSwapErr) {}
                            return swappedText;
                        }
                    } catch (swapErr) {
                        console.warn("[OCR] PP-OCRv6 重试失败：" + ZemoAsk.safeStringify(swapErr));
                    }
                }
                console.warn("[OCR] 改用系统 OCR");
                return await ZemoAsk.recognizeBillTextWithCoreVision(pixelMap, "ppocrv6-runtime-failed " + firstError);
            }
        }

        const reason = ppocrStatus.enabled
            ? ("ppocrv6-model-not-ready detExists=" + ppocrStatus.detExists + ", recExists=" + ppocrStatus.recExists + ", dictExists=" + ppocrStatus.dictExists
                + ", detPath=" + ppocrStatus.detPath + ", recPath=" + ppocrStatus.recPath + ", dictPath=" + ppocrStatus.dictPath)
            : "switch-disabled";
        return await ZemoAsk.recognizeBillTextWithCoreVision(pixelMap, reason);
    }

    private static async getPpOcrMindSporeModule(): Promise<ZemoAskOcrMindSporeModule> {
        if (ZemoAsk.ppOcrMindSporeModulePromise == null) {
            ZemoAsk.ppOcrMindSporeModulePromise = new Promise<ZemoAskOcrMindSporeModule>((resolve, reject) => {
                import('./ZemoAskOcrMindSpore.ets').then((module: Object): void => {
                    resolve(module as ZemoAskOcrMindSporeModule);
                }).catch((err: Error): void => {
                    reject(err);
                });
            });
        }
        return ZemoAsk.ppOcrMindSporeModulePromise as Promise<ZemoAskOcrMindSporeModule>;
    }

    private static async recognizeBillTextWithCoreVision(pixelMap: image.PixelMap, reason: string): Promise<string> {
        const visionInfo: textRecognition.VisionInfo = { pixelMap: pixelMap };
        const configuration: textRecognition.TextRecognitionConfiguration = {
            isDirectionDetectionSupported: true
        };
        console.log("[OCR] 使用系统 OCR：原因=" + reason);
        const result = await textRecognition.recognizeText(visionInfo, configuration);
        const lineText = ZemoAsk.collectTextRecognitionLines(result);
        const blockCount = result != null && result.blocks != null ? result.blocks.length : 0;
        const text = ZemoAsk.normalizeTextRecognitionResult(result);
        console.log("[OCR] 系统 OCR 完成：块=" + blockCount + "，原文长度=" + lineText.length + "，过滤后长度=" + text.length + "，预览=" + ZemoAsk.formatDebugPreview(lineText));
        return text;
    }

    private static resolvePpOcrV6Status(): DeskPetPpOcrV6Status {
        const status = new DeskPetPpOcrV6Status();
        status.detPath = ZemoAsk.readStoredPath(OCR_PPOCRV6_DET_PATH_KEY, OCR_PPOCRV6_DET_PATH);
        status.recPath = ZemoAsk.readStoredPath(OCR_PPOCRV6_REC_PATH_KEY, OCR_PPOCRV6_REC_PATH);
        status.dictPath = ZemoAsk.readStoredPath(OCR_PPOCRV6_DICT_PATH_KEY, OCR_PPOCRV6_DICT_YAML_PATH);
        try {
            const raw = uni.getStorageSync(OCR_PPOCRV6_ENABLED_KEY);
            status.enabled = raw != null && raw.toString() == "1";
        } catch (_err) {
            status.enabled = false;
        }
        status.detExists = ZemoAsk.fileExists(status.detPath);
        status.recExists = ZemoAsk.fileExists(status.recPath);
        status.dictExists = ZemoAsk.fileExists(status.dictPath);
        status.ready = status.enabled && status.detExists && status.recExists && status.dictExists;
        if (!status.enabled) {
            status.reason = "switch-disabled";
        } else if (!status.ready) {
            status.reason = "model-not-ready";
        }
        return status;
    }

    private static isScreenOcrEnabled(): boolean {
        try {
            const raw = uni.getStorageSync(OCR_SCREEN_ENABLED_KEY);
            if (raw == null || raw.toString() == "") return true;
            return raw.toString() == "1";
        } catch (_err) {
            return true;
        }
    }

    private static fileExists(path: string): boolean {
        try {
            let value = (path ?? "").toString().replace(/\\/g, "/");
            if (value.startsWith("file://")) value = value.substring(7);
            const stat = fs.statSync(value);
            return stat != null && !stat.isDirectory();
        } catch (_err) {}
        return false;
    }

    private static readStoredPath(key: string, fallback: string): string {
        try {
            const value = uni.getStorageSync(key);
            const text = value != null ? value.toString().trim() : "";
            if (text != "") return text;
        } catch (_err) {}
        return fallback;
    }

    private static normalizePpOcrV6Payload(payload: string): string {
        try {
            const parsed = JSON.parse(payload) as DeskPetPpOcrV6Payload;
            const lines = new Array<string>();
            const compactLines = parsed.lines;
            if (compactLines != null) {
                for (let i = 0; i < compactLines.length; i++) {
                    const raw = compactLines[i];
                    const cleaned = ZemoAsk.cleanPpOcrV6ItemLine(raw.t != null ? raw.t : "");
                    if (cleaned == "") continue;
                    const box = raw.b;
                    if (box != null && box.length >= 4) {
                        const left = ZemoAsk.safeNumber(box[0]);
                        const top = ZemoAsk.safeNumber(box[1]);
                        const width = Math.max(0, ZemoAsk.safeNumber(box[2]));
                        const height = Math.max(0, ZemoAsk.safeNumber(box[3]));
                        if (width > 0 && height > 0) {
                            lines.push("[[ocr:" + Math.round(left) + "," + Math.round(top) + "," + Math.round(width) + "," + Math.round(height) + "]]" + cleaned);
                        } else {
                            lines.push(cleaned);
                        }
                    } else {
                        lines.push(cleaned);
                    }
                }
            }
            const rawItems = parsed.items;
            if (lines.length == 0 && rawItems != null) {
                for (let i = 0; i < rawItems.length; i++) {
                    const raw = rawItems[i];
                    const textValue = raw.text != null ? raw.text : "";
                    const cleaned = ZemoAsk.cleanPpOcrV6ItemLine(textValue);
                    if (cleaned == "") continue;
                    const box = raw.box;
                    if (box != null) {
                        const left = ZemoAsk.safeNumber(box.left);
                        const top = ZemoAsk.safeNumber(box.top);
                        const right = ZemoAsk.safeNumber(box.right);
                        const bottom = ZemoAsk.safeNumber(box.bottom);
                        const width = Math.max(0, right - left);
                        const height = Math.max(0, bottom - top);
                        if (width > 0 && height > 0) {
                            lines.push("[[ocr:" + Math.round(left) + "," + Math.round(top) + "," + Math.round(width) + "," + Math.round(height) + "]]" + cleaned);
                        } else {
                            lines.push(cleaned);
                        }
                    } else {
                        lines.push(cleaned);
                    }
                }
            }
            if (lines.length > 0) return lines.join("\n");
            const fullText = parsed.fullText;
            if (fullText != null) return ZemoAsk.cleanRecognizedBillText(fullText);
        } catch (_err) {}
        return "";
    }

    private static safeNumber(value: number | null): number {
        if (value == null) return 0;
        return isNaN(value) ? 0 : value;
    }

    private static normalizeTextRecognitionResult(data: textRecognition.TextRecognitionResult): string {
        if (data == null) return "";
        const lineText = ZemoAsk.collectTextRecognitionLines(data);
        if (lineText.trim() != "") return lineText;
        if (data.value != null) return ZemoAsk.cleanRecognizedBillText(data.value.toString());
        return "";
    }

    private static collectTextRecognitionLines(data: textRecognition.TextRecognitionResult): string {
        if (data == null || data.blocks == null) return "";
        const lines = new Array<string>();
        for (let i = 0; i < data.blocks.length; i++) {
            const block = data.blocks[i];
            if (block == null || block.lines == null) continue;
            for (let j = 0; j < block.lines.length; j++) {
                const line = block.lines[j];
                if (line != null && line.value != null && line.value.toString().trim() != "") {
                    const cleaned = ZemoAsk.cleanRecognizedLine(line.value.toString());
                    if (cleaned != "") lines.push(ZemoAsk.formatOcrLineWithPosition(line, cleaned));
                }
            }
        }
        return lines.join("\n");
    }

    private static formatOcrLineWithPosition(line: Object, cleaned: string): string {
        const box = ZemoAsk.extractOcrLineBox(line);
        if (box == null || box.width <= 0 || box.height <= 0) return cleaned;
        return "[[ocr:" + Math.round(box.x) + "," + Math.round(box.y) + "," + Math.round(box.width) + "," + Math.round(box.height) + "]]" + cleaned;
    }

    private static extractOcrLineBox(line: Object): DeskPetOcrLineBox | null {
        try {
            const typedLine = line as DeskPetOcrLineLike;
            const rect = typedLine.textRect != null ? typedLine.textRect
                : (typedLine.boundingBox != null ? typedLine.boundingBox
                : (typedLine.rect != null ? typedLine.rect
                : (typedLine.box != null ? typedLine.box : null)));
            const rectBox = ZemoAsk.boxFromRect(rect);
            if (rectBox != null) return rectBox;

            const points = typedLine.cornerPoints != null ? typedLine.cornerPoints
                : (typedLine.points != null ? typedLine.points
                : (typedLine.vertexes != null ? typedLine.vertexes : null));
            return ZemoAsk.boxFromPoints(points);
        } catch (_err) {
            return null;
        }
    }

    private static boxFromRect(rect: DeskPetOcrRectLike | null): DeskPetOcrLineBox | null {
        if (rect == null) return null;
        const left = rect.left != null ? rect.left : (rect.x != null ? rect.x : -1);
        const top = rect.top != null ? rect.top : (rect.y != null ? rect.y : -1);
        let width = rect.width != null ? rect.width : (rect.w != null ? rect.w : -1);
        let height = rect.height != null ? rect.height : (rect.h != null ? rect.h : -1);
        const right = rect.right != null ? rect.right : -1;
        const bottom = rect.bottom != null ? rect.bottom : -1;
        if (width <= 0 && right > left) width = right - left;
        if (height <= 0 && bottom > top) height = bottom - top;
        if (left < 0 || top < 0 || width <= 0 || height <= 0) return null;
        return new DeskPetOcrLineBox(left, top, width, height);
    }

    private static boxFromPoints(points: Array<DeskPetOcrPointLike> | null): DeskPetOcrLineBox | null {
        if (points == null || points.length <= 0) return null;
        let minX = 999999;
        let minY = 999999;
        let maxX = -1;
        let maxY = -1;
        for (let i = 0; i < points.length; i++) {
            const p: DeskPetOcrPointLike = points[i];
            const x = p.x != null ? p.x : -1;
            const y = p.y != null ? p.y : -1;
            if (x < 0 || y < 0) continue;
            if (x < minX) minX = x;
            if (y < minY) minY = y;
            if (x > maxX) maxX = x;
            if (y > maxY) maxY = y;
        }
        if (maxX <= minX || maxY <= minY) return null;
        return new DeskPetOcrLineBox(minX, minY, maxX - minX, maxY - minY);
    }

    private static cleanRecognizedBillText(text: string): string {
        const lines = text.toString().replace(/\r\n/g, "\n").split("\n");
        const result = new Array<string>();
        for (let i = 0; i < lines.length; i++) {
            const cleaned = ZemoAsk.cleanRecognizedLine(lines[i]);
            if (cleaned != "") result.push(cleaned);
        }
        return result.join("\n");
    }

    private static cleanPpOcrV6ItemLine(raw: string): string {
        const line = raw.toString().trim();
        if (line == "") return "";
        if (ZemoAsk.isDeskPetPipLine(line)) return "";
        if (ZemoAsk.isStatusBarLine(line)) return "";
        if (/^[\u4e00-\u9fa5]$/.test(line) && line != "收" && line != "付") return "";
        if (line == "记录点什么…" || line == "记录点什么…." || line == "Mo Finance") return "";
        return line;
    }

    private static cleanRecognizedLine(raw: string): string {
        const line = raw.toString().trim();
        if (line == "") return "";
        if (ZemoAsk.isDeskPetPipLine(line)) return "";
        if (ZemoAsk.isStatusBarLine(line)) return "";
        if (ZemoAsk.isDeskPetOverlayLine(line)) return "";
        if (/^[\u4e00-\u9fa5]$/.test(line) && line != "收" && line != "付") return "";
        if (line == "记录点什么…" || line == "记录点什么…." || line == "Mo Finance") return "";
        return line;
    }

    private static isDeskPetPipLine(line: string): boolean {
        const normalized = line.toString().replace(/\s+/g, "").toLowerCase();
        if (normalized == "") return true;
        const pipExact = [
            "zemofinance", "cny", "asset", "assetcard", "我的资产", "的资产", "净资产总览",
            "等待金额", "正在识别屏幕内容", "请进入账单详情再记账", "再点暂停完成，5秒后自动确认",
            "识别到账单", "未识别到金额", "记账中", "请勿遮挡记账金额"
        ];
        for (let i = 0; i < pipExact.length; i++) {
            if (normalized == pipExact[i].replace(/\s+/g, "").toLowerCase()) return true;
        }
        if (normalized.indexOf("zemofinance") >= 0) return true;
        if (normalized.indexOf("进入账单详情再记账") >= 0) return true;
        if (normalized.indexOf("自动确认") >= 0 && normalized.indexOf("暂停") >= 0) return true;
        return false;
    }

    private static async takeScreenshotPixelMap(): Promise<image.PixelMap | null> {
        const captureOption: screenshot.CaptureOption = {
            displayId: 0
        };
        ZemoAsk.pipBillScreenshotInProgress = true;
        ZemoAsk.refreshPipContent();
        await new Promise<void>((resolve) => setTimeout(resolve, 80));
        try {
            return await screenshot.capture(captureOption);
        } finally {
            ZemoAsk.pipBillScreenshotInProgress = false;
            ZemoAsk.refreshPipContent();
        }
    }

    private static requestScreenCapturePermission(): Promise<boolean> {
        return new Promise((resolve) => {
            try {
                UTSHarmony.requestSystemPermission([CUSTOM_SCREEN_CAPTURE_PERMISSION],
                    (allRight: boolean, grantedList: string[]) => {
                        resolve(allRight || grantedList.indexOf(CUSTOM_SCREEN_CAPTURE_PERMISSION) >= 0);
                    },
                    (_doNotAskAgain: boolean, grantedList: string[]) => {
                        resolve(grantedList.indexOf(CUSTOM_SCREEN_CAPTURE_PERMISSION) >= 0);
                    }
                );
            } catch (permissionErr) {
                console.warn("[截图] 请求系统截屏权限失败：" + ZemoAsk.safeStringify(permissionErr));
                resolve(false);
            }
        });
    }

    private static safeStringify(value: Object | null): string {
        if (value == null) return "";
        try {
            const err = value as Error;
            const name = err.name != null ? err.name.toString() : "";
            const message = err.message != null ? err.message.toString() : "";
            const stack = err.stack != null ? err.stack.toString() : "";
            if (name != "" || message != "" || stack != "") {
                let text = "";
                if (name != "") text += name;
                if (message != "") text += (text == "" ? "" : ": ") + message;
                if (stack != "") text += (text == "" ? "" : "\n") + stack;
                return text;
            }
        } catch (_errCastErr) {}
        try {
            const json = JSON.stringify(value);
            if (json != "{}") return json;
        } catch (_jsonErr) {
        }
        try {
            return value.toString();
        } catch (_stringErr) {
            return "[unknown error]";
        }
    }

    private static formatDebugPreview(text: string): string {
        let value = (text ?? "").toString().replace(/\r\n/g, "\n").replace(/\s+/g, " / ").trim();
        if (value.length > 240) value = value.substring(0, 240) + "...";
        return value == "" ? "无" : value;
    }

    private static storeBillDebugStatus(status: string): void {
        try {
            uni.setStorageSync(DESK_PET_BILL_DEBUG_STATUS_KEY, status);
        } catch (_storeErr) {}
    }

    private static parseBillPreview(text: string): DeskPetBillPreview {
        const preview = new DeskPetBillPreview();
        const normalized = ZemoAsk.normalizeBillText(text);
        if (!ZemoAsk.looksLikeBillText(normalized)) {
            preview.status = "请勿遮挡记账金额";
            return preview;
        }
        preview.amount = ZemoAsk.extractBillAmount(normalized);
        preview.note = ZemoAsk.extractBillNote(normalized);
        preview.mode = ZemoAsk.detectBillMode(normalized);
        preview.category = ZemoAsk.detectBillCategory(normalized, preview.mode);
        preview.sourceApp = ZemoAsk.detectBillSourceApp(normalized);
        preview.payMethod = ZemoAsk.extractBillPayMethod(normalized);
        preview.cardTail = ZemoAsk.extractBillCardTail(normalized);
        preview.accountType = ZemoAsk.detectBillAccountType(normalized + "\n" + preview.payMethod);
        preview.status = preview.amount == "" ? "请勿遮挡记账金额" : "识别到账单";
        return preview;
    }

    private static detectBillMode(text: string): string {
        const signed = ZemoAsk.extractPrimarySignedAmount(text);
        if (ZemoAsk.containsAny(text, ["退款成功", "已退款", "退回", "原路退", "退款金额"])) return "refund";
        if (ZemoAsk.containsAny(text, ["零钱提现发起", "提现金额", "提现银行", "提现时间", "预计到账时间"])) return "transfer";
        if (ZemoAsk.isWeChatOutgoingRedPacket(text)) return "expense";
        if (ZemoAsk.isWeChatIncomingRedPacket(text)) return "income";
        if (ZemoAsk.containsAny(text, ["你已收款", "资金正在转入零钱通"])) return "income";
        if (/待.+确认收款/.test(text) || ZemoAsk.containsAny(text, ["等待对方领取"])) return "expense";
        if (text.indexOf("收款账户") >= 0 && text.indexOf("付款账户") < 0 && text.indexOf("付款方式") < 0) return "income";
        if (ZemoAsk.containsAny(text, ["转账给", "转账到", "转出", "转入", "转账成功", "转账收款"])) {
            if (signed.startsWith("+") || ZemoAsk.containsAny(text, ["转入", "转账收款", "收到转账"])) return "income";
            return "expense";
        }
        if (signed.startsWith("+")) return "income";
        if (signed.startsWith("-")) return "expense";
        let incomeScore = ZemoAsk.scoreKeywords(text, ["已收款", "收款成功", "收款金额", "收款账户", "收入", "入账", "到账", "工资", "薪资", "奖金", "报销到账", "+¥", "+￥"]);
        let expenseScore = ZemoAsk.scoreKeywords(text, ["付款成功", "支付成功", "实付", "付款金额", "消费", "支出", "扣款", "已付款", "支付给", "向商家付款", "未入账", "交易卡号", "银行交易类型", "-¥", "-￥"]);
        incomeScore += ZemoAsk.countSignedAmountLines(text, "+") * 120;
        expenseScore += ZemoAsk.countSignedAmountLines(text, "-") * 120;
        if (ZemoAsk.containsAny(text, ["商家转账", "来自拼多多", "现金红包", "收款方式", "你已收款"])) incomeScore += 60;
        return incomeScore > expenseScore ? "income" : "expense";
    }

    private static isWeChatIncomingRedPacket(text: string): boolean {
        return text.indexOf("发出的红包") >= 0
            && text.indexOf("红包金额") < 0
            && text.indexOf("等待对方领取") < 0;
    }

    private static isWeChatOutgoingRedPacket(text: string): boolean {
        return text.indexOf("红包金额") >= 0
            || text.indexOf("等待对方领取") >= 0;
    }

    private static looksLikeBillText(text: string): boolean {
        if (text.trim() == "") return false;
        return ZemoAsk.containsAny(text, [
            "账单", "付款", "支付", "收款", "收入", "支出", "交易", "订单", "商户", "商家",
            "支付宝", "微信", "银联", "银行卡", "退款", "实付", "消费", "转账", "入账",
            "红包", "零钱通"
        ]);
    }

    private static looksLikeFundHoldingListText(text: string): boolean {
        if (text.trim() == "") return false;
        const plain = ZemoAsk.stripOcrMetaFromBillText(text.toString().replace(/\r\n/g, "\n"));
        if (plain.trim() == "") return false;
        const hasMarketHeader = ZemoAsk.containsAny(plain, ["持仓汇总", "自选股", "持仓股", "最新", "涨幅", "涨跌"]);
        const hasMarketFund = ZemoAsk.containsAny(plain, ["ETF", "基金", "场内"]);
        const hasCode = /(^|\D)[0-9]{5,8}(\D|$)/.test(plain);
        const hasBrokerHeader = ZemoAsk.containsAny(plain, ["证券代码", "持仓/可用", "持仓可用", "证券名称", "名称/代码", "持仓市值", "持有市值"])
            && ZemoAsk.containsAny(plain, ["市值", "盈亏", "收益", "持仓"]);
        const hasBrokerContext = ZemoAsk.containsAny(plain, ["买入", "卖出", "撤单", "查询", "持仓管理", "总资产", "总市值", "证券", "同花顺"]);
        const hasBrokerFund = ZemoAsk.containsAny(plain, ["ETF", "基金", "LOF", "QDII", "指数", "标普", "纳指", "债"]);
        const hasBrokerMetric = plain.indexOf("%") >= 0 || /[+\-]?[0-9]+(?:\.[0-9]+)?/.test(plain);
        const hasAlipayHolding = plain.indexOf("我的持有") >= 0
            && plain.indexOf("金额/昨日收益") >= 0
            && plain.indexOf("持有收益/率") >= 0
            && ZemoAsk.containsAny(plain, ["基金", "财富号", "QDII", "混合", "债", "股票", "指数"]);
        return (hasMarketHeader && hasMarketFund && hasCode) || (hasBrokerHeader && hasBrokerContext && hasBrokerFund && hasBrokerMetric) || hasAlipayHolding;
    }

    private static detectFundHoldingSourceApp(text: string): string {
        const plain = ZemoAsk.stripOcrMetaFromBillText(text.toString().replace(/\r\n/g, "\n"));
        if (ZemoAsk.containsAny(plain, ["支付宝", "蚂蚁财富"])) return "支付宝理财";
        if (ZemoAsk.containsAny(plain, ["天天基金"])) return "天天基金";
        if (ZemoAsk.containsAny(plain, ["同花顺"])) return "同花顺持仓";
        const lines = plain.split("\n");
        for (let i = 0; i < lines.length && i < 10; i++) {
            const line = lines[i].trim();
            if (line.indexOf("证券") >= 0 || line.indexOf("基金") >= 0) return line;
        }
        return "理财持仓";
    }

    private static normalizeBillText(text: string): string {
        const raw = text.toString().replace(/\r\n/g, "\n").replace(/\t/g, " ").trim();
        const positioned = ZemoAsk.extractLatestBillBlockByPosition(raw);
        if (positioned != "") return ZemoAsk.stripOcrMetaFromBillText(positioned);
        const normalized = ZemoAsk.stripOcrMetaFromBillText(raw);
        return ZemoAsk.extractLatestBillBlock(normalized);
    }

    private static extractLatestBillBlockByPosition(text: string): string {
        const lines = ZemoAsk.parseBillOcrLines(text);
        if (lines.length == 0) return "";
        const cleaned = new Array<DeskPetBillOcrLine>();
        for (let i = 0; i < lines.length; i++) {
            const item = lines[i];
            if (item.text == "") continue;
            if (ZemoAsk.isStatusBarLine(item.text)) continue;
            if (ZemoAsk.isDeskPetOverlayLine(item.text)) continue;
            cleaned.push(item);
        }
        if (cleaned.length == 0) return "";
        const sourceContext = ZemoAsk.findSourceContextLineFromOcr(cleaned);
        const detailBlock = ZemoAsk.extractPositionedDetailBlock(cleaned, sourceContext);
        if (detailBlock != "") return detailBlock;
        return "";
    }

    private static extractPositionedDetailBlock(lines: DeskPetBillOcrLine[], sourceContext: string): string {
        let hasDetailSignal = false;
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i].text;
            if (line == "账单详情" || line == "交易详情" || ZemoAsk.containsAny(line, ["付款方式", "支付方式", "交易单号", "订单号", "当前状态", "创建时间", "完成时间"])) {
                hasDetailSignal = true;
                break;
            }
        }
        if (!hasDetailSignal) return "";

        let amountIndex = -1;
        let bestScore = -9999;
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            if (ZemoAsk.extractBillAmountCandidates(line.text).length == 0) continue;
            const near = ZemoAsk.nearbyOcrText(lines, line, 190);
            let score = ZemoAsk.scoreBillAmountCandidate(near, i, lines.length, ZemoAsk.cleanBillAmount(line.text));
            if (ZemoAsk.containsAny(near, ["交易详情", "账单详情", "支付", "付款"])) score += 80;
            if (ZemoAsk.containsAny(near, ["余额", "优惠", "订单号", "交易单号"])) score -= 120;
            if (amountIndex < 0 || score > bestScore || (score == bestScore && line.y > lines[amountIndex].y)) {
                amountIndex = i;
                bestScore = score;
            }
        }
        if (amountIndex < 0) return "";

        const anchor = lines[amountIndex];
        let startIndex = amountIndex;
        for (let i = amountIndex - 1; i >= 0; i--) {
            const item = lines[i];
            if (anchor.y - item.y > 760) break;
            const line = item.text;
            if (line == "" || ZemoAsk.isLowValueBillLine(line)) continue;
            if (ZemoAsk.extractBillAmountCandidates(line).length > 0) continue;
            if (ZemoAsk.containsAny(line, ["使用", "付款方式", "支付方式", "扣款方式", "收款账户", "付款账户", "入账账户"])) continue;
            if (ZemoAsk.containsAny(line, ["查看", "前往", "订单详情页", "账单详情", "交易详情"])) continue;
            startIndex = i;
            break;
        }

        let endIndex = amountIndex;
        for (let i = amountIndex + 1; i < lines.length; i++) {
            const line = lines[i].text;
            if (ZemoAsk.isBillListTabLine(line)) break;
            endIndex = i;
            if (line == "账单详情" || line == "交易详情") break;
        }

        const blockLines = new Array<string>();
        if (sourceContext != "") blockLines.push(sourceContext);
        for (let i = startIndex; i <= endIndex && i < lines.length; i++) {
            const line = lines[i].text;
            if (line == "" || ZemoAsk.isBillListTabLine(line)) continue;
            if (blockLines.indexOf(lines[i].raw) < 0 && blockLines.indexOf(line) < 0) blockLines.push(lines[i].raw);
        }
        const block = blockLines.join("\n").trim();
        const plain = ZemoAsk.stripOcrMetaFromBillText(block);
        return ZemoAsk.isCandidateBillBlock(plain) ? block : "";
    }

    private static parseBillOcrLines(text: string): DeskPetBillOcrLine[] {
        const rawLines = text.split("\n");
        const result = new Array<DeskPetBillOcrLine>();
        const reg = /^\[\[ocr:([0-9.\-]+),([0-9.\-]+),([0-9.\-]+),([0-9.\-]+)\]\](.*)$/;
        for (let i = 0; i < rawLines.length; i++) {
            const raw = rawLines[i].trim();
            const m = reg.exec(raw);
            if (m == null || m.length < 6) continue;
            const value = m[5].trim();
            if (value == "") continue;
            const item = new DeskPetBillOcrLine();
            item.raw = raw;
            item.text = value;
            item.x = parseFloat(m[1]);
            item.y = parseFloat(m[2]);
            item.w = parseFloat(m[3]);
            item.h = parseFloat(m[4]);
            item.index = i;
            result.push(item);
        }
        result.sort((a: DeskPetBillOcrLine, b: DeskPetBillOcrLine): number => {
            const dy = a.y - b.y;
            if (Math.abs(dy) > 8) return dy;
            return a.x - b.x;
        });
        return result;
    }

    private static nearbyOcrText(lines: DeskPetBillOcrLine[], anchor: DeskPetBillOcrLine, tolerance: number): string {
        const result = new Array<string>();
        const y = ZemoAsk.centerBillOcrY(anchor);
        for (let i = 0; i < lines.length; i++) {
            if (Math.abs(ZemoAsk.centerBillOcrY(lines[i]) - y) <= tolerance) result.push(lines[i].text);
        }
        return result.join("\n");
    }

    private static centerBillOcrY(line: DeskPetBillOcrLine): number {
        return line.y + line.h / 2;
    }

    private static findSourceContextLineFromOcr(lines: DeskPetBillOcrLine[]): string {
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i].text;
            if (line == "微信支付" || line == "支付宝" || line == "云闪付") return line;
        }
        return "";
    }

    private static stripOcrMetaFromBillText(text: string): string {
        const lines = text.split("\n");
        const result = new Array<string>();
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i].replace(/^\[\[ocr:[0-9.\-]+,[0-9.\-]+,[0-9.\-]+,[0-9.\-]+\]\]/, "").trim();
            if (line != "") result.push(line);
        }
        return result.join("\n");
    }

    private static extractLatestBillBlock(text: string): string {
        const lines = text.split("\n");
        const cleaned = new Array<string>();
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i].trim();
            if (line == "") continue;
            if (ZemoAsk.isStatusBarLine(line)) continue;
            if (ZemoAsk.isDeskPetOverlayLine(line)) continue;
            cleaned.push(line);
        }
        if (cleaned.length == 0) return "";
        const sourceContext = ZemoAsk.findSourceContextLine(cleaned);
        const listCardBlock = ZemoAsk.extractWeChatListCardBlock(cleaned, sourceContext);
        if (listCardBlock != "") return listCardBlock;

        const candidates = new Array<string>();
        let current = new Array<string>();
        let inBlock = false;
        let seenAmount = false;
        for (let i = 0; i < cleaned.length; i++) {
            const line = cleaned[i];
            if (ZemoAsk.isBillListTabLine(line)) break;

            if (ZemoAsk.isNextBillDividerLine(line)) {
                const pending = ZemoAsk.withSourceContext(current.join("\n").trim(), sourceContext);
                if (ZemoAsk.isCandidateBillBlock(pending)) candidates.push(pending);
                current = new Array<string>();
                inBlock = true;
                seenAmount = false;
                continue;
            }

            if (!inBlock) {
                if (!ZemoAsk.isStrongBillStartLine(line)) continue;
                inBlock = true;
            }

            current.push(line);
            if (ZemoAsk.extractBillAmountCandidates(line).length > 0) seenAmount = true;
            if (seenAmount && (line == "账单详情" || line == "交易详情")) {
                const block = ZemoAsk.withSourceContext(current.join("\n").trim(), sourceContext);
                if (ZemoAsk.isCandidateBillBlock(block)) candidates.push(block);
                current = new Array<string>();
                inBlock = false;
                seenAmount = false;
            }
        }
        const tail = ZemoAsk.withSourceContext(current.join("\n").trim(), sourceContext);
        if (ZemoAsk.isCandidateBillBlock(tail)) candidates.push(tail);
        if (candidates.length > 0) return candidates[candidates.length - 1];

        let start = 0;
        for (let i = 0; i < cleaned.length; i++) {
            if (ZemoAsk.isStrongBillStartLine(cleaned[i])) {
                start = i;
                break;
            }
        }

        const result = new Array<string>();
        let fallbackSeenAmount = false;
        for (let i = start; i < cleaned.length; i++) {
            const line = cleaned[i];
            if (fallbackSeenAmount && ZemoAsk.isBillListTabLine(line)) break;
            if (fallbackSeenAmount && (line == "账单详情" || line == "交易详情")) {
                result.push(line);
                break;
            }
            result.push(line);
            if (ZemoAsk.extractBillAmountCandidates(line).length > 0) fallbackSeenAmount = true;
            if (fallbackSeenAmount && i > start && ZemoAsk.isNextBillDividerLine(line)) break;
        }
        return result.join("\n").trim();
    }

    private static extractWeChatListCardBlock(lines: string[], sourceContext: string): string {
        let markerIndex = -1;
        for (let i = lines.length - 1; i >= 0; i--) {
            const line = lines[i];
            if (ZemoAsk.containsAny(line, ["你有一笔收款入账", "你有一笔付款", "收款金额", "付款金额", "零钱提现发起", "提现金额", "提现银行", "提现时间", "预计到账时间"])) {
                markerIndex = i;
                break;
            }
        }
        if (markerIndex < 0) return "";

        let start = markerIndex;
        for (let i = markerIndex - 1; i >= 0; i--) {
            const line = lines[i];
            if (ZemoAsk.isNextBillDividerLine(line)) {
                start = i + 1;
                break;
            }
            if (ZemoAsk.isBillListTabLine(line)) break;
            if (ZemoAsk.isLikelyListMerchantLine(line)) start = i;
        }

        let end = lines.length - 1;
        for (let i = markerIndex; i < lines.length; i++) {
            const line = lines[i];
            if (ZemoAsk.isBillListTabLine(line)) {
                end = i - 1;
                break;
            }
            if (i > markerIndex && ZemoAsk.isNextBillDividerLine(line)) {
                end = i - 1;
                break;
            }
            if (line == "账单详情" || line == "交易详情") {
                end = i;
                break;
            }
        }

        const blockLines = new Array<string>();
        for (let i = start; i <= end && i < lines.length; i++) {
            const line = lines[i];
            if (line == "") continue;
            if (ZemoAsk.isNextBillDividerLine(line)) continue;
            blockLines.push(line);
        }
        const block = ZemoAsk.withSourceContext(blockLines.join("\n").trim(), sourceContext);
        return ZemoAsk.isCandidateBillBlock(block) ? block : "";
    }

    private static isLikelyListMerchantLine(line: string): boolean {
        if (line == "") return false;
        if (ZemoAsk.isStatusBarLine(line)) return false;
        if (ZemoAsk.isLowValueBillLine(line)) return false;
        if (ZemoAsk.extractBillAmountCandidates(line).length > 0) return false;
        if (ZemoAsk.containsAny(line, ["微信支付", "支付宝", "云闪付", "账单详情", "交易详情"])) return false;
        return line.length <= 24;
    }

    private static findSourceContextLine(lines: string[]): string {
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            if (line == "微信支付" || line == "支付宝" || line == "云闪付") return line;
        }
        return "";
    }

    private static withSourceContext(block: string, sourceContext: string): string {
        const value = block.trim();
        if (value == "") return "";
        if (sourceContext == "") return value;
        if (value.indexOf(sourceContext) >= 0) return value;
        return sourceContext + "\n" + value;
    }

    private static isCandidateBillBlock(text: string): boolean {
        if (text.trim() == "") return false;
        if (ZemoAsk.containsAny(text, ["零钱提现发起", "提现金额", "提现银行", "提现时间", "预计到账时间"])) return true;
        if (ZemoAsk.extractBillAmountCandidates(text).length == 0) return false;
        return ZemoAsk.containsAny(text, [
            "账单详情", "交易详情", "收款金额", "付款金额", "订单", "商家", "商户",
            "微信支付", "支付宝", "云闪付", "你有一笔收款入账", "支付给商家"
        ]);
    }

    private static isStrongBillStartLine(line: string): boolean {
        if (line == "微信支付" || line == "支付宝" || line == "云闪付") return true;
        if (/^[+\-]?\s*[¥￥]\s*[0-9]+(?:\.[0-9]{1,2})?$/.test(line)) return true;
        if (ZemoAsk.containsAny(line, ["支付成功", "收款成功", "付款成功", "你已收款", "你有一笔收款入账"])) return true;
        return false;
    }

    private static isStatusBarLine(line: string): boolean {
        if (/^\d{1,2}:\d{2}$/.test(line)) return true;
        if (/^\d{1,2}:\d{2}\s*[（(]?$/.test(line)) return true;
        if (/^\d{1,2}\s*k\/s$/i.test(line)) return true;
        if (/^(wifi|5g|4g|volte|hd)$/i.test(line)) return true;
        if (/^\d{1,3}%$/.test(line)) return true;
        return false;
    }

    private static isDeskPetOverlayLine(line: string): boolean {
        const normalized = line.replace(/\s+/g, "").toLowerCase();
        if (normalized == "") return true;
        if (normalized.indexOf("zemofinance") >= 0) return true;
        if (normalized == "cny" || normalized == "asset" || normalized == "我的资产" || normalized == "净资产总览") return true;
        if (/^[0-9]{3,}(?:\.[0-9]{1,2})?$/.test(normalized)) return true;
        return false;
    }

    private static isBillListTabLine(line: string): boolean {
        return line == "我的账单" || line == "支付服务" || line == "摇优惠";
    }

    private static isNextBillDividerLine(line: string): boolean {
        return /^(今天|昨天|前天)\s*\d{1,2}:\d{2}$/.test(line)
            || /^\d{1,2}月\d{1,2}日\s*\d{1,2}:\d{2}$/.test(line)
            || /^(周|星期)[一二三四五六日天]\s*\d{1,2}:\d{2}$/.test(line);
    }

    private static extractBillAmount(text: string): string {
        const normalized = text.replace(/,/g, "");
        const signed = ZemoAsk.extractPrimarySignedAmount(normalized);
        if (signed != "") {
            const signedAmount = ZemoAsk.cleanBillAmount(signed);
            if (signedAmount != "") return signedAmount;
        }
        const lines = normalized.split("\n");
        let bestAmount = "";
        let bestScore = -999;
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i].trim();
            if (line == "") continue;
            const prev = i > 0 ? lines[i - 1].trim() : "";
            const next = i < lines.length - 1 ? lines[i + 1].trim() : "";
            const context = prev + "\n" + line + "\n" + next;
            const signedAmountLine = /^[+\-]\s*[0-9]+(?:\.[0-9]{1,2})?$/.test(line);
            const standaloneDecimalLine = /^[0-9]+\.[0-9]{2}$/.test(line);
            if (!signedAmountLine && !standaloneDecimalLine && !ZemoAsk.looksLikeBillText(context)) continue;
            const candidates = ZemoAsk.extractBillAmountCandidates(line);
            for (let j = 0; j < candidates.length; j++) {
                const amount = ZemoAsk.cleanBillAmount(candidates[j]);
                if (amount == "") continue;
                const score = ZemoAsk.scoreBillAmountCandidate(context, i, lines.length, amount);
                if (score > bestScore) {
                    bestScore = score;
                    bestAmount = amount;
                }
            }
        }
        if (bestAmount != "") return bestAmount;
        const loose = ZemoAsk.extractBillAmountCandidates(normalized);
        for (let i = 0; i < loose.length; i++) {
            const amount = ZemoAsk.cleanBillAmount(loose[i]);
            if (amount != "") return amount;
        }
        return "";
    }

    private static extractBillAmountCandidates(text: string): string[] {
        const result = new Array<string>();
        const patterns: RegExp[] = [
            /(^|\s)[+\-]\s*(?:[¥￥$£€])\s*([0-9]+(?:\.[0-9]{1,2})?)(\s|$)/g,
            /[+\-]?\s*[¥￥]\s*([0-9]+(?:\.[0-9]{1,2})?)/g,
            /(?:实付|付款金额|支付金额|消费金额|支出金额|收款金额|收入金额|入账金额|退款金额|订单金额|合计|总计|金额)[^0-9¥￥]{0,12}[¥￥]?\s*([0-9]+(?:\.[0-9]{1,2})?)/g,
            /([0-9]+(?:\.[0-9]{1,2})?)\s*元/g,
            /(^|\s)([0-9]+\.[0-9]{2})(\s|$)/g,
            /(^|\s)[+\-]\s*([0-9]+(?:\.[0-9]{1,2})?)(\s|$)/g
        ];
        for (let i = 0; i < patterns.length; i++) {
            const reg = patterns[i];
            let match = reg.exec(text);
            while (match != null) {
                if (match.length > 2 && (match[0].indexOf("+") >= 0 || match[0].indexOf("-") >= 0)) {
                    result.push(match[2]);
                } else if (match.length > 2 && /^[0-9]/.test(match[2])) {
                    result.push(match[2]);
                } else if (match.length > 1) {
                    result.push(match[1]);
                }
                match = reg.exec(text);
            }
        }
        return result;
    }

    private static scoreBillAmountCandidate(line: string, lineIndex: number, lineCount: number, amount: string): number {
        let score = 0;
        if (ZemoAsk.containsAny(line, ["实付", "付款金额", "支付金额", "消费金额", "支出金额", "收款金额", "收入金额", "入账金额", "退款金额"])) score += 80;
        if (ZemoAsk.containsAny(line, ["使用", "支付"]) && line.indexOf("支付分先用后付") < 0) score += 90;
        if (ZemoAsk.containsAny(line, ["实际支付", "已支付", "扣款成功"])) score += 80;
        if (ZemoAsk.containsAny(line, ["预计扣费", "预计金额", "预计在", "预计", "不会超过", "自动确认收"])) score -= 180;
        if (/(^|\n)[+\-]\s*[0-9]+(?:\.[0-9]{1,2})?(\n|$)/.test(line)) score += 90;
        if (ZemoAsk.containsAny(line, ["合计", "总计", "订单金额", "金额"])) score += 45;
        if (line.indexOf("¥") >= 0 || line.indexOf("￥") >= 0 || line.indexOf("元") >= 0) score += 20;
        if (ZemoAsk.containsAny(line, ["余额", "优惠", "红包抵扣", "积分", "卡号", "订单号", "交易单号", "流水号", "商户单号"])) score -= 80;
        if (ZemoAsk.containsAny(line, ["时间", "日期", "年", "月", "日"])) score -= 40;
        if (lineIndex <= 4) score += 10;
        if (lineCount > 0 && lineIndex >= lineCount - 3) score -= 5;
        const val = parseFloat(amount);
        if (!isNaN(val) && val > 0 && val < 100000) score += 8;
        return score;
    }

    private static cleanBillAmount(raw: string): string {
        const value = parseFloat(raw.replace(/[^0-9.]/g, ""));
        if (isNaN(value) || value <= 0 || value > 100000000) return "";
        return value.toFixed(2);
    }

    private static extractPrimarySignedAmount(text: string): string {
        const lines = text.split("\n");
        const reg = /^[+\-]\s*[¥￥$£€]?\s*[0-9]+(?:\.[0-9]{1,2})?$/;
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i].trim();
            if (reg.test(line)) return line;
        }
        return "";
    }

    private static extractBillNote(text: string): string {
        const lines = text.split("\n");
        if (ZemoAsk.containsAny(text, ["零钱提现发起", "提现金额", "提现银行"])) return "微信零钱提现";
        const wechatNote = ZemoAsk.extractWeChatBillNote(text);
        if (wechatNote != "") return wechatNote;

        const labelPattern = /^(商户|商家|商户全称|店铺|收款方|付款方|对方|交易对象|商品|商品说明|商品名称|说明|备注|订单标题|交易说明|付款给|支付给|转账给|转账备注)[:：\s]+/;
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i].trim();
            if (line == "") continue;
            if (labelPattern.test(line)) {
                const candidate = ZemoAsk.cleanBillNote(line.replace(labelPattern, ""));
                if (candidate != "") return candidate;
            }
            if (/^(转账备注|备注|说明)$/.test(line) && i < lines.length - 1) {
                const candidate = ZemoAsk.cleanBillNote(lines[i + 1].trim());
                if (candidate != "" && !ZemoAsk.isLowValueBillLine(candidate)) return candidate;
            }
        }
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i].trim();
            if (line == "" || ZemoAsk.isLowValueBillLine(line)) continue;
            const candidate = ZemoAsk.cleanBillNote(line);
            if (candidate != "") return candidate;
        }
        return "桌宠识别账单";
    }

    private static extractWeChatBillNote(text: string): string {
        const lines = text.split("\n");
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i].trim();
            if (line.indexOf("发出的红包") >= 0) {
                const sender = ZemoAsk.cleanBillNote(line.replace("发出的红包", ""));
                if (sender != "") return ZemoAsk.isWeChatOutgoingRedPacket(text) ? "微信红包" : sender + "的红包";
                return "微信红包";
            }
            const pendingMatch = /待(.+)确认收款/.exec(line);
            if (pendingMatch != null && pendingMatch.length > 1) {
                const target = ZemoAsk.cleanBillNote(pendingMatch[1]);
                if (target != "") return "转账给" + target;
                return "微信转账";
            }
            if (line.indexOf("你已收款") >= 0) {
                const payer = ZemoAsk.findNearestWeChatBillName(lines, i);
                return payer == "" ? "微信转账收款" : payer + "转账";
            }
        }
        return "";
    }

    private static findNearestWeChatBillName(lines: string[], anchor: number): string {
        let i = anchor - 1;
        while (i >= 0 && anchor - i <= 4) {
            const candidate = ZemoAsk.cleanBillNote(lines[i].trim());
            if (candidate != "" && !ZemoAsk.isLowValueBillLine(candidate)) return candidate;
            i -= 1;
        }
        return "";
    }

    private static cleanBillNote(raw: string): string {
        let note = raw.trim();
        note = note.replace(/[¥￥]\s*[0-9]+(?:\.[0-9]{1,2})?/g, "");
        note = note.replace(/(^|[\s,，])([0-9]+(?:\.[0-9]{1,2})?)\s*元/g, "$1");
        note = note.replace(/(^|[\s,，])([0-9]+\.[0-9]{1,2})(?=$|[\s,，])/g, "$1");
        note = note.replace(/(实付|付款|支付|金额|合计|收款|退款|收入|支出)[:：\s]*/g, "");
        note = note.replace(/\s+/g, " ").trim();
        if (/^[:：]+$/.test(note)) return "";
        if (ZemoAsk.isLikelyOcrNoise(note)) return "";
        if (note.length > 24) note = note.substring(0, 24);
        return note;
    }

    private static isLowValueBillLine(line: string): boolean {
        if (ZemoAsk.isLikelyOcrNoise(line)) return true;
        if (/^[\u4e00-\u9fa5]$/.test(line) && line != "收" && line != "付") return true;
        if (line == "记录点什么…" || line == "记录点什么…." || line == "Mo Finance") return true;
        if (/^\d{1,2}:\d{2}$/.test(line)) return true;
        if (/^[+\-]\s*[¥￥$£€]?\s*[0-9]+(?:\.[0-9]{1,2})?$/.test(line)) return true;
        if (/^(商户|商家|商户全称|店铺|收款方|付款方|对方|交易对象|商品|商品说明|商品名称|说明|备注|订单标题|交易说明|付款给|支付给|转账给|转账备注|收款账户|付款账户)[:：\s]*$/.test(line)) return true;
        if (/^[¥￥]?\s*[0-9]+(?:\.[0-9]{1,2})?\s*元?$/.test(line)) return true;
        if (/^\d{4}[-/]\d{1,2}[-/]\d{1,2}/.test(line)) return true;
        return ZemoAsk.containsAny(line, ["支付宝", "微信支付", "云闪付", "向商家付款", "交易成功", "支付成功", "付款成功", "收款成功", "账单详情", "交易详情", "未入账", "账单分类", "订单号", "交易单号", "付款方式", "支付方式", "当前状态", "创建时间", "完成时间", "交易卡号", "交易时间", "国家或地区", "原交易金额", "币种", "银行交易类型", "所属账本", "请选择", "分类", "消费", "封面由个人制作", "了解红包封面", "刚刚", "[转账]", "零钱"]);
    }

    private static isLikelyOcrNoise(line: string): boolean {
        const value = line.trim();
        if (value == "") return true;
        if (/^[|!！:：;；'"`.,，。()（）\[\]{}<>《》\\\/=_\-+*~^]+$/.test(value)) return true;
        if (/^[A-Za-z]:[!！]+$/.test(value)) return true;
        if (/^[\u4e00-\u9fa5]{1,3}$/.test(value) && ZemoAsk.containsAny(value, ["浓", "悶", "闷", "路"])) return true;
        return false;
    }

    private static detectBillCategory(text: string, mode: string): string {
        if (mode == "transfer") return "transfer";
        if (mode == "refund") return "income_refund";
        if (mode == "income") {
            if (ZemoAsk.containsAny(text, ["理财", "基金", "收益", "分红", "利息"])) return "income_finance";
            if (ZemoAsk.containsAny(text, ["礼金", "红包", "转账"])) return "income_gift";
            if (ZemoAsk.containsAny(text, ["奖金", "奖励", "补贴", "退款", "退回"])) return "income_reward";
            return "income_salary";
        }
        if (ZemoAsk.containsAny(text, ["发票", "税", "报销"])) return "expense_tax";
        if (ZemoAsk.containsAny(text, ["打车", "滴滴", "公交", "地铁", "高铁", "火车", "机票", "加油", "停车"])) return "expense_transport";
        if (ZemoAsk.containsAny(text, ["礼金", "红包", "份子钱", "转账", "确认收款"])) return "expense_gift";
        return "expense_shopping";
    }

    private static detectBillSourceApp(text: string): string {
        if (ZemoAsk.containsAny(text, ["支付宝", "花呗", "余额宝"])) return "支付宝";
        if (ZemoAsk.containsAny(text, ["微信支付", "微信零钱", "微信收款", "零钱通", "发出的红包", "红包金额", "[转账]", "确认收款"])) return "微信";
        if (ZemoAsk.containsAny(text, ["云闪付", "银联"])) return "云闪付";
        if (ZemoAsk.containsAny(text, ["招商银行", "招行"])) return "招商银行";
        if (ZemoAsk.containsAny(text, ["工商银行", "工行"])) return "工商银行";
        if (ZemoAsk.containsAny(text, ["建设银行", "建行"])) return "建设银行";
        if (ZemoAsk.containsAny(text, ["农业银行", "农行"])) return "农业银行";
        if (ZemoAsk.containsAny(text, ["中国银行", "中行"])) return "中国银行";
        return "";
    }

    private static extractBillPayMethod(text: string): string {
        const lines = text.split("\n");
        const labelPattern = /^(付款方式|支付方式|扣款方式|收款账户|入账账户|付款账户|银行卡|支付账户)[:：\s]+/;
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i].trim();
            if (line == "") continue;
            if (labelPattern.test(line)) return line.replace(labelPattern, "").trim();
            if (ZemoAsk.containsAny(line, ["零钱", "支付宝余额", "余额宝", "花呗", "银行卡", "储蓄卡", "信用卡"]) && line.length <= 48) return line;
        }
        return "";
    }

    private static extractBillCardTail(text: string): string {
        const reg = /(?:尾号|末四位|卡号|账户)[^\n]*?([0-9]{3,6})(?!.*[0-9])/g;
        let match = reg.exec(text);
        let result = "";
        while (match != null) {
            if (match.length > 1) result = match[1];
            match = reg.exec(text);
        }
        return result;
    }

    private static detectBillAccountType(text: string): string {
        if (ZemoAsk.containsAny(text, ["信用卡", "花呗", "白条"])) return "信用";
        if (ZemoAsk.containsAny(text, ["储蓄卡", "借记卡"])) return "储蓄";
        if (ZemoAsk.containsAny(text, ["零钱", "支付宝余额", "余额宝", "钱包"])) return "钱包";
        if (ZemoAsk.containsAny(text, ["银行卡", "卡号"])) return "银行卡";
        return "";
    }

    private static containsAny(text: string, keywords: string[]): boolean {
        for (let i = 0; i < keywords.length; i++) {
            if (text.indexOf(keywords[i]) >= 0) return true;
        }
        return false;
    }

    private static scoreKeywords(text: string, keywords: string[]): number {
        let score = 0;
        for (let i = 0; i < keywords.length; i++) {
            if (text.indexOf(keywords[i]) >= 0) score += keywords[i].length;
        }
        return score;
    }

    private static countSignedAmountLines(text: string, sign: string): number {
        const lines = text.split("\n");
        let count = 0;
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i].trim();
            if (line.startsWith(sign) && /^[+\-]\s*[¥￥$£€]?\s*[0-9]+(?:\.[0-9]{1,2})?$/.test(line)) count += 1;
        }
        return count;
    }

    private static buildFinanceRecordLink(preview: DeskPetBillPreview): string {
        const params = new Array<string>();
        if (preview.amount != "") params.push("amount=" + encodeURIComponent(preview.amount));
        if (preview.note != "") params.push("note=" + encodeURIComponent(preview.note));
        if (preview.category != "") params.push("category=" + encodeURIComponent(preview.category));
        if (preview.sourceApp != "") params.push("sourceApp=" + encodeURIComponent(preview.sourceApp));
        if (preview.payMethod != "") params.push("payMethod=" + encodeURIComponent(preview.payMethod));
        params.push("activeMode=" + encodeURIComponent(preview.mode));
        params.push("deskPetAutoBackground=1");
        return FINANCE_RECORD_DEEP_LINK + "?" + params.join("&");
    }

    private static markPendingDeskPetBillCapture(clearStored: boolean = true): void {
        try {
            if (clearStored) ZemoAsk.clearStoredBillCaptureText();
            uni.setStorageSync(DESK_PET_PENDING_BILL_CAPTURE_KEY, "pip-return:" + Date.now().toString());
        } catch (err) {
            console.warn("Failed to store pending desk pet bill capture: " + JSON.stringify(err));
        }
    }

    private static clearStoredBillCaptureText(): void {
        try { uni.removeStorageSync(DESK_PET_CAPTURED_BILL_TEXT_KEY); } catch (_capturedErr) {}
        try { uni.removeStorageSync(DESK_PET_CAPTURED_SCREEN_IMAGE_KEY); } catch (_imageErr) {}
        try { uni.removeStorageSync(DESK_PET_BILL_DEBUG_TEXT_KEY); } catch (_debugTextErr) {}
        try { uni.removeStorageSync(DESK_PET_BILL_DEBUG_STATUS_KEY); } catch (_debugStatusErr) {}
    }

    private static markPendingRecordOpen(link: string): void {
        try {
            uni.setStorageSync(DESK_PET_PENDING_RECORD_OPEN_KEY, Date.now().toString());
            const queryIndex = link.indexOf("?");
            if (queryIndex >= 0 && queryIndex < link.length - 1) {
                uni.setStorageSync(DESK_PET_PENDING_RECORD_QUERY_KEY, link.substring(queryIndex + 1));
            } else {
                uni.removeStorageSync(DESK_PET_PENDING_RECORD_QUERY_KEY);
            }
        } catch (err) {
            console.warn("Failed to store pending desk pet record open: " + JSON.stringify(err));
        }
    }

    private static clearPendingFinanceRecordAction(): void {
        try { uni.removeStorageSync(DESK_PET_PENDING_RECORD_OPEN_KEY); } catch (_openErr) {}
        try { uni.removeStorageSync(DESK_PET_PENDING_RECORD_QUERY_KEY); } catch (_queryErr) {}
        try { uni.removeStorageSync(DESK_PET_PENDING_RECORD_AUTO_CONFIRM_KEY); } catch (_confirmErr) {}
        try { uni.removeStorageSync(DESK_PET_PENDING_BILL_CAPTURE_KEY); } catch (_captureErr) {}
    }

    private static emitDeskPetBillCapture(): void {
        try {
            uni.$emit(DESK_PET_BILL_CAPTURE_EVENT, null);
        } catch (err) {
            console.error("Failed to emit desk pet bill capture event: " + JSON.stringify(err));
            try {
                promptAction.showToast({
                    message: "账单识别启动失败",
                    duration: 2000
                });
            } catch (_toastErr) {}
        }
    }

    private static emitDeskPetRecordOpen(): void {
        try {
            uni.$emit(DESK_PET_RECORD_OPEN_EVENT, null);
        } catch (err) {
            console.error("Failed to emit desk pet record open event: " + JSON.stringify(err));
        }
    }

    private static async bringFinanceAppToFront(markCapture: boolean = true): Promise<void> {
        await ZemoAsk.restoreFinanceAppByFloatingBall(markCapture);
    }

    static moveFinanceAppToBackground(): void {
        try {
            void UTSHarmony.getUIAbilityContext().moveAbilityToBackground();
        } catch (err) {
            console.warn("Failed to move finance app to background: " + JSON.stringify(err));
        }
    }

    private static async startDeepLink(link: string): Promise<void> {
        await ZemoAsk.prepareFloatWindowForUserLaunch();
        try {
            const context = UTSHarmony.getUIAbilityContext();
            await context.openLink(link, { appLinkingOnly: false });
        } catch (err) {
            const bErr = err as BusinessError;
            console.error("Failed to open finance deep link: " + JSON.stringify(err.message));
            if (bErr.code == 201) {
                console.error("System rejected openLink while app is backgrounded; pending desk pet action remains stored.1");
            }
            try {
                promptAction.showToast({
                    message: bErr.code == 201
                        ? "系统拒绝后台拉起，请手动打开记账应用继续录入"
                        : "请先打开记账应用",
                    duration: 3000
                });
            } catch (_toastErr) {}
        } finally {
            ZemoAsk.restoreFloatWindowNonFocusableLater();
        }
    }

    private static async prepareFloatWindowForUserLaunch(): Promise<void> {
        const win = ZemoAsk.subWindow;
        if (win == null) return;
        try {
            await win.setWindowFocusable(true);
        } catch (focusErr) {
            console.warn("Failed to make desk pet float window focusable before openLink: " + JSON.stringify(focusErr));
        }
    }

    private static restoreFloatWindowNonFocusableLater(): void {
        setTimeout(() => {
            const win = ZemoAsk.subWindow;
            if (win == null) return;
            win.setWindowFocusable(false).catch((focusErr: BusinessError) => {
                console.warn("Failed to restore desk pet float window non-focusable: " + JSON.stringify(focusErr));
            });
        }, 800);
    }

    private static async restoreFinanceAppByFloatingBall(markCapture: boolean = true): Promise<void> {
        if (markCapture) ZemoAsk.markPendingDeskPetBillCapture();
        try {
            const controller = await ZemoAsk.ensureFloatingBallController();
            await ZemoAsk.startOrUpdateFinanceFloatingBall(controller);
            ZemoAsk.registerFloatingBallClick(controller);
            try {
                await controller.restoreMainWindow(ZemoAsk.financeRestoreWant());
                await ZemoAsk.stopFloatingBallQuietly();
                console.info("Floating ball auto restore succeeded.");
                return;
            } catch (restoreErr) {
                const bErr = restoreErr as BusinessError;
                console.warn("Floating ball auto restore failed, waiting for click. Cause: " + JSON.stringify(bErr));
            }
            try {
                promptAction.showToast({
                    message: "点击闪控球继续记账",
                    duration: 3000
                });
            } catch (_toastErr) {}
        } catch (err) {
            const bErr = err as BusinessError;
            console.error("Failed to show floating ball for finance restore: " + JSON.stringify(bErr));
            await ZemoAsk.startDeepLinkFallback();
        }
    }

    private static async startDeepLinkFallback(): Promise<void> {
        const ts = Date.now().toString();
        await ZemoAsk.startDeepLink(`zemo://finance/bill-capture?ts=${ts}`);
    }

    private static async ensureFloatingBallController(): Promise<floatingBall.FloatingBallController> {
        if (!floatingBall.isFloatingBallEnabled()) {
            throw new Error("当前设备不支持闪控球");
        }
        if (ZemoAsk.floatingBallController != null) {
            return ZemoAsk.floatingBallController;
        }
        const context = UTSHarmony.getUIAbilityContext() as common.UIAbilityContext;
        const config: floatingBall.FloatingBallConfiguration = { context: context };
        const controller = await floatingBall.create(config);
        ZemoAsk.floatingBallController = controller;
        return controller;
    }

    private static async startOrUpdateFinanceFloatingBall(controller: floatingBall.FloatingBallController): Promise<void> {
        const params: floatingBall.FloatingBallParams = {
            template: floatingBall.FloatingBallTemplate.EMPHATIC,
            title: "泽墨账",
            content: "点击确认识别账单",
            backgroundColor: "#FF0D5B63"
        };
        try {
            await controller.updateFloatingBall(params);
        } catch (_updateErr) {
            await controller.startFloatingBall(params);
        }
    }

    private static registerFloatingBallClick(controller: floatingBall.FloatingBallController): void {
        if (ZemoAsk.floatingBallClickRegistered) return;
        ZemoAsk.floatingBallClickRegistered = true;
        controller.on("click", () => {
            void (async () => {
                try {
                    await controller.restoreMainWindow(ZemoAsk.financeRestoreWant());
                    await ZemoAsk.stopFloatingBallQuietly();
                } catch (err) {
                    const bErr = err as BusinessError;
                    console.error("Floating ball click restore failed: " + JSON.stringify(bErr));
                }
            })();
        });
    }

    private static async stopFloatingBallQuietly(): Promise<void> {
        const controller = ZemoAsk.floatingBallController;
        if (controller == null) return;
        try {
            controller.off("click");
        } catch (_offErr) {}
        ZemoAsk.floatingBallClickRegistered = false;
        try {
            await controller.stopFloatingBall();
        } catch (_stopErr) {}
        ZemoAsk.floatingBallController = null;
    }

    private static financeRestoreWant(): Want {
        return {
            bundleName: FINANCE_BUNDLE_NAME,
            abilityName: FINANCE_ENTRY_ABILITY,
            parameters: {
                source: "desk_pet_bill_capture",
                ts: Date.now().toString()
            }
        };
    }

    private static executeQuickAction(action: DeskPetQuickAction): void {
        const link = action.link.trim();
        if (link == "") {
            try {
                promptAction.showToast({ message: "快捷功能链接为空", duration: 1800 });
            } catch (_toastErr) {}
            return;
        }
        if (ZemoAsk.isFinanceRecordLink(link)) {
            ZemoAsk.markPendingRecordOpen(link);
            ZemoAsk.emitDeskPetRecordOpen();
        }
        void ZemoAsk.startDeepLink(link);
    }

    private static isFinanceRecordLink(link: string): boolean {
        return link == FINANCE_RECORD_DEEP_LINK || link.startsWith(FINANCE_RECORD_DEEP_LINK + "?") || link.startsWith(FINANCE_RECORD_DEEP_LINK + "/");
    }

    private static defaultQuickActions(): DeskPetQuickAction[] {
        return [new DeskPetQuickAction("finance_record", "记一笔", FINANCE_RECORD_DEEP_LINK)];
    }

    private static normalizeQuickAction(value: DeskPetQuickAction, index: number): DeskPetQuickAction | null {
        if (value == null) return null;
        const title = value.title != null ? value.title.toString().trim() : "";
        const link = value.link != null ? value.link.toString().trim() : "";
        if (title == "" || link == "") return null;
        let id = value.id != null ? value.id.toString().trim() : "";
        if (id == "") id = "quick_" + index.toString();
        return new DeskPetQuickAction(id, title, link);
    }

    private static readStoredText(key: string): string {
        try {
            const value = uni.getStorageSync(key);
            return value != null ? value.toString() : "";
        } catch (_e) {
            return "";
        }
    }

    private static normalizePipMetric(mode: string): string {
        const m = (mode ?? "").toString().trim();
        if (m == "cash" || m == "investment" || m == "credit" || m == "heavy" || m == "today_expense" || m == "calendar_month") return m;
        return "net_worth";
    }

    private static mapPipMetric(mode: string, item: string, root: string): DeskPetAssetMetric {
        const title = ZemoAsk.readJsonText(item, "title", ZemoAsk.defaultPipMetric(mode).title);
        const amount = ZemoAsk.readJsonNumber(item, "amount", 0.0);
        const badge = ZemoAsk.readJsonText(item, "badge", "CNY");
        const desc = ZemoAsk.readJsonText(item, "desc", "");
        const tail = ZemoAsk.readJsonText(item, "tail", "ASSET");
        const holder = ZemoAsk.readJsonText(item, "holder", "ASSET CARD");
        const updatedAt = ZemoAsk.readJsonNumber(root, "updatedAt", 0.0);
        return new DeskPetAssetMetric(mode, title, amount, badge, desc, tail, holder, updatedAt);
    }

    private static defaultPipMetric(mode: string): DeskPetAssetMetric {
        if (mode == "cash") return new DeskPetAssetMetric(mode, "资金账户", 0.0, "0 个", "可用资金合计", "CASH", "ASSET CARD", 0);
        if (mode == "investment") return new DeskPetAssetMetric(mode, "理财资产", 0.0, "0 项", "持仓市值合计", "FUND", "ASSET CARD", 0);
        if (mode == "credit") return new DeskPetAssetMetric(mode, "信贷负债", 0.0, "0 张", "当前欠款合计", "DEBT", "ASSET CARD", 0);
        if (mode == "heavy") return new DeskPetAssetMetric(mode, "重资产", 0.0, "0 项", "固定资产估值", "HOME", "ASSET CARD", 0);
        if (mode == "today_expense") return new DeskPetAssetMetric(mode, "今日消耗", 0.0, "TODAY", "全部支出种类合计", "TODAY", "DAILY COST", 0);
        if (mode == "calendar_month") return new DeskPetAssetMetric(mode, "日历金额", 0.0, "CAL", "每日收支金额", "CAL", "AMOUNT CALENDAR", 0);
        return new DeskPetAssetMetric();
    }

    private static readPipMetricJson(raw: string, mode: string): string {
        const metricsIndex: number = raw.indexOf("\"metrics\"");
        if (metricsIndex < 0) return "";
        const keyIndex: number = raw.indexOf("\"" + mode + "\"", metricsIndex);
        if (keyIndex < 0) return "";
        const objectStart: number = raw.indexOf("{", keyIndex);
        if (objectStart < 0) return "";
        return ZemoAsk.readJsonObjectAt(raw, objectStart);
    }

    private static readJsonObjectAt(raw: string, start: number): string {
        let depth: number = 0;
        let inString: boolean = false;
        let escaped: boolean = false;
        for (let i = start; i < raw.length; i++) {
            const ch: string = raw.charAt(i);
            if (inString) {
                if (escaped) {
                    escaped = false;
                } else if (ch == "\\") {
                    escaped = true;
                } else if (ch == "\"") {
                    inString = false;
                }
            } else if (ch == "\"") {
                inString = true;
            } else if (ch == "{") {
                depth++;
            } else if (ch == "}") {
                depth--;
                if (depth == 0) return raw.substring(start, i + 1);
            }
        }
        return "";
    }

    private static findJsonValueStart(raw: string, key: string): number {
        const keyIndex: number = raw.indexOf("\"" + key + "\"");
        if (keyIndex < 0) return -1;
        const colonIndex: number = raw.indexOf(":", keyIndex);
        if (colonIndex < 0) return -1;
        let start: number = colonIndex + 1;
        while (start < raw.length && raw.charAt(start) == " ") start++;
        return start;
    }

    private static readJsonText(raw: string, key: string, fallback: string): string {
        const start: number = ZemoAsk.findJsonValueStart(raw, key);
        if (start < 0) return fallback;
        let text: string = "";
        if (raw.charAt(start) == "\"") {
            let escaped: boolean = false;
            for (let i = start + 1; i < raw.length; i++) {
                const ch: string = raw.charAt(i);
                if (escaped) {
                    text += ch;
                    escaped = false;
                } else if (ch == "\\") {
                    escaped = true;
                } else if (ch == "\"") {
                    return text == "" ? fallback : text;
                } else {
                    text += ch;
                }
            }
            return fallback;
        }
        let end: number = start;
        while (end < raw.length && raw.charAt(end) != "," && raw.charAt(end) != "}") end++;
        text = raw.substring(start, end).trim();
        return text == "" ? fallback : text;
    }

    private static readJsonNumber(raw: string, key: string, fallback: number): number {
        const start: number = ZemoAsk.findJsonValueStart(raw, key);
        if (start < 0) return fallback;
        let end: number = start;
        while (end < raw.length && raw.charAt(end) != "," && raw.charAt(end) != "}") end++;
        const parsed: number = parseFloat(raw.substring(start, end).trim());
        return isNaN(parsed) ? fallback : parsed;
    }

    private static getQuickActionRect(index: number, count: number): DeskPetQuickActionRect {
        const totalHeight = count * DESK_PET_BUBBLE_SIZE + Math.max(0, count - 1) * DESK_PET_BUBBLE_GAP;
        const y = Math.round((DESK_PET_BUBBLE_PANEL_HEIGHT - totalHeight) / 2 + index * (DESK_PET_BUBBLE_SIZE + DESK_PET_BUBBLE_GAP));
        const x = ZemoAsk.dockSide == "left"
            ? DESK_PET_BUBBLE_SIDE_GAP
            : DESK_PET_BUBBLE_PANEL_WIDTH - DESK_PET_BUBBLE_SIDE_GAP - DESK_PET_BUBBLE_SIZE;
        return new DeskPetQuickActionRect(x, y, DESK_PET_BUBBLE_SIZE, DESK_PET_BUBBLE_SIZE);
    }

    private static clamp(value: number, min: number, max: number): number {
        if (value < min) return min;
        if (value > max) return max;
        return value;
    }

    private static async resizeWindowFrame(width: number, height: number, x: number, y: number): Promise<void> {
        const d = display.getDefaultDisplaySync();
        const density = d.densityPixels;
        ZemoAsk.winWidth = width;
        ZemoAsk.winHeight = height;
        ZemoAsk.currentX = x;
        ZemoAsk.currentY = y;
        if (ZemoAsk.pipController != null) {
            try {
                ZemoAsk.pipController.updateContentSize(Math.round(width * density), Math.round(height * density));
            } catch (resizeErr) {
                console.warn("Failed to resize desk pet PiP frame: " + JSON.stringify(resizeErr));
            }
            return;
        }
        const win = ZemoAsk.subWindow;
        if (win == null) return;
        try {
            await win.resize(Math.round(width * density), Math.round(height * density));
        } catch (resizeErr) {
            console.warn("Failed to resize desk pet frame: " + JSON.stringify(resizeErr));
        }
        try {
            await win.moveWindowTo(Math.round(x * density), Math.round(y * density));
        } catch (moveErr) {
            console.warn("Failed to move desk pet frame: " + JSON.stringify(moveErr));
        }
    }

    private static async resizeToSideLine(): Promise<void> {
        const d = display.getDefaultDisplaySync();
        const density = d.densityPixels;
        const screenWidth = d.width / density;
        const screenHeight = d.height / density;
        const centerX = ZemoAsk.currentX + ZemoAsk.winWidth / 2;
        const centerY = ZemoAsk.currentY + ZemoAsk.winHeight / 2;
        ZemoAsk.dockSide = centerX < screenWidth / 2 ? "left" : "right";
        const x = ZemoAsk.dockSide == "left" ? 0 : Math.max(0, screenWidth - DESK_PET_DOCK_LINE_WIDTH);
        const y = ZemoAsk.clamp(centerY - DESK_PET_DOCK_LINE_HEIGHT / 2, 0, Math.max(0, screenHeight - DESK_PET_DOCK_LINE_HEIGHT));
        await ZemoAsk.resizeWindowFrame(DESK_PET_DOCK_LINE_WIDTH, DESK_PET_DOCK_LINE_HEIGHT, x, y);
    }

    private static async resizeToQuickPanel(): Promise<void> {
        const d = display.getDefaultDisplaySync();
        const density = d.densityPixels;
        const screenWidth = d.width / density;
        const screenHeight = d.height / density;
        const centerY = ZemoAsk.currentY + ZemoAsk.winHeight / 2;
        const x = ZemoAsk.dockSide == "left" ? 0 : Math.max(0, screenWidth - DESK_PET_BUBBLE_PANEL_WIDTH);
        const y = ZemoAsk.clamp(centerY - DESK_PET_BUBBLE_PANEL_HEIGHT / 2, 0, Math.max(0, screenHeight - DESK_PET_BUBBLE_PANEL_HEIGHT));
        await ZemoAsk.resizeWindowFrame(DESK_PET_BUBBLE_PANEL_WIDTH, DESK_PET_BUBBLE_PANEL_HEIGHT, x, y);
    }

    static triggerAutoDockSmooth(): void {
        const win = ZemoAsk.subWindow;
        if (win == null) return;
        try {
            const d = display.getDefaultDisplaySync();
            const density = d.densityPixels;
            
            const lsw = d.width / density;
            const maxX = Math.max(0, lsw - ZemoAsk.winWidth);
            
            const startX = ZemoAsk.currentX;
            const targetX = (startX + ZemoAsk.winWidth / 2 < lsw / 2) ? 0 : maxX;
            
            let step = 0;
            const steps = 12; // 200ms easing animation
            const intervalId = setInterval(() => {
                step++;
                const t = step / steps;
                // easeOutQuad curve
                const easedT = t * (2 - t);
                const nextX = Math.round(startX + (targetX - startX) * easedT);
                ZemoAsk.moveWindow(nextX, ZemoAsk.currentY);
                if (step >= steps) {
                    clearInterval(intervalId);
                }
            }, 16);
        } catch (err) {
            console.error("Auto docking error: " + JSON.stringify(err));
        }
    }
}
