package uts.sdk.modules.zemoRpa

import android.accessibilityservice.AccessibilityService
import android.accessibilityservice.GestureDescription
import android.graphics.Path
import android.graphics.Rect
import android.util.Log
import android.view.accessibility.AccessibilityEvent
import android.view.accessibility.AccessibilityNodeInfo

/**
 * Top-level listener interface for UTS bridge.
 * Kept outside the class so UTS can import it directly as
 * `import { AccessibilityEventListener } from "uts.sdk.modules.zemoRpa"`.
 */
interface AccessibilityEventListener {
    fun onServiceConnected()
    fun onAccessibilityEvent(
        eventType: Int,
        packageName: String,
        className: String,
        source: AccessibilityNodeInfo?
    )
    fun onInterrupt()
}

/**
 * Native Kotlin AccessibilityService.
 *
 * This class MUST live in src/ so it gets compiled into the main APK
 * (package uts.sdk.modules.zemoRpa) where the Android system class-loader
 * can instantiate it.
 *
 * All business logic stays in the UTS layer; this class merely bridges
 * system callbacks → UTS via a static listener interface.
 */
class GlobalInputListener : AccessibilityService() {

    companion object {
        private const val TAG = "GlobalInputListener"

        @JvmStatic
        var instance: GlobalInputListener? = null
            private set

        /** UTS-side listener – set from UTS via registerEventListener() */
        @JvmStatic
        var eventListener: AccessibilityEventListener? = null

        @JvmStatic
        fun registerEventListener(listener: AccessibilityEventListener?) {
            eventListener = listener
        }

        /**
         * Static entry point – called from UTS to dispatch a tap gesture.
         */
        @JvmStatic
        fun dispatchTap(x: Float, y: Float): Boolean {
            val svc = instance ?: run {
                Log.w(TAG, "dispatchTap failed: Service not connected")
                return false
            }
            return svc.performTap(x, y)
        }
    }

    // ──────────────── Service Lifecycle ────────────────

    override fun onServiceConnected() {
        super.onServiceConnected()
        instance = this
        Log.i(TAG, "GlobalInputListener connected")
        eventListener?.onServiceConnected()
    }

    override fun onAccessibilityEvent(event: AccessibilityEvent) {
        val eventType = event.eventType
        val pkg = event.packageName?.toString() ?: ""
        val cls = event.className?.toString() ?: ""
        val source = event.source  // may be null
        eventListener?.onAccessibilityEvent(eventType, pkg, cls, source)
    }

    override fun onInterrupt() {
        Log.w(TAG, "GlobalInputListener interrupted")
        eventListener?.onInterrupt()
    }

    override fun onDestroy() {
        super.onDestroy()
        if (instance == this) instance = null
    }

    // ──────────────── Gesture helpers ────────────────

    fun performTap(x: Float, y: Float): Boolean {
        Log.d(TAG, "dispatchTap($x, $y)")
        val path = Path().apply { moveTo(x, y) }
        val stroke = GestureDescription.StrokeDescription(path, 0, 50)
        val gesture = GestureDescription.Builder().addStroke(stroke).build()
        return dispatchGesture(gesture, null, null)
    }

    fun performSwipe(x1: Float, y1: Float, x2: Float, y2: Float, durationMs: Long): Boolean {
        val path = Path().apply {
            moveTo(x1, y1)
            lineTo(x2, y2)
        }
        val stroke = GestureDescription.StrokeDescription(path, 0, durationMs)
        val gesture = GestureDescription.Builder().addStroke(stroke).build()
        return dispatchGesture(gesture, null, null)
    }

    /**
     * Convenience: find root node.
     */
    fun getRoot(): AccessibilityNodeInfo? = rootInActiveWindow

    /**
     * Find the deepest visible node at the given screen (x, y).
     */
    fun findNodeAt(x: Int, y: Int): AccessibilityNodeInfo? {
        val root = rootInActiveWindow ?: return null
        val found = findNodeRecursive(root, x, y)
        if (found !== root) root.recycle()
        return found
    }

    private fun findNodeRecursive(
        node: AccessibilityNodeInfo, x: Int, y: Int
    ): AccessibilityNodeInfo? {
        val rect = Rect()
        node.getBoundsInScreen(rect)
        if (!rect.contains(x, y)) return null

        for (i in node.childCount - 1 downTo 0) {
            val child = node.getChild(i) ?: continue
            val result = findNodeRecursive(child, x, y)
            if (result != null) {
                if (result !== child) child.recycle()
                return result
            }
            child.recycle()
        }

        return if (node.isVisibleToUser) node else null
    }
}
