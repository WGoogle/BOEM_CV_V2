"""
annotation.editor — Interactive Mask Annotation GUI
Reference ANNOTATION_GUIDE.txt for usage instructions.
"""
from __future__ import annotations
import copy
import logging
import time
from collections import deque
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

plt.rcParams["keymap.save"] = []

logger = logging.getLogger(__name__)

class AnnotationEditor:
    def __init__(self, records: list[dict], output_dir: Path, annotator: str = "anonymous", overlay_alpha: float = 0.45, max_undo: int = 50):
        self.records = records
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.annotator = annotator
        self.overlay_alpha = overlay_alpha
        self.max_undo = max_undo

        # State
        self.current_idx = 0
        self.brush_radius = 1
        self.min_brush = 0
        self.max_brush = 30
        self._reset_pending_until = 0.0
        self.show_overlay = True
        self.drawing = False
        self.erase_mode = False

        # Current image/mask
        self.image: np.ndarray | None = None
        self.mask: np.ndarray | None = None
        self.original_mask: np.ndarray | None = None

        # Undo/redo stacks
        self.undo_stack: deque[np.ndarray] = deque(maxlen=max_undo)
        self.redo_stack: deque[np.ndarray] = deque(maxlen=max_undo)

        # Track which patches have been modified
        self.modified_patches: set[str] = set()

        # Peek mode: hold Space to see raw image
        self._peeking = False

        # Outline mode: show mask as contour outlines instead of filled
        self.outline_mode = False

        # Mouse tracking for smooth strokes
        self._last_xy: tuple[int, int] | None = None

    def _load_patch(self, idx: int):
        rec = self.records[idx]

        # Load image (BGR to RGB)
        img = cv2.imread(rec["image_path"], cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Image not found: {rec['image_path']}")
        self.image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Check for existing corrected mask first
        patch_id = rec.get("patch_id", f"patch_{idx:04d}")
        corrected_path = self.output_dir / f"{patch_id}.png"
        if corrected_path.exists():
            mask = cv2.imread(str(corrected_path), cv2.IMREAD_GRAYSCALE)
            logger.info(f"  Loaded existing corrected mask for {patch_id}")
        else:
            mask = cv2.imread(rec["mask_path"], cv2.IMREAD_GRAYSCALE)

        if mask is None:
            raise FileNotFoundError(f"Mask not found: {rec['mask_path']}")
        self.mask = (mask > 127).astype(np.uint8)
        self.original_mask = self.mask.copy()

        # Reset undo/redo
        self.undo_stack.clear()
        self.redo_stack.clear()
        self._last_xy = None

    def _save_mask(self):
        rec = self.records[self.current_idx]
        patch_id = rec.get("patch_id", f"patch_{self.current_idx:04d}")
        out_path = self.output_dir / f"{patch_id}.png"
        cv2.imwrite(str(out_path), self.mask * 255)
        self.modified_patches.add(patch_id)
        logger.info(f"  Saved corrected mask → {out_path}")
        return out_path

    def _push_undo(self):
        self.undo_stack.append(self.mask.copy())
        self.redo_stack.clear()

    def _undo(self):
        if self.undo_stack:
            self.redo_stack.append(self.mask.copy())
            self.mask = self.undo_stack.pop()

    def _redo(self):
        if self.redo_stack:
            self.undo_stack.append(self.mask.copy())
            self.mask = self.redo_stack.pop()

    def _paint_at(self, x: int, y: int, erase: bool = False):
        h, w = self.mask.shape
        x, y = int(round(x)), int(round(y))
        if not (0 <= x < w and 0 <= y < h):
            return
        value = 0 if erase else 1
        cv2.circle(self.mask, (x, y), self.brush_radius, int(value), -1)

    def _paint_line(self, x0: int, y0: int, x1: int, y1: int, erase: bool = False):
        h, w = self.mask.shape
        value = 0 if erase else 1
        cv2.line(self.mask, (int(x0), int(y0)), (int(x1), int(y1)),
                 int(value), max(1, self.brush_radius * 2))

    def _render_composite(self):
        composite = self.image.copy()

        # Peek mode: show raw image, no overlay
        if self._peeking or not self.show_overlay or self.mask is None:
            return composite

        mask_bool = self.mask > 0

        if self.outline_mode:
            # Outline mode: draw contour edges only (much easier to see underlying image)
            mask_u8 = (self.mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            # Original contours in green, new in cyan
            orig_u8 = (self.original_mask * 255).astype(np.uint8)
            orig_contours, _ = cv2.findContours(orig_u8, cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(composite, orig_contours, -1, (0, 255, 0), 2)

            # New additions: contours present in current but not original
            new_mask = (mask_bool & ~(self.original_mask > 0)).astype(np.uint8) * 255
            new_contours, _ = cv2.findContours(new_mask, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(composite, new_contours, -1, (0, 255, 255), 2)

            # Removed: contours in original but not current
            rem_mask = ((self.original_mask > 0) & ~mask_bool).astype(np.uint8) * 255
            rem_contours, _ = cv2.findContours(rem_mask, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(composite, rem_contours, -1, (255, 0, 0), 2)
        else:
            # Filled overlay mode (original behavior)
            overlay_color = np.zeros_like(composite)
            # Original pixels in green
            orig_bool = (self.original_mask > 0) & mask_bool
            overlay_color[orig_bool] = [0, 255, 0]
            # Newly added pixels in cyan
            new_bool = mask_bool & ~(self.original_mask > 0)
            overlay_color[new_bool] = [0, 255, 255]
            # Removed pixels in red
            removed_bool = (self.original_mask > 0) & ~mask_bool
            overlay_color[removed_bool] = [255, 0, 0]

            alpha = self.overlay_alpha
            composite = np.where(
                (overlay_color > 0).any(axis=2, keepdims=True),
                ((1 - alpha) * composite + alpha * overlay_color).astype(np.uint8),
                composite,
            )
        return composite

    # Main
    def launch(self, start_idx: int = 0):
        self.current_idx = start_idx
        self._load_patch(self.current_idx)

        # Create figure
        self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 10))
        plt.subplots_adjust(bottom=0.18, top=0.93)

        # Initial display
        self._img_display = self.ax.imshow(self._render_composite())
        self._update_title()

        # Brush cursor (circle)
        self._cursor_circle = plt.Circle(
            (0, 0), self.brush_radius, fill=False,
            edgecolor="yellow", linewidth=1, visible=False,
        )
        self.ax.add_patch(self._cursor_circle)

        # Controls
        # Brush size slider
        ax_brush = plt.axes([0.25, 0.08, 0.50, 0.03])
        self._brush_slider = Slider(
            ax_brush, "Brush", self.min_brush, self.max_brush,
            valinit=self.brush_radius, valstep=1,
        )
        self._brush_slider.on_changed(self._on_brush_slider)

        # Buttons
        ax_save = plt.axes([0.05, 0.02, 0.10, 0.04])
        ax_prev = plt.axes([0.20, 0.02, 0.10, 0.04])
        ax_next = plt.axes([0.35, 0.02, 0.10, 0.04])
        ax_undo = plt.axes([0.50, 0.02, 0.10, 0.04])
        ax_toggle = plt.axes([0.65, 0.02, 0.12, 0.04])
        ax_reset = plt.axes([0.82, 0.02, 0.12, 0.04])

        self._btn_save = Button(ax_save, "Save (S)")
        self._btn_prev = Button(ax_prev, "< Prev (P)")
        self._btn_next = Button(ax_next, "Next (N) >")
        self._btn_undo = Button(ax_undo, "Undo (Z)")
        self._btn_toggle = Button(ax_toggle, "Toggle (O)")
        self._btn_reset = Button(ax_reset, "Reset (R)")

        def _click_and_refocus(func):
            def wrapper(_event):
                func()
                self.fig.canvas.manager.window.focus_force()
                self.ax.figure.canvas.draw_idle()
            return wrapper

        self._btn_save.on_clicked(_click_and_refocus(self._do_save))
        self._btn_prev.on_clicked(_click_and_refocus(self._go_prev))
        self._btn_next.on_clicked(_click_and_refocus(self._go_next))
        self._btn_undo.on_clicked(_click_and_refocus(lambda: (self._undo(), self._refresh())))
        self._btn_toggle.on_clicked(_click_and_refocus(self._toggle_overlay))
        self._btn_reset.on_clicked(_click_and_refocus(self._reset_mask))

        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("key_release_event", self._on_key_release)
        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)

        logger.info("Annotation editor launched. Controls:")
        logger.info("  Left-click: paint nodules | Right-click: erase")
        logger.info("  +/- or scroll: zoom in/out | Arrow keys: pan | S: save | N/P: next/prev")
        logger.info("  SPACE (hold): peek at raw image underneath overlay")
        logger.info("  C: toggle outline mode (contours only, no fill)")
        logger.info("  O: toggle overlay | [/]: opacity | Ctrl+Z: undo | R: reset")
        logger.info("  Brush size: use the slider at the bottom")
        plt.show()

    def _on_press(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        self._push_undo()
        self.drawing = True
        self.erase_mode = (event.button == 3)  # right-click = erase
        self._paint_at(event.xdata, event.ydata, erase=self.erase_mode)
        self._last_xy = (event.xdata, event.ydata)
        self._refresh()

    def _on_release(self, event):
        self.drawing = False
        self._last_xy = None

    def _on_motion(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        self._cursor_circle.center = (event.xdata, event.ydata)
        self._cursor_circle.radius = max(0.5, self.brush_radius)
        self._cursor_circle.set_visible(True)
        color = "red" if self.erase_mode and self.drawing else "yellow"
        self._cursor_circle.set_edgecolor(color)

        if self.drawing:
            if self._last_xy is not None:
                self._paint_line(
                    self._last_xy[0], self._last_xy[1],
                    event.xdata, event.ydata,
                    erase=self.erase_mode,
                )
            else:
                self._paint_at(event.xdata, event.ydata, erase=self.erase_mode)
            self._last_xy = (event.xdata, event.ydata)

        self._refresh()

    def _on_key(self, event):
        key = event.key

        if key == " ":
            # Hold space to peek at raw image
            self._peeking = True
            self._refresh()
        elif key == "s":
            self._do_save()
        elif key == "n":
            self._go_next()
        elif key == "p":
            self._go_prev()
        elif key == "o":
            self._toggle_overlay()
        elif key == "c":
            # Toggle outline/contour mode vs filled overlay
            self.outline_mode = not self.outline_mode
            self._refresh()
        elif key == "r":
            self._reset_mask()
        elif key in ("+", "="):
            anchor = (event.xdata, event.ydata) if event.inaxes is self.ax else None
            self._zoom(zoom_in=True, anchor=anchor)
        elif key in ("-", "_"):
            anchor = (event.xdata, event.ydata) if event.inaxes is self.ax else None
            self._zoom(zoom_in=False, anchor=anchor)
        elif key == "left":
            self._pan(-1, 0)
        elif key == "right":
            self._pan(1, 0)
        elif key == "up":
            self._pan(0, -1)
        elif key == "down":
            self._pan(0, 1)
        elif key == "[":
            self.overlay_alpha = max(0.1, self.overlay_alpha - 0.05)
            self._refresh()
        elif key == "]":
            self.overlay_alpha = min(0.9, self.overlay_alpha + 0.05)
            self._refresh()
        elif key in ("z", "ctrl+z"):
            self._undo()
            self._refresh()
        elif key in ("y", "ctrl+y"):
            self._redo()
            self._refresh()

    def _on_key_release(self, event):
        if event.key == " ":
            self._peeking = False
            self._refresh()

    def _on_scroll(self, event):
        """Mouse-wheel zoom, anchored at the cursor when over the patch."""
        if event.button not in ("up", "down"):
            return
        zoom_in = event.button == "up"
        if event.inaxes is self.ax and event.xdata is not None:
            anchor = (event.xdata, event.ydata)
        else:
            anchor = None
        self._zoom(zoom_in=zoom_in, anchor=anchor)

    def _clamp_view(self):
        h, w = self.image.shape[:2]
        xlim = list(self.ax.get_xlim())
        ylim = list(self.ax.get_ylim())

        # Current view size
        vw = xlim[1] - xlim[0]
        vh = ylim[0] - ylim[1]  # y is inverted (ylim[0] > ylim[1])

        # If view is >= full image, just reset to full view (no panning needed)
        if vw >= w:
            xlim = [-0.5, w - 0.5]
        else:
            if xlim[0] < -0.5:
                xlim[0] = -0.5
                xlim[1] = xlim[0] + vw
            if xlim[1] > w - 0.5:
                xlim[1] = w - 0.5
                xlim[0] = xlim[1] - vw

        if vh >= h:
            ylim = [h - 0.5, -0.5]
        else:
            if ylim[1] < -0.5:
                ylim[1] = -0.5
                ylim[0] = ylim[1] + vh
            if ylim[0] > h - 0.5:
                ylim[0] = h - 0.5
                ylim[1] = ylim[0] - vh

        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)

    def _zoom(self, zoom_in: bool = True, anchor=None):
        scale = 1 / 1.4 if zoom_in else 1.4
        h, w = self.image.shape[:2]

        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()

        new_width = min(max((cur_xlim[1] - cur_xlim[0]) * scale, 4), w)
        new_height = min(max((cur_ylim[0] - cur_ylim[1]) * scale, 4), h)

        if anchor is not None:
            ax_mouse, ay_mouse = anchor
            fx = (ax_mouse - cur_xlim[0]) / (cur_xlim[1] - cur_xlim[0])
            fy = (ay_mouse - cur_ylim[1]) / (cur_ylim[0] - cur_ylim[1])
            new_x0 = ax_mouse - fx * new_width
            new_y1 = ay_mouse - fy * new_height
            self.ax.set_xlim(new_x0, new_x0 + new_width)
            self.ax.set_ylim(new_y1 + new_height, new_y1)
        else:
            cx = (cur_xlim[0] + cur_xlim[1]) / 2
            cy = (cur_ylim[0] + cur_ylim[1]) / 2
            self.ax.set_xlim(cx - new_width / 2, cx + new_width / 2)
            self.ax.set_ylim(cy + new_height / 2, cy - new_height / 2)

        self._clamp_view()
        self.fig.canvas.draw_idle()

    def _pan(self, dx: int, dy: int):
        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()

        # Pan by 20% of current view size
        step_x = (cur_xlim[1] - cur_xlim[0]) * 0.2 * dx
        step_y = (cur_ylim[0] - cur_ylim[1]) * 0.2 * dy 

        self.ax.set_xlim(cur_xlim[0] + step_x, cur_xlim[1] + step_x)
        self.ax.set_ylim(cur_ylim[0] + step_y, cur_ylim[1] + step_y)
        self._clamp_view()
        self.fig.canvas.draw_idle()

    def _on_brush_slider(self, val):
        self.brush_radius = int(val)
        self._cursor_circle.radius = self.brush_radius
        self.fig.canvas.draw_idle()

    # Actions
    def _do_save(self):
        self._save_mask()
        self._update_title()
        self.fig.canvas.draw_idle()

    def _go_next(self):
        if self.current_idx < len(self.records) - 1:
            self._prompt_save_if_modified()
            self.current_idx += 1
            self._load_patch(self.current_idx)
            self._reset_view()
            self._refresh()

    def _go_prev(self):
        if self.current_idx > 0:
            self._prompt_save_if_modified()
            self.current_idx -= 1
            self._load_patch(self.current_idx)
            self._reset_view()
            self._refresh()

    def _toggle_overlay(self):
        self.show_overlay = not self.show_overlay
        self._refresh()

    def _reset_mask(self):
        """Reset mask to original proxy label (requires double-confirm within 2s)."""
        now = time.monotonic()
        if now > self._reset_pending_until:
            self._reset_pending_until = now + 2.0
            logger.info("  Reset requested — press R again within 2s to confirm.")
            self._update_title()
            self.fig.canvas.draw_idle()
            return
        self._reset_pending_until = 0.0
        self._push_undo()
        self.mask = self.original_mask.copy()
        logger.info("  Mask reset to original.")
        self._refresh()

    def _prompt_save_if_modified(self):
        """Auto-save if mask was modified."""
        if self.mask is not None and self.original_mask is not None:
            if not np.array_equal(self.mask, self.original_mask):
                self._save_mask()
                logger.info("  Auto-saved modified mask before navigation.")

    def _reset_view(self):
        if self.image is not None:
            h, w = self.image.shape[:2]
            self.ax.set_xlim(-0.5, w - 0.5)
            self.ax.set_ylim(h - 0.5, -0.5)
    def _refresh(self):
        self._img_display.set_data(self._render_composite())
        self._update_title()
        self.fig.canvas.draw_idle()
    def _count_corrected(self):
        count = 0
        for rec in self.records:
            pid = rec.get("patch_id", "")
            if pid in self.modified_patches:
                count += 1
            elif (self.output_dir / f"{pid}.png").exists():
                count += 1
        return count

    def _update_title(self):
        rec = self.records[self.current_idx]
        patch_id = rec.get("patch_id", f"patch_{self.current_idx:04d}")
        modified = " [MODIFIED]" if not np.array_equal(self.mask, self.original_mask) else ""
        saved = " [SAVED]" if patch_id in self.modified_patches else ""
        n_pixels = int(self.mask.sum()) if self.mask is not None else 0
        done = self._count_corrected()
        total = len(self.records)
        remaining = total - done
        self.ax.set_title(
            f"[{self.current_idx + 1}/{total}] {patch_id}"
            f"  |  Done: {done}  Remaining: {remaining}"
            f"  |  mask={n_pixels}px"
            f"  |  alpha={self.overlay_alpha:.2f}{modified}{saved}",
            fontsize=10,
        )
        # Help text at figure level
        self.fig.texts.clear()
        mode = "OUTLINE" if self.outline_mode else "FILLED"
        peek = "  ** PEEKING **" if self._peeking else ""
        self.fig.text(
            0.5, 0.96,
            f"Green=proxy label  Cyan=added  Red=removed  |  "
            f"Left=paint  Right=erase  +/-=zoom  Arrows=pan  |  "
            f"SPACE=peek  C=outline  [{mode}]{peek}",
            ha="center", fontsize=8, color="gray",
        )
