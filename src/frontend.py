"""Gradio frontend for TraffiCount."""
from __future__ import annotations

from typing import List

import gradio as gr

from controller import get_counts, load_frame_preview, start_job, stop_job
from draw_overlay import OverlayManager, overlays_table_payload, render_overlay_preview


def _render_preview(base_image, overlays_payload, pending_points=None):
    if base_image is None:
        return None
    try:
        return render_overlay_preview(
            base_image,
            overlays_payload or [],
            pending_points or [],
        )
    except ValueError:
        return base_image


def handle_point_selection(evt: gr.SelectData, overlays_payload, pending_points, current_image):
    """Collect two points for a candidate overlay line."""
    if evt is None or evt.index is None:
        return pending_points, current_image, gr.update(), "Click on the image to set points."

    x, y = map(int, evt.index)
    points = list(pending_points or [])
    points.append((x, y))

    message = f"Point {len(points)} selected at ({x}, {y})."
    updated_image = _render_preview(current_image, overlays_payload, points)
    return points[:2], updated_image, gr.update(), message


def save_line(line_name: str, overlays_payload, pending_points, current_image):
    """Persist a line drawn by the user."""
    if not line_name:
        return overlays_payload, pending_points, current_image, gr.update(), "Enter a line name first."
    if not pending_points or len(pending_points) < 2:
        return overlays_payload, pending_points, current_image, gr.update(), "Select two points before saving."

    manager = OverlayManager.from_payload(overlays_payload or [])
    try:
        manager.add_line(line_name, pending_points[:2])
    except ValueError as exc:
        return overlays_payload, pending_points, current_image, gr.update(), str(exc)

    overlays = manager.to_payload()
    table_rows = overlays_table_payload(overlays)
    updated_image = _render_preview(current_image, overlays, [])
    return overlays, [], updated_image, table_rows, f"Saved line '{line_name}'."


def reset_pending_points(overlays_payload, current_image):
    """Clear in-progress overlay points."""
    updated_image = _render_preview(current_image, overlays_payload, [])
    return [], updated_image, gr.update(value=""), "Pending points cleared."


def clear_overlay_lines(current_image):
    """Remove all saved overlay lines."""
    overlays = []
    table_rows: List[List[str]] = []
    updated_image = _render_preview(current_image, overlays, [])
    return overlays, updated_image, table_rows, [], gr.update(value=""), "All overlay lines cleared."


def reset_after_completion():
    """Reset overlay state and status after a video completes."""
    return [], [], [], gr.update(value=""), "Ready for next video."


def build_frontend():
    with gr.Blocks(title="TraffiCount") as demo:
        gr.Markdown("# TraffiCount")

        overlays_state = gr.State([])
        pending_points_state = gr.State([])

        video_input = gr.File(label="Video file", file_count="single", file_types=[".mp4"])
        detection_view = gr.Image(label="Detection View", image_mode="RGB", type="numpy")
        overlay_table = gr.Dataframe(headers=["Label", "Points"], interactive=False, label="Overlay lines")
        detected_label = gr.Textbox(label="Vehicles Detected", interactive=False, value="0")
        identified_label = gr.Textbox(label="Vehicles Identified", interactive=False, value="0")
        status = gr.Markdown("Ready.")

        with gr.Row():
            start_btn = gr.Button("Start Detection", variant="primary")
            stop_btn = gr.Button("Stop Detection", variant="secondary")
            frame_btn = gr.Button("Load Preview Frame")

        with gr.Row():
            line_name = gr.Textbox(label="Line Name", placeholder="Example: Entry Line A")
            save_line_btn = gr.Button("Save Line")
            reset_points_btn = gr.Button("Reset Points")
            clear_lines_btn = gr.Button("Clear All Lines")

        def refresh_counts():
            return get_counts()

        refresh_btn = gr.Button("Refresh Counts", visible=False)
        refresh_btn.click(refresh_counts, outputs=[detected_label, identified_label])

        start_chain = start_btn.click(
            start_job,
            inputs=[video_input, overlays_state],
            outputs=[detection_view, status, detected_label, identified_label],
        )
        start_chain.then(
            reset_after_completion,
            outputs=[overlays_state, pending_points_state, overlay_table, line_name, status],
        )
        stop_chain = stop_btn.click(stop_job, None, [status, detected_label, identified_label])
        stop_chain.then(
            reset_after_completion,
            outputs=[overlays_state, pending_points_state, overlay_table, line_name, status],
        )

        frame_btn.click(
            load_frame_preview,
            inputs=[video_input],
            outputs=[detection_view, status],
        )

        video_input.change(
            load_frame_preview,
            inputs=[video_input],
            outputs=[detection_view, status],
        )

        detection_view.select(
            handle_point_selection,
            inputs=[overlays_state, pending_points_state, detection_view],
            outputs=[pending_points_state, detection_view, overlay_table, status],
        )

        save_line_btn.click(
            save_line,
            inputs=[line_name, overlays_state, pending_points_state, detection_view],
            outputs=[overlays_state, pending_points_state, detection_view, overlay_table, status],
        )

        reset_points_btn.click(
            reset_pending_points,
            inputs=[overlays_state, detection_view],
            outputs=[pending_points_state, detection_view, line_name, status],
        )

        clear_lines_btn.click(
            clear_overlay_lines,
            inputs=[detection_view],
            outputs=[overlays_state, detection_view, overlay_table, pending_points_state, line_name, status],
        )

        def auto_refresh():
            while True:
                yield get_counts()

        try:
            gr.Timer(2.0, fn=auto_refresh, outputs=[detected_label, identified_label])
        except Exception:
            pass

    return demo


def launch():
    app = build_frontend()
    app.queue()
    app.launch(share=False, show_error=True)


if __name__ == "__main__":
    launch()
