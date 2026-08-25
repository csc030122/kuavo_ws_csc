"""Isaac Sim panel for the fixed-dexterous-hand grab_box demo."""

from __future__ import annotations

import carb
import omni.ext
import omni.ui as ui

from .controller import DexhandBoxCarryController


class Extension(omni.ext.IExt):
    def on_startup(self, ext_id: str) -> None:
        self._ext_id = ext_id
        self._status_label = None
        self._controller = DexhandBoxCarryController(self._set_status)
        self._window = ui.Window("Kuavo Dexhand Box Carry Demo", width=410, height=285)
        self._window.frame.set_build_fn(self._build_ui)
        self._window.visible = True

    def on_shutdown(self) -> None:
        if self._controller is not None:
            self._controller.shutdown()
        self._controller = None
        self._window = None
        self._status_label = None

    def _build_ui(self) -> None:
        with ui.VStack(spacing=8, height=0):
            ui.Label(
                "Open lab.usd first. The robot approaches grab_box, grips it "
                "by physical palm contact with both fixed dexterous hands, "
                "lifts it 150 mm, backs 450 mm clear of the pickup stack, "
                "turns 180 degrees, then carries it to the "
                "rack2 level-3 shelf, releases it, retreats, and restores.",
                word_wrap=True,
                height=66,
            )
            with ui.HStack(height=32, spacing=6):
                ui.Button("Reset to Start", clicked_fn=lambda: self._call(self._controller.reset))
                ui.Button("Play From Start", clicked_fn=lambda: self._call(self._controller.replay))
            with ui.HStack(height=32, spacing=6):
                ui.Button("Pause", clicked_fn=lambda: self._call(self._controller.pause))
                ui.Button("Resume", clicked_fn=lambda: self._call(self._controller.play))
            ui.Separator(height=4)
            ui.Label("Status", height=20)
            self._status_label = ui.Label(
                "DETACHED - open lab.usd, then click Play From Start",
                word_wrap=True,
                height=55,
            )

    def _call(self, callback) -> None:
        try:
            callback()
        except Exception as exc:
            carb.log_error(f"Kuavo Dexhand Box Carry Demo UI: {exc}")
            self._set_status(f"ERROR - {exc}")

    def _set_status(self, text: str) -> None:
        if self._status_label is not None:
            self._status_label.text = text
