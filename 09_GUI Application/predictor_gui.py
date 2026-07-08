import tkinter as tk
from tkinter import ttk, messagebox

from backend_predictor import AppendicitisPredictor
from gui.patient_frame import PatientFrame
from gui.vital_frame import VitalFrame
from gui.symptoms_frame import SymptomsFrame
from gui.laboratory_frame import LaboratoryFrame
from gui.ultrasound_frame import UltrasoundFrame
from gui.result_frame import ResultFrame
from gui.widgets import *


class AppendicitisGUI:
    def __init__(self):
        self.configure_window()
        self.configure_theme()
        self.initialize_backend()
        self.initialize_variables()
        self.create_scrollable_container()
        self.build_interface()
        self.bind_events()


    def configure_window(self):
        self.root = tk.Tk()
        self.root.title("Pediatric Appendicitis Diagnosis Prediction System")
        self.root.geometry("1450x900")
        self.root.minsize(1200, 800)


    def configure_theme(self):
        self.COLOR_BACKGROUND = "#F5F7FA"
        self.COLOR_PRIMARY = "#1565C0"
        self.COLOR_SECONDARY = "#1976D2"
        self.COLOR_SUCCESS = "#2E7D32"
        self.COLOR_WARNING = "#F57C00"
        self.COLOR_DANGER = "#C62828"
        self.COLOR_WHITE = "#FFFFFF"
        self.COLOR_FRAME = "#E3F2FD"
        self.COLOR_ENTRY = "#FFFFFF"

        self.FONT_TITLE = ("Segoe UI", 20, "bold")
        self.FONT_SECTION = ("Segoe UI", 12, "bold")
        self.FONT_NORMAL = ("Segoe UI", 10)
        self.FONT_RESULT = ("Segoe UI", 11, "bold")
        self.FONT_SMALL = ("Segoe UI", 9)

        self.root.configure(bg=self.COLOR_BACKGROUND)


    def initialize_backend(self):
        self.backend = AppendicitisPredictor()

    
    def initialize_variables(self):
        self.model_var = tk.StringVar()
        available_models = self.backend.get_available_models()
        if available_models:
            self.model_var.set(available_models[0])
        else:
            self.model_var.set("")

    
    def create_scrollable_container(self):
        self.canvas = tk.Canvas(self.root, bg=self.COLOR_BACKGROUND, highlightthickness=0)
        self.v_scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.v_scrollbar.set)
        self.v_scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.main_frame = tk.Frame(self.canvas, bg=self.COLOR_BACKGROUND)
        self.canvas_window = self.canvas.create_window((0, 0), window=self.main_frame, anchor="nw")
        self.main_frame.bind("<Configure>", self.update_scroll_region)
        self.canvas.bind("<Configure>", self.resize_canvas)


    def update_scroll_region(self, event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))


    def resize_canvas(self, event):
        self.canvas.itemconfigure(self.canvas_window, width=event.width)


    def bind_events(self):
        self.canvas.bind_all("<MouseWheel>", self.on_mousewheel)
        self.canvas.bind_all("<Button-4>", self.on_mousewheel_linux)
        self.canvas.bind_all("<Button-5>", self.on_mousewheel_linux)


    def on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")


    def on_mousewheel_linux(self, event):
        if event.num == 4:
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.canvas.yview_scroll(1, "units")


    def build_interface(self):
        self.create_title()
        self.patient_frame = PatientFrame(parent=self.main_frame, app=self)
        self.patient_frame.build()
        self.vital_frame = VitalFrame(parent=self.main_frame, app=self)
        self.vital_frame.build()
        self.symptoms_frame = SymptomsFrame(parent=self.main_frame, app=self)
        self.symptoms_frame.build()
        self.laboratory_frame = LaboratoryFrame(parent=self.main_frame, app=self)
        self.laboratory_frame.build()
        self.ultrasound_frame = UltrasoundFrame(parent=self.main_frame, app=self)
        self.ultrasound_frame.build()
        self.result_frame = ResultFrame(parent=self.main_frame, app=self)
        self.result_frame.build()
        self.create_buttons()


    def create_title(self):
        self.title_label = tk.Label(
            self.main_frame,
            text="Pediatric Appendicitis Diagnosis Prediction System",
            font=self.FONT_TITLE,
            bg=self.COLOR_BACKGROUND,
            fg=self.COLOR_PRIMARY
        )
        self.title_label.pack(pady=(10, 20))

    
    def create_buttons(self):
        self.button_frame = tk.Frame(self.main_frame, bg=self.COLOR_BACKGROUND)
        self.button_frame.pack(fill="x", pady=20, padx=15)
        self.predict_button = tk.Button(
            self.button_frame,
            text="Predict Diagnosis",
            command=self.predict,
            bg=self.COLOR_SUCCESS,
            fg="white",
            font=("Segoe UI", 11, "bold"),
            padx=20,
            pady=10
        )
        self.predict_button.pack(side="left", padx=5)
        self.clear_button = tk.Button(
            self.button_frame,
            text="Clear",
            command=self.clear_fields,
            bg=self.COLOR_WARNING,
            fg="white",
            font=("Segoe UI", 11, "bold"),
            padx=20,
            pady=10
        )
        self.clear_button.pack(side="left", padx=5)
        self.exit_button = tk.Button(
            self.button_frame,
            text="Exit",
            command=self.root.destroy,
            bg=self.COLOR_DANGER,
            fg="white",
            font=("Segoe UI", 11, "bold"),
            padx=20,
            pady=10
        )
        self.exit_button.pack(side="right", padx=5)

        
    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = AppendicitisGUI()
    app.run()