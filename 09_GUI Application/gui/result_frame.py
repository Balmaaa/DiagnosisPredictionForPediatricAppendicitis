import tkinter as tk
from gui.widgets import create_section_frame


class ResultFrame:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.frame = None


    def build(self):
        self.frame = create_section_frame(parent=self.parent, title="Prediction Result")
        self.frame.columnconfigure(0, weight=1)
        self.model_label = tk.Label(self.frame, text="Model : -", anchor="w", font=self.app.FONT_NORMAL, bg=self.app.COLOR_FRAME)
        self.model_label.grid(row=0, column=0, sticky="w", padx=10, pady=4)
        self.diagnosis_label = tk.Label(self.frame, text="Diagnosis : -", anchor="w", font=("Segoe UI", 14, "bold"), bg=self.app.COLOR_FRAME, fg=self.app.COLOR_PRIMARY)
        self.diagnosis_label.grid(row=1, column=0, sticky="w", padx=10, pady=6)
        self.appendicitis_prob_label = tk.Label(self.frame, text="Appendicitis Probability : -", anchor="w", font=self.app.FONT_NORMAL, bg=self.app.COLOR_FRAME)
        self.appendicitis_prob_label.grid(row=2, column=0, sticky="w", padx=10, pady=4)
        self.no_appendicitis_prob_label = tk.Label(self.frame, text="No Appendicitis Probability : -", anchor="w", font=self.app.FONT_NORMAL, bg=self.app.COLOR_FRAME )
        self.no_appendicitis_prob_label.grid(row=3, column=0, sticky="w", padx=10, pady=4)
        self.threshold_label = tk.Label(self.frame, text="Decision Threshold : -", anchor="w", font=self.app.FONT_NORMAL, bg=self.app.COLOR_FRAME)
        self.threshold_label.grid(row=4, column=0, sticky="w", padx=10, pady=4)
        self.laboratory_label = tk.Label(self.frame, text="Laboratory Status : -", anchor="w", font=self.app.FONT_NORMAL, bg=self.app.COLOR_FRAME)
        self.laboratory_label.grid(row=5, column=0, sticky="w", padx=10, pady=4)
        self.missing_label = tk.Label(self.frame, text="Missing Laboratory Fields : -", anchor="w", justify="left", wraplength=1000, font=self.app.FONT_SMALL, bg=self.app.COLOR_FRAME)
        self.missing_label.grid(row=6, column=0, sticky="w", padx=10, pady=4)


    def update_results(self, result):
        self.model_label.config(text=f"Model : {result['model']}")
        self.diagnosis_label.config(text=f"Diagnosis : {result['diagnosis']}")

        if result["prediction"] == 1:
            self.diagnosis_label.config(fg=self.app.COLOR_DANGER)
        else:
            self.diagnosis_label.config(fg=self.app.COLOR_SUCCESS)

        self.appendicitis_prob_label.config(text=f"Appendicitis Probability : {result['prob_appendicitis']:.2%}")
        self.no_appendicitis_prob_label.config(text=f"No Appendicitis Probability : {result['prob_no_appendicitis']:.2%}")
        self.threshold_label.config(text=f"Decision Threshold : {result['threshold']:.2f}")

        if result["laboratory_available"]:
            self.laboratory_label.config(text="Laboratory Status : Complete")
        else:
            self.laboratory_label.config(text="Laboratory Status : Missing Laboratory Values")

        missing = result["missing_laboratory_fields"]

        if len(missing) == 0:
            self.missing_label.config(text="Missing Laboratory Fields : None")
        else:
            self.missing_label.config(text="Missing Laboratory Fields :\n" + "\n".join(missing))

    # =====================================================
    # CLEAR
    # =====================================================

    def clear(self):
        self.model_label.config(text="Model : -")
        self.diagnosis_label.config(text="Diagnosis : -", fg=self.app.COLOR_PRIMARY)
        self.appendicitis_prob_label.config(text="Appendicitis Probability : -")
        self.no_appendicitis_prob_label.config(text="No Appendicitis Probability : -")
        self.threshold_label.config(text="Decision Threshold : -")
        self.laboratory_label.config(text="Laboratory Status : -")
        self.missing_label.config(text="Missing Laboratory Fields : -")