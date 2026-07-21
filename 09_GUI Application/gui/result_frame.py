import tkinter as tk
from gui.widgets import create_section_frame


class ResultFrame:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.frame = None
        self.model_labels = {}


    def build(self):
        self.frame = create_section_frame(parent=self.parent, title="Prediction Result")
        self.frame.columnconfigure(0, weight=1)
        row = 0

        # ==============================
        # MODEL RESULTS
        # ==============================

        title = tk.Label(self.frame, text="Individual Model Predictions", font=self.app.FONT_SECTION, bg=self.app.COLOR_FRAME)
        title.grid(row=row, column=0, sticky="w", padx=10, pady=8)
        row += 1
        for model_name in ["Decision Tree", "Gradient Boosting", "XGBoost", "Transformer"]:
            label = tk.Label(self.frame, text=f"{model_name}: -", anchor="w", font=self.app.FONT_NORMAL, bg=self.app.COLOR_FRAME)
            label.grid(row=row, column=0, sticky="w", padx=20, pady=3)
            self.model_labels[model_name] = label
            row += 1

        # ==============================
        # CONSENSUS RESULT
        # ==============================

        row += 1
        self.consensus_label = tk.Label(self.frame, text="Consensus Diagnosis : -", font=("Segoe UI",14,"bold"), bg=self.app.COLOR_FRAME, fg=self.app.COLOR_PRIMARY)
        self.consensus_label.grid(row=row, column=0, sticky="w", padx=10, pady=8)
        row += 1
        self.agreement_label = tk.Label(self.frame, text="Model Agreement : -", font=self.app.FONT_NORMAL, bg=self.app.COLOR_FRAME)
        self.agreement_label.grid(row=row, column=0, sticky="w", padx=10, pady=3)
        row += 1
        self.confidence_label = tk.Label( self.frame, text="Confidence : -", font=self.app.FONT_NORMAL, bg=self.app.COLOR_FRAME )
        self.confidence_label.grid(row=row, column=0, sticky="w", padx=10, pady=3)
        row += 1
        self.highest_model_label = tk.Label(self.frame, text="Highest Confidence Model : -", font=self.app.FONT_NORMAL, bg=self.app.COLOR_FRAME)
        self.highest_model_label.grid(row=row, column=0, sticky="w", padx=10, pady=3)
        row += 1
        self.highest_probability_label = tk.Label(self.frame, text="Highest Probability : -", font=self.app.FONT_NORMAL, bg=self.app.COLOR_FRAME)
        self.highest_probability_label.grid(row=row, column=0, sticky="w", padx=10, pady=3)
        row += 1
        self.laboratory_label = tk.Label(self.frame, text="Laboratory Status : -", font=self.app.FONT_NORMAL, bg=self.app.COLOR_FRAME)
        self.laboratory_label.grid(row=row, column=0, sticky="w", padx=10, pady=3)
        row += 1
        self.missing_label = tk.Label(self.frame, text="Missing Laboratory Fields : -", justify="left", wraplength=1000, font=self.app.FONT_SMALL, bg=self.app.COLOR_FRAME)
        self.missing_label.grid(row=row, column=0, sticky="w", padx=10, pady=5)


    def update_results(self, result):

        # ==============================
        # DISPLAY EACH MODEL
        # ==============================

        for model_name, model_result in result["models"].items():
            prediction = model_result["diagnosis"]
            probability = model_result["prob_appendicitis"]
            threshold = model_result["threshold"]
            text = (f"{model_name}: " f"{prediction} | " f"Appendicitis Probability: {probability:.2%} | " f"Threshold: {threshold:.2f}")
            self.model_labels[model_name].config(text=text)

        # ==============================
        # FINAL CONSENSUS
        # ==============================

        diagnosis = result["diagnosis"]
        self.consensus_label.config(text=f"Consensus Diagnosis : {diagnosis}")
        
        if diagnosis == "Appendicitis":
            self.consensus_label.config(fg=self.app.COLOR_DANGER)
        elif diagnosis == "No Appendicitis":
            self.consensus_label.config(fg=self.app.COLOR_SUCCESS)
        else:
            self.consensus_label.config(fg=self.app.COLOR_WARNING)

        self.agreement_label.config(text=(f"Model Agreement : " f"{result['agreement']}/{result['total_models']}"))
        self.confidence_label.config(text=f"Confidence : {result['confidence']}")
        self.highest_model_label.config(text=(f"Highest Confidence Model : " f"{result['highest_model']}"))
        self.highest_probability_label.config(text=(f"Highest Probability : " f"{result['highest_probability']:.2%}"))

        # ==============================
        # LAB STATUS
        # ==============================

        missing_fields = []
        for model_result in result["models"].values():
            if len(model_result["missing_laboratory_fields"]) > 0:
                missing_fields.extend(model_result["missing_laboratory_fields"])

        missing_fields = sorted(list(set(missing_fields)))
        
        if len(missing_fields)==0:
            self.laboratory_label.config(text="Laboratory Status : Complete")
            self.missing_label.config(text="Missing Laboratory Fields : None")
        else:
            self.laboratory_label.config(text="Laboratory Status : Missing Values")
            self.missing_label.config(text=("Missing Laboratory Fields:\n" + "\n".join(missing_fields)))


    def clear(self):
        for model_name,label in self.model_labels.items():
            label.config(text=f"{model_name}: -")

        self.consensus_label.config(text="Consensus Diagnosis : -", fg=self.app.COLOR_PRIMARY)
        self.agreement_label.config(text="Model Agreement : -")
        self.confidence_label.config(text="Confidence : -")
        self.highest_model_label.config(text="Highest Confidence Model : -")
        self.highest_probability_label.config(text="Highest Probability : -")
        self.laboratory_label.config(text="Laboratory Status : Unknown")
        self.missing_label.config(text="Missing Laboratory Fields : -")