import tkinter as tk
from tkinter import ttk, messagebox

from backend_predictor import AppendicitisPredictor


class AppendicitisGUI:

    # =====================================================
    # Initialization
    # =====================================================

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Pediatric Appendicitis Diagnosis Prediction System")
        self.root.geometry("1450x900")
        self.root.minsize(1200, 800)

        # Backend
        self.backend = AppendicitisPredictor()

        # Theme Colors
        self.COLOR_BACKGROUND = "#F5F7FA"
        self.COLOR_PRIMARY = "#1565C0"
        self.COLOR_SECONDARY = "#1976D2"
        self.COLOR_SUCCESS = "#2E7D32"
        self.COLOR_WARNING = "#F57C00"
        self.COLOR_DANGER = "#C62828"
        self.COLOR_WHITE = "#FFFFFF"
        self.COLOR_FRAME = "#E3F2FD"
        self.COLOR_ENTRY = "#FFFFFF"

        # Fonts
        self.FONT_TITLE = ("Segoe UI", 20, "bold")
        self.FONT_SECTION = ("Segoe UI", 12, "bold")
        self.FONT_NORMAL = ("Segoe UI", 10)
        self.FONT_RESULT = ("Segoe UI", 11, "bold")

        # Variables
        self.model_var = tk.StringVar()
        self.model_var.set(self.backend.get_available_models()[0])

        # Patient Information
        self.age_var = tk.StringVar()
        self.sex_var = tk.StringVar()
        self.weight_var = tk.StringVar()
        self.height_var = tk.StringVar()
        self.bmi_var = tk.StringVar()

        # Clinical
        self.body_temperature_var = tk.StringVar()
        self.abdominal_pain_var = tk.StringVar()
        self.migratory_pain_var = tk.StringVar()
        self.nausea_var = tk.StringVar()
        self.vomiting_var = tk.StringVar()
        self.anorexia_var = tk.StringVar()
        self.diarrhea_var = tk.StringVar()
        self.constipation_var = tk.StringVar()
        self.rebound_tenderness_var = tk.StringVar()
        self.guarding_var = tk.StringVar()

        # Laboratory
        self.wbc_var = tk.StringVar()
        self.rbc_var = tk.StringVar()
        self.hemoglobin_var = tk.StringVar()
        self.rdw_var = tk.StringVar()
        self.segmented_neutrophils_var = tk.StringVar()
        self.thrombocyte_var = tk.StringVar()
        self.crp_var = tk.StringVar()
        self.neutrophil_percentage_var = tk.StringVar()

        # Result Variables
        self.result_var = tk.StringVar(value="Waiting for prediction...")
        self.probability_var = tk.StringVar(value="-")
        self.missing_lab_var = tk.StringVar(value="-")
        self.threshold_var = tk.StringVar(value="-")

        # Main Container
        self.main_frame = tk.Frame(self.root, bg=self.COLOR_BACKGROUND)
        self.main_frame.pack(fill="both", expand=True, padx=15, pady=15)

        # Build GUI
        self.create_title()
        self.create_patient_frame()
        self.create_symptom_frame()
        self.create_lab_frame()
        self.create_model_frame()
        self.create_result_frame()
        self.create_buttons()


    # =====================================================
    # TITLE
    # =====================================================

    def create_title(self):
        title = tk.Label( self.main_frame, text="Pediatric Appendicitis Diagnosis Prediction System", font=self.FONT_TITLE, bg=self.COLOR_BACKGROUND, fg=self.COLOR_PRIMARY )
        title.pack(pady=(0, 15))


    # =====================================================
    # PATIENT INFORMATION
    # =====================================================

    def create_patient_frame(self):
        frame = tk.LabelFrame(self.main_frame, text="Patient Information", font=self.FONT_SECTION, bg=self.COLOR_FRAME, padx=15, pady=15)
        frame.pack(fill="x", pady=5)

        # ----------------------------
        # ROW 1
        # ----------------------------

        tk.Label(frame, text="Age", bg=self.COLOR_FRAME, font=self.FONT_NORMAL).grid(row=0, column=0, sticky="w", padx=5, pady=5)
        tk.Entry(frame, textvariable=self.age_var, width=12).grid(row=0, column=1, padx=5, pady=5)
        tk.Label(frame, text="Sex", bg=self.COLOR_FRAME, font=self.FONT_NORMAL).grid(row=0, column=2, sticky="w", padx=5, pady=5)
        ttk.Combobox(frame, textvariable=self.sex_var, values=["Male", "Female"], state="readonly", width=12).grid(row=0, column=3, padx=5, pady=5)
        tk.Label(frame, text="Weight (kg)", bg=self.COLOR_FRAME, font=self.FONT_NORMAL).grid(row=0, column=4, sticky="w", padx=5, pady=5)
        tk.Entry(frame, textvariable=self.weight_var, width=12).grid(row=0, column=5, padx=5, pady=5)

        # ----------------------------
        # ROW 2
        # ----------------------------

        tk.Label(frame, text="Height (cm)", bg=self.COLOR_FRAME, font=self.FONT_NORMAL).grid(row=1, column=0, sticky="w", padx=5, pady=5)
        tk.Entry(frame, textvariable=self.height_var, width=12).grid(row=1, column=1, padx=5, pady=5)
        tk.Label(frame, text="BMI", bg=self.COLOR_FRAME, font=self.FONT_NORMAL).grid(row=1, column=2, sticky="w", padx=5, pady=5)
        tk.Entry(frame, textvariable=self.bmi_var, width=12).grid(row=1, column=3, padx=5, pady=5)

        frame.grid_columnconfigure(6, weight=1)


    # =====================================================
    # CLINICAL SYMPTOMS
    # =====================================================

    def create_symptom_frame(self):
        frame = tk.LabelFrame(self.main_frame, text="Clinical Symptoms", font=self.FONT_SECTION, bg=self.COLOR_FRAME, padx=15, pady=15)
        frame.pack(fill="x", pady=5)
        yes_no = ["", "Yes", "No"]

        # ----------------------------
        # LEFT COLUMN
        # ----------------------------

        tk.Label(frame, text="Body Temperature (°C)", bg=self.COLOR_FRAME, font=self.FONT_NORMAL).grid(row=0, column=0, sticky="w", padx=5, pady=5)
        tk.Entry(frame, textvariable=self.body_temperature_var, width=12).grid(row=0, column=1, padx=5)
        tk.Label(frame, text="Abdominal Pain", bg=self.COLOR_FRAME, font=self.FONT_NORMAL).grid(row=1, column=0, sticky="w", padx=5, pady=5)
        ttk.Combobox(frame, textvariable=self.abdominal_pain_var, values=yes_no, state="readonly", width=12).grid(row=1, column=1)
        tk.Label(frame, text="Migratory Pain", bg=self.COLOR_FRAME, font=self.FONT_NORMAL).grid(row=2, column=0, sticky="w", padx=5, pady=5)
        ttk.Combobox(frame, textvariable=self.migratory_pain_var, values=yes_no, state="readonly", width=12).grid(row=2, column=1)
        tk.Label(frame, text="Nausea", bg=self.COLOR_FRAME, font=self.FONT_NORMAL).grid(row=3, column=0, sticky="w", padx=5, pady=5)
        ttk.Combobox(frame, textvariable=self.nausea_var, values=yes_no, state="readonly", width=12).grid(row=3, column=1)
        tk.Label(frame, text="Vomiting", bg=self.COLOR_FRAME, font=self.FONT_NORMAL).grid(row=4, column=0, sticky="w", padx=5, pady=5)
        ttk.Combobox(frame, textvariable=self.vomiting_var, values=yes_no, state="readonly", width=12).grid(row=4, column=1)

        # ----------------------------
        # RIGHT COLUMN
        # ----------------------------

        tk.Label(frame, text="Anorexia", bg=self.COLOR_FRAME, font=self.FONT_NORMAL).grid(row=0, column=2, sticky="w", padx=(40,5), pady=5)
        ttk.Combobox(frame, textvariable=self.anorexia_var, values=yes_no, state="readonly", width=12).grid(row=0, column=3)
        tk.Label(frame, text="Diarrhea", bg=self.COLOR_FRAME, font=self.FONT_NORMAL).grid(row=1, column=2, sticky="w", padx=(40,5), pady=5)
        ttk.Combobox(frame, textvariable=self.diarrhea_var, values=yes_no, state="readonly", width=12).grid(row=1, column=3)
        tk.Label(frame, text="Constipation", bg=self.COLOR_FRAME, font=self.FONT_NORMAL).grid(row=2, column=2, sticky="w", padx=(40,5), pady=5)
        ttk.Combobox(frame, textvariable=self.constipation_var, values=yes_no, state="readonly", width=12).grid(row=2, column=3)
        tk.Label(frame, text="Rebound Tenderness", bg=self.COLOR_FRAME, font=self.FONT_NORMAL).grid(row=3, column=2, sticky="w", padx=(40,5), pady=5)
        ttk.Combobox(frame, textvariable=self.rebound_tenderness_var, values=yes_no, state="readonly", width=12).grid(row=3, column=3)
        tk.Label(frame, text="Guarding", bg=self.COLOR_FRAME, font=self.FONT_NORMAL).grid(row=4, column=2, sticky="w", padx=(40,5), pady=5)
        ttk.Combobox(frame, textvariable=self.guarding_var, values=yes_no, state="readonly", width=12).grid(row=4, column=3)

        frame.grid_columnconfigure(4, weight=1)


    # =====================================================
    # LABORATORY RESULTS
    # =====================================================

    def create_lab_frame(self):
        frame = tk.LabelFrame( self.main_frame, text="Laboratory Results (Optional)", font=self.FONT_SECTION, bg=self.COLOR_FRAME, padx=15, pady=15 )
        frame.pack(fill="x", pady=5)

        # ----------------------------
        # LEFT COLUMN
        # ----------------------------

        tk.Label(frame, text="WBC Count", bg=self.COLOR_FRAME, font=self.FONT_NORMAL).grid(row=0, column=0, sticky="w", padx=5, pady=5)
        tk.Entry(frame, textvariable=self.wbc_var, width=15).grid(row=0, column=1, padx=5)
        tk.Label(frame, text="RBC Count", bg=self.COLOR_FRAME, font=self.FONT_NORMAL).grid(row=1, column=0, sticky="w", padx=5, pady=5)
        tk.Entry(frame, textvariable=self.rbc_var, width=15).grid(row=1, column=1, padx=5)
        tk.Label(frame, text="Hemoglobin", bg=self.COLOR_FRAME, font=self.FONT_NORMAL).grid(row=2, column=0, sticky="w", padx=5, pady=5)
        tk.Entry(frame, textvariable=self.hemoglobin_var, width=15).grid(row=2, column=1, padx=5)
        tk.Label(frame, text="RDW", bg=self.COLOR_FRAME, font=self.FONT_NORMAL).grid(row=3, column=0, sticky="w", padx=5, pady=5)
        tk.Entry(frame, textvariable=self.rdw_var, width=15).grid(row=3, column=1, padx=5)

        # ----------------------------
        # RIGHT COLUMN
        # ----------------------------

        tk.Label(frame, text="Segmented Neutrophils", bg=self.COLOR_FRAME, font=self.FONT_NORMAL).grid(row=0, column=2, sticky="w", padx=(40,5), pady=5)
        tk.Entry(frame, textvariable=self.segmented_neutrophils_var, width=15).grid(row=0, column=3, padx=5)
        tk.Label(frame, text="Thrombocyte Count", bg=self.COLOR_FRAME, font=self.FONT_NORMAL).grid(row=1, column=2, sticky="w", padx=(40,5), pady=5)
        tk.Entry(frame, textvariable=self.thrombocyte_var, width=15).grid(row=1, column=3, padx=5)
        tk.Label(frame, text="CRP", bg=self.COLOR_FRAME, font=self.FONT_NORMAL).grid(row=2, column=2, sticky="w", padx=(40,5), pady=5)
        tk.Entry(frame, textvariable=self.crp_var, width=15).grid(row=2, column=3, padx=5)
        tk.Label(frame, text="Neutrophil Percentage", bg=self.COLOR_FRAME, font=self.FONT_NORMAL).grid(row=3, column=2, sticky="w", padx=(40,5), pady=5)
        tk.Entry(frame, textvariable=self.neutrophil_percentage_var, width=15).grid(row=3, column=3, padx=5)

        # Information Label
        info = tk.Label(frame, text="Leave laboratory fields blank if unavailable. " "The backend will automatically handle missing laboratory values.", bg=self.COLOR_FRAME, fg=self.COLOR_PRIMARY, font=("Segoe UI", 9, "italic"))
        info.grid(row=4, column=0, columnspan=4, sticky="w", pady=(12,0))
        frame.grid_columnconfigure(4, weight=1)


    # =====================================================
    # MODEL SELECTION
    # =====================================================

    def create_model_frame(self):
        frame = tk.LabelFrame(self.main_frame, text="Prediction Model", font=self.FONT_SECTION, bg=self.COLOR_FRAME, padx=15, pady=15)
        frame.pack(fill="x", pady=5)
        tk.Label(frame, text="Select AI Model:", bg=self.COLOR_FRAME, font=self.FONT_NORMAL ).grid(row=0, column=0, padx=5, pady=5)
        self.model_combo = ttk.Combobox(frame, textvariable=self.model_var, values=self.backend.get_available_models(), state="readonly", width=30)
        self.model_combo.grid(row=0, column=1, padx=10, pady=5, sticky="w")
        frame.grid_columnconfigure(2, weight=1)


    # =====================================================
    # RESULT FRAME
    # =====================================================

    def create_result_frame(self):
        frame = tk.LabelFrame(self.main_frame, text="Prediction Result", font=self.FONT_SECTION, bg=self.COLOR_FRAME, padx=20, pady=20)
        frame.pack(fill="x", pady=10)

        # Diagnosis
        tk.Label(frame, text="Diagnosis", font=self.FONT_RESULT, bg=self.COLOR_FRAME).grid(row=0, column=0, sticky="w", pady=5)
        self.result_label = tk.Label(frame, textvariable=self.result_var, font=("Segoe UI", 13, "bold"), bg=self.COLOR_FRAME, fg=self.COLOR_PRIMARY)
        self.result_label.grid(row=0, column=1, sticky="w", padx=15)

        # Probability
        tk.Label(frame, text="Appendicitis Probability", font=self.FONT_RESULT, bg=self.COLOR_FRAME ).grid(row=1, column=0, sticky="w", pady=5)
        tk.Label(frame, textvariable=self.probability_var, font=self.FONT_NORMAL, bg=self.COLOR_FRAME ).grid(row=1, column=1, sticky="w", padx=15)

        # Threshold
        tk.Label(frame, text="Decision Threshold", font=self.FONT_RESULT, bg=self.COLOR_FRAME).grid(row=2, column=0, sticky="w", pady=5)
        tk.Label(frame, textvariable=self.threshold_var, font=self.FONT_NORMAL, bg=self.COLOR_FRAME).grid(row=2, column=1, sticky="w", padx=15)

        # Missing Labs
        tk.Label(frame, text="Missing Laboratory Fields", font=self.FONT_RESULT, bg=self.COLOR_FRAME).grid(row=3, column=0, sticky="nw", pady=5)
        tk.Label(frame, textvariable=self.missing_lab_var, justify="left", wraplength=700, bg=self.COLOR_FRAME, font=self.FONT_NORMAL).grid(row=3, column=1, sticky="w", padx=15)
        frame.grid_columnconfigure(1, weight=1)


    # =====================================================
    # BUTTONS
    # =====================================================

    def create_buttons(self):
        frame = tk.Frame(self.main_frame, bg=self.COLOR_BACKGROUND)
        frame.pack(fill="x", pady=15)
        self.predict_button = tk.Button(frame, text="Predict Diagnosis", command=self.predict, bg=self.COLOR_SUCCESS, fg="white", font=("Segoe UI", 11, "bold"), padx=20, pady=10)
        self.predict_button.pack(side="left", padx=10)
        self.clear_button = tk.Button(frame, text="Clear", command=self.clear_fields, bg=self.COLOR_WARNING, fg="white", font=("Segoe UI", 11, "bold"), padx=20, pady=10)
        self.clear_button.pack(side="left", padx=10)
        self.exit_button = tk.Button(frame, text="Exit", command=self.root.destroy, bg=self.COLOR_DANGER, fg="white", font=("Segoe UI", 11, "bold"), padx=20, pady=10)
        self.exit_button.pack(side="right", padx=10)


    # =====================================================
    # INPUTS
    # =====================================================

    def collect_inputs(self):
        patient = {}

        # =====================================================
        # REQUIRED
        # =====================================================

        patient["Age"] = (None if self.age_var.get().strip() == "" else float(self.age_var.get()))
        patient["Sex"] = self.sex_var.get()

        # =====================================================
        # PHYSICAL
        # =====================================================

        patient["Weight"] = (None if self.weight_var.get().strip() == "" else float(self.weight_var.get()))
        patient["Height"] = (None if self.height_var.get().strip() == "" else float(self.height_var.get()))
        patient["BMI"] = (None if self.bmi_var.get().strip() == "" else float(self.bmi_var.get()))
        patient["Body_Temperature"] = (None if self.body_temperature_var.get().strip() == "" else float(self.body_temperature_var.get()))

        # =====================================================
        # CLINICAL SYMPTOMS
        # =====================================================

        patient["Abdominal_Pain"] = self.abdominal_pain_var.get()
        patient["Migratory_Pain"] = self.migratory_pain_var.get()
        patient["Anorexia"] = self.anorexia_var.get()
        patient["Nausea"] = self.nausea_var.get()
        patient["Vomiting"] = self.vomiting_var.get()
        patient["Diarrhea"] = self.diarrhea_var.get()
        patient["Constipation"] = self.constipation_var.get()
        patient["Rebound_Tenderness"] = self.rebound_tenderness_var.get()
        patient["Guarding"] = self.guarding_var.get()

        # =====================================================
        # LABORATORY
        # =====================================================

        patient["WBC_Count"] = (None if self.wbc_var.get().strip() == "" else float(self.wbc_var.get()))
        patient["RBC_Count"] = (None if self.rbc_var.get().strip() == "" else float(self.rbc_var.get()))
        patient["Hemoglobin"] = (None if self.hemoglobin_var.get().strip() == "" else float(self.hemoglobin_var.get()))
        patient["RDW"] = (None if self.rdw_var.get().strip() == "" else float(self.rdw_var.get()))
        patient["Segmented_Neutrophils"] = (None if self.segmented_neutrophils_var.get().strip() == "" else float(self.segmented_neutrophils_var.get()))
        patient["Thrombocyte_Count"] = (None if self.thrombocyte_var.get().strip() == "" else float(self.thrombocyte_var.get()))
        patient["CRP"] = (None if self.crp_var.get().strip() == "" else float(self.crp_var.get()))
        patient["Neutrophil_Percentage"] = (None if self.neutrophil_percentage_var.get().strip() == "" else float(self.neutrophil_percentage_var.get()))

        return patient


    # =====================================================
    # PREDICTION
    # =====================================================

    def predict(self):
        try:
            model_name = self.model_var.get()

            if model_name == "": messagebox.showwarning("Model Required", "Please select a prediction model.")
            return

            patient = self.collect_inputs()
            result = self.backend.predict(model_name=model_name, input_data=patient)
            self.display_result(result)

        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
        except Exception as e:
            messagebox.showerror("Prediction Error", f"An unexpected error occurred.\n\n{e}")


    # =====================================================
    # DISPLAY RESULTS
    # =====================================================

    def display_result(self, result):
        diagnosis = result["diagnosis"]
        probability = result["prob_appendicitis"] * 100
        threshold = result["threshold"] * 100
        model = result["model"]
        missing_labs = result["missing_laboratory_fields"]

        if diagnosis == "Appendicitis":
            self.result_label.configure(text="APPENDICITIS DETECTED", text_color="red")
        else:
            self.result_label.configure(text="NO APPENDICITIS DETECTED", text_color="green")

        self.probability_label.configure(text=f"Appendicitis Probability: {probability:.2f}%")
        self.threshold_label.configure(text=f"Decision Threshold: {threshold:.0f}%")
        self.model_used_label.configure(text=f"Model Used: {model}")

        if len(missing_labs) == 0:
            self.lab_status_label.configure(text="Laboratory Data: Complete", text_color="green")
        else:
            missing = ", ".join(missing_labs)
            self.lab_status_label.configure(text=f"Missing Laboratory Fields:\n{missing}", text_color="orange")


    # =====================================================
    # CLEAR INPUT FIELDS
    # =====================================================

    def clear_fields(self):
        pass

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = AppendicitisGUI()
    app.run()