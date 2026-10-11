import tkinter as tk
from tkinter import ttk, messagebox

from backend_predictor import AppendicitisPredictor
from gui.patient_frame import PatientFrame
from gui.vital_frame import VitalFrame
from gui.symptoms_frame import SymptomsFrame
from gui.laboratory_frame import LaboratoryFrame
from gui.ultrasound_frame import UltrasoundFrame
from gui.complications_frame import ComplicationsFrame
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
        # Modern Flat UI Palette (Clean & Minimalist Dashboard Aesthetic)
        self.COLOR_BACKGROUND = "#F3F4F6"      # Soft light gray background
        self.COLOR_PRIMARY = "#2563EB"         # Modern vibrant blue (Tailwind blue-600)
        self.COLOR_SUCCESS = "#16A34A"         # Modern green
        self.COLOR_WARNING = "#D97706"         # Modern amber
        self.COLOR_DANGER = "#DC2626"          # Modern red
        self.COLOR_WHITE = "#FFFFFF"           # Card surface color
        self.COLOR_FRAME = "#FFFFFF"           # Clean white card containers
        self.COLOR_TEXT = "#1F2937"            # Dark slate for high legibility

        self.FONT_TITLE = ("Segoe UI", 18, "bold")
        self.FONT_SECTION = ("Segoe UI", 11, "bold")
        self.FONT_NORMAL = ("Segoe UI", 10)
        self.FONT_RESULT = ("Segoe UI", 11, "bold")
        self.FONT_SMALL = ("Segoe UI", 9)

        self.root.configure(bg=self.COLOR_BACKGROUND)

        # Configure Modern TTK Styles (Clam theme base)
        self.style = ttk.Style()
        self.style.theme_use("clam")

        # Modern Combobox Style
        self.style.configure(
            "TCombobox",
            fieldbackground="white",
            background="white",
            selectbackground=self.COLOR_PRIMARY,
            borderwidth=1
        )


    def initialize_backend(self):
        self.backend = AppendicitisPredictor()

    
    def initialize_variables(self):

        # =====================================================
        # PATIENT INFORMATION
        # =====================================================

        self.age_var = tk.StringVar()
        self.sex_var = tk.StringVar()
        self.weight_var = tk.StringVar()
        self.height_var = tk.StringVar()
        self.bmi_var = tk.StringVar()

        # =====================================================
        # VITAL SIGNS
        # =====================================================

        self.body_temperature_var = tk.StringVar()

        # =====================================================
        # CLINICAL SYMPTOMS
        # =====================================================
        self.migratory_pain_var = tk.StringVar()
        self.lower_right_abd_pain_var = tk.StringVar()
        self.contralateral_rebound_tenderness_var = tk.StringVar()
        self.coughing_pain_var = tk.StringVar()
        self.nausea_var = tk.StringVar()
        self.loss_of_appetite_var = tk.StringVar()
        self.dysuria_var = tk.StringVar()
        self.stool_var = tk.StringVar()
        self.peritonitis_var = tk.StringVar()
        self.psoas_sign_var = tk.StringVar()
        self.ipsilateral_rebound_tenderness_var = tk.StringVar()

        # =====================================================
        # LABORATORY VARIABLES
        # =====================================================

        self.appendix_diameter_var = tk.StringVar()
        self.wbc_count_var = tk.StringVar()
        self.crp_var = tk.StringVar()
        self.neutrophil_percentage_var = tk.StringVar()
        self.segmented_neutrophils_var = tk.StringVar()
        self.neutrophilia_var = tk.StringVar()
        self.rbc_count_var = tk.StringVar()
        self.hemoglobin_var = tk.StringVar()
        self.rdw_var = tk.StringVar()
        self.thrombocyte_count_var = tk.StringVar()
        self.ketones_in_urine_var = tk.StringVar()
        self.rbc_in_urine_var = tk.StringVar()
        self.wbc_in_urine_var = tk.StringVar()

        # =====================================================
        # ULTRASOUND VARIABLES
        # =====================================================

        self.appendix_on_us_var = tk.StringVar()
        self.free_fluids_var = tk.StringVar()
        self.appendix_wall_layers_var = tk.StringVar()
        self.target_sign_var = tk.StringVar()
        self.appendicolith_var = tk.StringVar()
        self.perfusion_var = tk.StringVar()
        self.perforation_var = tk.StringVar()
        self.surrounding_tissue_reaction_var = tk.StringVar()

        # =====================================================
        # ULTRASOUND COMPLICATIONS
        # =====================================================

        self.appendicular_abscess_var = tk.StringVar()
        self.abscess_location_var = tk.StringVar()
        self.pathological_lymph_nodes_var = tk.StringVar()
        self.lymph_nodes_location_var = tk.StringVar()
        self.bowel_wall_thickening_var = tk.StringVar()
        self.conglomerate_of_bowel_loops_var = tk.StringVar()
        self.ileus_var = tk.StringVar()
        self.coprostasis_var = tk.StringVar()
        self.meteorism_var = tk.StringVar()
        self.enteritis_var = tk.StringVar()

    
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
        self.complications_frame = ComplicationsFrame(parent=self.main_frame, app=self)
        self.complications_frame.build()
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
        self.title_label.pack(pady=(15, 15))

    
    def create_buttons(self):
        self.button_frame = tk.Frame(self.main_frame, bg=self.COLOR_BACKGROUND)
        self.button_frame.pack(fill="x", pady=20, padx=15)
        
        self.predict_button = tk.Button(
            self.button_frame,
            text="Predict Diagnosis",
            command=self.predict,
            bg=self.COLOR_SUCCESS,
            fg="white",
            font=("Segoe UI", 10, "bold"),
            relief="flat",
            padx=20,
            pady=10,
            cursor="hand2"
        )
        self.predict_button.pack(side="left", padx=5)

        self.clear_button = tk.Button(
            self.button_frame,
            text="Clear Fields",
            command=self.clear_fields,
            bg=self.COLOR_WARNING,
            fg="white",
            font=("Segoe UI", 10, "bold"),
            relief="flat",
            padx=20,
            pady=10,
            cursor="hand2"
        )
        self.clear_button.pack(side="left", padx=5)

        self.exit_button = tk.Button(
            self.button_frame,
            text="Exit Application",
            command=self.root.destroy,
            bg=self.COLOR_DANGER,
            fg="white",
            font=("Segoe UI", 10, "bold"),
            relief="flat",
            padx=20,
            pady=10,
            cursor="hand2"
        )
        self.exit_button.pack(side="right", padx=5)


    def _parse_float(self, val_str, field_name):
        cleaned = val_str.strip()
        if cleaned == "":
            return None
        try:
            return float(cleaned)
        except ValueError:
            raise ValueError(f"Invalid numeric value for '{field_name}': '{val_str}'. Please enter a valid number.")


    def collect_inputs(self):
        patient = {}

        patient["Age"] = self._parse_float(self.age_var.get(), "Age")
        if patient["Age"] is None:
            raise ValueError("Age is required.")

        patient["Sex"] = self.sex_var.get().strip()
        if patient["Sex"] == "":
            raise ValueError("Sex is required.")

        patient["Weight"] = self._parse_float(self.weight_var.get(), "Weight (kg)")
        patient["Height"] = self._parse_float(self.height_var.get(), "Height (cm)")
        patient["BMI"] = self._parse_float(self.bmi_var.get(), "BMI")
        patient["Body_Temperature"] = self._parse_float(self.body_temperature_var.get(), "Body Temperature (°C)")

        # Symptoms
        patient["Migratory_Pain"] = self.migratory_pain_var.get().strip()
        patient["Lower_Right_Abd_Pain"] = self.lower_right_abd_pain_var.get().strip()
        patient["Contralateral_Rebound_Tenderness"] = self.contralateral_rebound_tenderness_var.get().strip()
        patient["Coughing_Pain"] = self.coughing_pain_var.get().strip()
        patient["Nausea"] = self.nausea_var.get().strip()
        patient["Loss_of_Appetite"] = self.loss_of_appetite_var.get().strip()
        patient["Dysuria"] = self.dysuria_var.get().strip()
        patient["Stool"] = self.stool_var.get().strip()
        patient["Peritonitis"] = self.peritonitis_var.get().strip()
        patient["Psoas_Sign"] = self.psoas_sign_var.get().strip()
        patient["Ipsilateral_Rebound_Tenderness"] = self.ipsilateral_rebound_tenderness_var.get().strip()

        # Laboratory
        patient["Appendix_Diameter"] = self._parse_float(self.appendix_diameter_var.get(), "Appendix Diameter")
        patient["WBC_Count"] = self._parse_float(self.wbc_count_var.get(), "WBC Count")
        patient["CRP"] = self._parse_float(self.crp_var.get(), "CRP")
        patient["Neutrophil_Percentage"] = self._parse_float(self.neutrophil_percentage_var.get(), "Neutrophil Percentage")
        patient["Segmented_Neutrophils"] = self._parse_float(self.segmented_neutrophils_var.get(), "Segmented Neutrophils")
        patient["Neutrophilia"] = self.neutrophilia_var.get().strip()
        patient["RBC_Count"] = self._parse_float(self.rbc_count_var.get(), "RBC Count")
        patient["Hemoglobin"] = self._parse_float(self.hemoglobin_var.get(), "Hemoglobin")
        patient["RDW"] = self._parse_float(self.rdw_var.get(), "RDW")
        patient["Thrombocyte_Count"] = self._parse_float(self.thrombocyte_count_var.get(), "Thrombocyte Count")
        patient["Ketones_in_Urine"] = self.ketones_in_urine_var.get().strip()
        patient["RBC_in_Urine"] = self.rbc_in_urine_var.get().strip()
        patient["WBC_in_Urine"] = self.wbc_in_urine_var.get().strip()

        # Ultrasound
        patient["Appendix_on_US"] = self.appendix_on_us_var.get().strip()
        patient["Free_Fluids"] = self.free_fluids_var.get().strip()
        patient["Appendix_Wall_Layers"] = self.appendix_wall_layers_var.get().strip()
        patient["Target_Sign"] = self.target_sign_var.get().strip()
        patient["Appendicolith"] = self.appendicolith_var.get().strip()
        patient["Perfusion"] = self.perfusion_var.get().strip()
        patient["Perforation"] = self.perforation_var.get().strip()
        patient["Surrounding_Tissue_Reaction"] = self.surrounding_tissue_reaction_var.get().strip()

        # Complications
        patient["Appendicular_Abscess"] = self.appendicular_abscess_var.get().strip()
        patient["Abscess_Location"] = self.abscess_location_var.get().strip()
        patient["Pathological_Lymph_Nodes"] = self.pathological_lymph_nodes_var.get().strip()
        patient["Lymph_Nodes_Location"] = self.lymph_nodes_location_var.get().strip()
        patient["Bowel_Wall_Thickening"] = self.bowel_wall_thickening_var.get().strip()
        patient["Conglomerate_of_Bowel_Loops"] = self.conglomerate_of_bowel_loops_var.get().strip()
        patient["Ileus"] = self.ileus_var.get().strip()
        patient["Coprostasis"] = self.coprostasis_var.get().strip()
        patient["Meteorism"] = self.meteorism_var.get().strip()
        patient["Enteritis"] = self.enteritis_var.get().strip()

        return patient


    def predict(self):
        try:
            patient = self.collect_inputs()
            result = self.backend.predict_all(patient)
            self.result_frame.update_results(result)

        except ValueError as e:
            messagebox.showerror("Input Validation Error", str(e))

        except Exception as e:
            messagebox.showerror("Prediction Error", f"An unexpected error occurred during model inference.\n\n{e}")


    def clear_fields(self):
        self.patient_frame.clear()
        self.vital_frame.clear()
        self.symptoms_frame.clear()
        self.laboratory_frame.clear()
        self.ultrasound_frame.clear()
        self.complications_frame.clear()
        self.result_frame.clear()


    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = AppendicitisGUI()
    app.run()