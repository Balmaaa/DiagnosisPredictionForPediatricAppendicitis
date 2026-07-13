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


    def collect_inputs(self):
        patient = {}

        # =====================================================
        # PATIENT INFORMATION
        # =====================================================

        patient["Age"] = (None if self.age_var.get().strip() == "" else float(self.age_var.get()))

        if patient["Age"] is None:
            raise ValueError("Age is required.")

        patient["Sex"] = self.sex_var.get()

        if patient["Sex"] == "":
            raise ValueError("Sex is required.")

        patient["Weight"] = (None if self.weight_var.get().strip() == "" else float(self.weight_var.get()))
        patient["Height"] = (None if self.height_var.get().strip() == "" else float(self.height_var.get()))
        patient["BMI"] = (None if self.bmi_var.get().strip() == "" else float(self.bmi_var.get()))

        # =====================================================
        # VITAL SIGNS
        # =====================================================

        patient["Body_Temperature"] = (None if self.vital_frame.body_temperature_var.get().strip() == "" else float(self.vital_frame.body_temperature_var.get()))

        # =====================================================
        # CLINICAL SYMPTOMS
        # =====================================================

        patient["Migratory_Pain"] = self.symptoms_frame.migratory_pain_var.get()
        patient["Lower_Right_Abd_Pain"] = (self.symptoms_frame.lower_right_abd_pain_var.get() )
        patient["Contralateral_Rebound_Tenderness"] = (self.symptoms_frame.contralateral_rebound_tenderness_var.get())
        patient["Coughing_Pain"] = (self.symptoms_frame.coughing_pain_var.get())
        patient["Nausea"] = (self.symptoms_frame.nausea_var.get())
        patient["Loss_of_Appetite"] = (self.symptoms_frame.loss_of_appetite_var.get())
        patient["Dysuria"] = (self.symptoms_frame.dysuria_var.get())
        patient["Stool"] = (self.symptoms_frame.stool_var.get())
        patient["Peritonitis"] = (self.symptoms_frame.peritonitis_var.get())
        patient["Psoas_Sign"] = (self.symptoms_frame.psoas_sign_var.get())
        patient["Ipsilateral_Rebound_Tenderness"] = ( self.symptoms_frame.ipsilateral_rebound_tenderness_var.get())

        # =====================================================
        # LABORATORY
        # =====================================================

        patient["Appendix_Diameter"] = (None if self.laboratory_frame.appendix_diameter_var.get().strip() == "" else float(self.laboratory_frame.appendix_diameter_var.get()))
        patient["WBC_Count"] = (None if self.laboratory_frame.wbc_count_var.get().strip() == "" else float(self.laboratory_frame.wbc_count_var.get()))
        patient["CRP"] = (None if self.laboratory_frame.crp_var.get().strip() == "" else float(self.laboratory_frame.crp_var.get()))
        patient["Neutrophil_Percentage"] = (None if self.laboratory_frame.neutrophil_percentage_var.get().strip() == "" else float(self.laboratory_frame.neutrophil_percentage_var.get()))
        patient["Segmented_Neutrophils"] = (None if self.laboratory_frame.segmented_neutrophils_var.get().strip() == "" else float(self.laboratory_frame.segmented_neutrophils_var.get()))
        patient["Neutrophilia"] = (self.laboratory_frame.neutrophilia_var.get())
        patient["RBC_Count"] = (None if self.laboratory_frame.rbc_count_var.get().strip() == "" else float(self.laboratory_frame.rbc_count_var.get()))
        patient["Hemoglobin"] = (None if self.laboratory_frame.hemoglobin_var.get().strip() == "" else float(self.laboratory_frame.hemoglobin_var.get()))
        patient["RDW"] = (None if self.laboratory_frame.rdw_var.get().strip() == "" else float(self.laboratory_frame.rdw_var.get()))
        patient["Thrombocyte_Count"] = (None if self.laboratory_frame.thrombocyte_count_var.get().strip() == "" else float(self.laboratory_frame.thrombocyte_count_var.get()))
        patient["Ketones_in_Urine"] = (self.laboratory_frame.ketones_in_urine_var.get())
        patient["RBC_in_Urine"] = (self.laboratory_frame.rbc_in_urine_var.get())
        patient["WBC_in_Urine"] = (self.laboratory_frame.wbc_in_urine_var.get())

        # =====================================================
        # ULTRASOUND
        # =====================================================

        patient["Appendix_on_US"] = self.appendix_on_us_var.get()
        patient["Target_Sign"] = self.target_sign_var.get()
        patient["Appendicolith"] = self.appendicolith_var.get()
        patient["Perfusion"] = self.perfusion_var.get()
        patient["Perforation"] = self.perforation_var.get()
        patient["Surrounding_Tissue_Reaction"] = self.surrounding_tissue_reaction_var.get()
        patient["Free_Fluids"] = self.free_fluids_var.get()

        # =====================================================
        # ULTRASOUND COMPLICATIONS
        # =====================================================

        patient["Appendicular_Abscess"] = (self.complications_frame.appendicular_abscess_var.get())
        patient["Abscess_Location"] = (self.complications_frame.abscess_location_var.get())
        patient["Pathological_Lymph_Nodes"] = (self.complications_frame.pathological_lymph_nodes_var.get())
        patient["Lymph_Nodes_Location"] = (self.complications_frame.lymph_nodes_location_var.get())
        patient["Bowel_Wall_Thickening"] = (self.complications_frame.bowel_wall_thickening_var.get())
        patient["Conglomerate_of_Bowel_Loops"] = (self.complications_frame.conglomerate_of_bowel_loops_var.get())
        patient["Ileus"] = (self.complications_frame.ileus_var.get())
        patient["Coprostasis"] = (self.complications_frame.coprostasis_var.get())
        patient["Meteorism"] = (self.complications_frame.meteorism_var.get())
        patient["Enteritis"] = (self.complications_frame.enteritis_var.get())

        return patient


    # =====================================================
    # PREDICTION
    # =====================================================

    def predict(self):
        try:
            model_name = self.model_var.get()
            if model_name == "":
                messagebox.showwarning("Model Required", "Please select a prediction model.")
                return

            patient = self.collect_inputs()
            result = self.backend.predict(model_name=model_name, input_data=patient)
            self.display_result(result)

        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
        except Exception as e:
            messagebox.showerror("Prediction Error", f"An unexpected error occurred.\n\n{e}")


    # =====================================================
    # DISPLAY RESULT
    # =====================================================

    def display_result(self, result):
        self.result_frame.update_results(result)


    # =====================================================
    # CLEAR ALL INPUTS
    # =====================================================

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