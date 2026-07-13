from gui.widgets import (create_section_frame, create_label, create_entry, create_combobox)


class LaboratoryFrame:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.frame = None


    def build(self):
        self.frame = create_section_frame(parent=self.parent, title="Laboratory Results")
        self.create_widgets()
        self.configure_grid()


    def create_widgets(self):
        create_label(self.frame, "Appendix Diameter (mm)", 0, 0)
        create_entry(self.frame, self.app.appendix_diameter_var, 0, 1)

        create_label(self.frame, "WBC Count", 0, 2)
        create_entry(self.frame, self.app.wbc_count_var, 0, 3)

        create_label(self.frame, "CRP", 0, 4)
        create_entry(self.frame, self.app.crp_var, 0, 5)

        create_label(self.frame, "Neutrophil Percentage", 1, 0)
        create_entry(self.frame, self.app.neutrophil_percentage_var, 1, 1)

        create_label(self.frame, "Segmented Neutrophils", 1, 2)
        create_entry(self.frame, self.app.segmented_neutrophils_var, 1, 3)

        create_label(self.frame, "Neutrophilia", 1, 4)
        create_combobox(self.frame, self.app.neutrophilia_var, ["", "Yes", "No"], 1, 5)

        create_label(self.frame, "RBC Count", 2, 0)
        create_entry(self.frame, self.app.rbc_count_var, 2, 1)

        create_label(self.frame, "Hemoglobin", 2, 2)
        create_entry(self.frame, self.app.hemoglobin_var, 2, 3)

        create_label(self.frame, "RDW", 2, 4)
        create_entry(self.frame, self.app.rdw_var, 2, 5)

        create_label(self.frame, "Thrombocyte Count", 3, 0)
        create_entry(self.frame, self.app.thrombocyte_count_var, 3, 1)

        create_label(self.frame, "Ketones in Urine", 3, 2)
        create_combobox(self.frame, self.app.ketones_in_urine_var, ["", "+", "++", "+++"], 3, 3)

        create_label(self.frame, "RBC in Urine", 3, 4)
        create_combobox(self.frame, self.app.rbc_in_urine_var, ["", "+", "++", "+++"], 3, 5)

        create_label(self.frame, "WBC in Urine", 4, 0)
        create_combobox(self.frame, self.app.wbc_in_urine_var, ["", "Yes", "No"], 4, 1)


    def configure_grid(self):
        for column in range(6):
            self.frame.grid_columnconfigure(column, weight=1)

    
    def clear(self):
        self.app.appendix_diameter_var.set("")
        self.app.wbc_count_var.set("")
        self.app.crp_var.set("")
        self.app.neutrophil_percentage_var.set("")
        self.app.segmented_neutrophils_var.set("")
        self.app.neutrophilia_var.set("")
        self.app.rbc_count_var.set("")
        self.app.hemoglobin_var.set("")
        self.app.rdw_var.set("")
        self.app.thrombocyte_count_var.set("")
        self.app.ketones_in_urine_var.set("")
        self.app.rbc_in_urine_var.set("")
        self.app.wbc_in_urine_var.set("")