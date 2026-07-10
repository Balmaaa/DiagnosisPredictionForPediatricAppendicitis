import tkinter as tk
from gui.widgets import (create_section_frame, create_label, create_entry, create_combobox)


class PatientFrame:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.frame = None


    def build(self):
        self.frame = create_section_frame(parent=self.parent, title="Patient Information")
        self.create_widgets()
        self.configure_grid()


    def create_widgets(self):
        create_label(self.frame, "Age", 0, 0)
        create_entry(self.frame, self.app.age_var, 0, 1)
        create_label(self.frame, "Sex", 0, 2)
        create_combobox(self.frame, self.app.sex_var, ["Male", "Female"], 0, 3)
        create_label(self.frame, "Weight (kg)", 0, 4)
        create_entry(self.frame, self.app.weight_var, 0, 5)

        create_label(self.frame, "Height (cm)", 1, 0)
        create_entry( self.frame, self.app.height_var, 1, 1 )
        create_label( self.frame, "BMI", 1, 2 )
        create_entry( self.frame, self.app.bmi_var, 1, 3 )


    def configure_grid(self):
        for column in range(6):
            self.frame.grid_columnconfigure(column, weight=1)


    def clear(self):
        self.age_var.set("")
        self.sex_var.set("")
        self.weight_var.set("")
        self.height_var.set("")
        self.bmi_var.set("")