from gui.widgets import (create_section_frame, create_label, create_combobox)


class SymptomsFrame:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.frame = None
        self.yes_no = ["", "Yes", "No"]


    def build(self):
        self.frame = create_section_frame(parent=self.parent, title="Clinical Symptoms")
        self.create_widgets()
        self.configure_grid()


    def create_widgets(self):
        create_label(self.frame, "Migratory Pain", 0, 0)
        create_combobox( self.frame, self.app.migratory_pain_var, self.yes_no, 0, 1)
        create_label(self.frame, "Lower Right Abdominal Pain", 1, 0)
        create_combobox(self.frame, self.app.lower_right_abd_pain_var, self.yes_no, 1, 1)
        create_label(self.frame, "Contralateral Rebound Tenderness", 2, 0)
        create_combobox(self.frame, self.app.contralateral_rebound_tenderness_var, self.yes_no, 2, 1)
        create_label(self.frame, "Coughing Pain", 3, 0)
        create_combobox(self.frame, self.app.coughing_pain_var, self.yes_no, 3, 1)
        create_label(self.frame, "Nausea", 4, 0)
        create_combobox(self.frame, self.app.nausea_var, self.yes_no, 4, 1)
        create_label(self.frame, "Loss of Appetite", 5, 0)
        create_combobox(self.frame, self.app.loss_of_appetite_var, self.yes_no, 5, 1)

        create_label(self.frame, "Dysuria", 0, 2)
        create_combobox(self.frame, self.app.dysuria_var, self.yes_no, 0, 3)
        create_label(self.frame, "Stool", 1, 2)
        create_combobox(self.frame, self.app.stool_var, [ "", "normal", "diarrhea", "constipation", "constipation, diarrhea" ], 1, 3)
        create_label(self.frame, "Peritonitis", 2, 2)
        create_combobox(self.frame, self.app.peritonitis_var, [ "", "No", "local", "generalized" ], 2, 3)
        create_label(self.frame, "Psoas Sign", 3, 2)
        create_combobox(self.frame, self.app.psoas_sign_var, self.yes_no, 3, 3)
        create_label(self.frame, "Ipsilateral Rebound Tenderness", 4, 2)
        create_combobox(self.frame, self.app.ipsilateral_rebound_tenderness_var, self.yes_no, 4, 3)


    def configure_grid(self):
        for column in range(4):
            self.frame.grid_columnconfigure(column, weight=1)