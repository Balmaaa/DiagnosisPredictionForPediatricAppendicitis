from gui.widgets import (create_section_frame, create_label, create_combobox)


class ComplicationsFrame:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.frame = None
        self.yes_no = ["", "Yes", "No"]


    def build(self):
        self.frame = create_section_frame(parent=self.parent, title="Ultrasound Complications")
        self.create_widgets()
        self.configure_grid()


    def create_widgets(self):
        create_label(self.frame, "Appendicular Abscess", 0, 0)
        create_combobox(self.frame, self.app.appendicular_abscess_var, [ "", "Yes", "No", "suspected" ], 0, 1)
        create_label(self.frame, "Abscess Location", 0, 2)
        create_combobox(self.frame, self.app.abscess_location_var, [ "", "douglas_pouch", "Other" ], 0, 3)

        create_label(self.frame, "Pathological Lymph Nodes", 1, 0)
        create_combobox(self.frame, self.app.pathological_lymph_nodes_var, self.yes_no, 1, 1)
        create_label(self.frame, "Lymph Node Location", 1, 2)
        create_combobox(self.frame, self.app.lymph_nodes_location_var, ["", "right_lower_quadrant", "mesenteric", "ileocecal", "right_mid_abdomen", "Other"], 1, 3)

        create_label(self.frame, "Bowel Wall Thickening", 2, 0)
        create_combobox(self.frame, self.app.bowel_wall_thickening_var, self.yes_no, 2, 1)
        create_label(self.frame, "Conglomerate of Bowel Loops", 2, 2)
        create_combobox(self.frame, self.app.conglomerate_of_bowel_loops_var, self.yes_no, 2, 3)

        create_label(self.frame, "Ileus", 3, 0)
        create_combobox(self.frame, self.app.ileus_var, self.yes_no, 3, 1)
        create_label(self.frame, "Coprostasis", 3, 2)
        create_combobox(self.frame, self.app.coprostasis_var, self.yes_no, 3, 3)

        create_label(self.frame, "Meteorism", 4, 0)
        create_combobox(self.frame, self.app.meteorism_var, self.yes_no, 4, 1)
        create_label(self.frame, "Enteritis", 4, 2)
        create_combobox(self.frame, self.app.enteritis_var, self.yes_no, 4, 3)


    # =====================================================
    # GRID CONFIGURATION
    # =====================================================

    def configure_grid(self):
        for column in range(4):
            self.frame.grid_columnconfigure( column, weight=1 )