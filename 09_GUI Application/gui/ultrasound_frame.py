from gui.widgets import (create_section_frame, create_label, create_combobox)


class UltrasoundFrame:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.frame = None
        self.yes_no = ["", "Yes", "No"]


    def build(self):
        self.frame = create_section_frame(parent=self.parent, title="Ultrasound Findings")
        self.create_widgets()
        self.configure_grid()


    def create_widgets(self):
        create_label(self.frame, "Ultrasound Performed", 0, 0)
        create_combobox(self.frame, self.app.us_performed_var, self.yes_no, 0, 1)
        create_label(self.frame, "Appendix Visualized", 0, 2)
        create_combobox(self.frame, self.app.appendix_on_us_var, self.yes_no, 0, 3)

        create_label(self.frame, "Free Fluids", 1, 0)
        create_combobox(self.frame, self.app.free_fluids_var, self.yes_no, 1, 1)
        create_label(self.frame, "Appendix Wall Layers", 1, 2)
        create_combobox(self.frame, self.app.appendix_wall_layers_var, ["", "intact", "raised", "partially raised", "upset"], 1, 3)

        create_label(self.frame, "Target Sign", 2, 0)
        create_combobox(self.frame, self.app.target_sign_var, self.yes_no, 2, 1)
        create_label(self.frame, "Appendicolith", 2, 2)
        create_combobox(self.frame, self.app.appendicolith_var, [ "", "Yes", "No", "suspected" ], 2, 3)

        create_label(self.frame, "Perfusion", 3, 0)
        create_combobox(self.frame, self.app.perfusion_var, [ "", "hyperperfused", "hypoperfused", "No", "present" ], 3, 1)
        create_label(self.frame, "Perforation", 3, 2)
        create_combobox(self.frame, self.app.perforation_var, ["", "Yes", "No", "suspected", "not excluded"], 3, 3)

        create_label(self.frame, "Surrounding Tissue Reaction", 4, 0)
        create_combobox(self.frame, self.app.surrounding_tissue_reaction_var, self.yes_no, 4, 1)


    def configure_grid(self):
        for column in range(4):
            self.frame.grid_columnconfigure(column, weight=1)


    def clear(self):
        self.appendix_on_us_var.set("")
        self.target_sign_var.set("")
        self.appendicolith_var.set("")
        self.perfusion_var.set("")
        self.perforation_var.set("")
        self.surrounding_tissue_reaction_var.set("")
        self.free_fluids_var.set("")
        self.appendix_location_var.set("")