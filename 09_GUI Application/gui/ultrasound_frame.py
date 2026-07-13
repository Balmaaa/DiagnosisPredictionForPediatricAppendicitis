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
        create_label(self.frame, "Appendix on Ultrasound", 0, 0)
        create_combobox(self.frame, self.app.appendix_on_us_var, self.yes_no, 0, 1)

        create_label(self.frame, "Free Fluids", 0, 2)
        create_combobox(self.frame, self.app.free_fluids_var, self.yes_no, 0, 3)

        create_label(self.frame, "Appendix Wall Layers", 1, 0)
        create_combobox(self.frame, self.app.appendix_wall_layers_var, ["", "intact", "raised", "partially raised", "upset"], 1, 1)

        create_label(self.frame, "Target Sign", 1, 2)
        create_combobox(self.frame, self.app.target_sign_var, self.yes_no, 1, 3)

        create_label(self.frame, "Appendicolith", 2, 0)
        create_combobox(self.frame, self.app.appendicolith_var, ["", "Yes", "No", "suspected"], 2, 1)

        create_label(self.frame, "Perfusion", 2, 2)
        create_combobox(self.frame, self.app.perfusion_var, ["", "hyperperfused", "hypoperfused", "No", "present"], 2, 3)

        create_label(self.frame, "Perforation", 3, 0)
        create_combobox(self.frame, self.app.perforation_var, ["", "Yes", "No", "suspected", "not excluded"], 3, 1)

        create_label(self.frame, "Surrounding Tissue Reaction", 3, 2)
        create_combobox(self.frame, self.app.surrounding_tissue_reaction_var, self.yes_no, 3, 3)


    def configure_grid(self):
        for column in range(4):
            self.frame.grid_columnconfigure(column, weight=1)


    def clear(self):
        self.app.appendix_on_us_var.set("")
        self.app.free_fluids_var.set("")
        self.app.appendix_wall_layers_var.set("")
        self.app.target_sign_var.set("")
        self.app.appendicolith_var.set("")
        self.app.perfusion_var.set("")
        self.app.perforation_var.set("")
        self.app.surrounding_tissue_reaction_var.set("")