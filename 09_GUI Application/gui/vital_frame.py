from gui.widgets import (create_section_frame, create_label, create_entry)


class VitalFrame:
    def __init__(self, parent, app):

        self.parent = parent
        self.app = app
        self.frame = None


    def build(self):
        self.frame = create_section_frame(parent=self.parent, title="Vital Signs")
        self.create_widgets()
        self.configure_grid()
    

    def create_widgets(self):
        create_label(parent=self.frame, text="Body Temperature (°C)", row=0, column=0)
        create_entry(parent=self.frame, variable=self.app.body_temperature_var, row=0, column=1, width=15)


    def configure_grid(self):
        for column in range(2):
            self.frame.grid_columnconfigure(column, weight=1)