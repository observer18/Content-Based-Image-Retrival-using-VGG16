
# Importing the required modules
from tkinter import *
from tkinter import filedialog

class files :
    """ This particular class is only to open a file from Windows File Explorer and returning the file path. """

    def __init__(self) :
        """ Initializing function. """
        self.filepath = ''
        self.window = Tk()
        self.window.withdraw()

    def openFile(self) :
        """ The main File Dialog Box open function. """
        self.filepath = filedialog.askopenfile(initialdir = "./combined_datasets",
                                               title = " Open",
                                               filetypes = (("JPEG Files", "*jpg"),
                                                            ("PNG Files", ".png"),
                                                            ("All Files", "*.*")))
        return self.filepath

    def __del__(self) :
        """ Destructor function. """
        pass
