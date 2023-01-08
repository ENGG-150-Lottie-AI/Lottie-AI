import tkinter as tk
from tkinter import ttk
import customtkinter   
from tkinter import filedialog as fd
#from PIL import Image, ImageTk

# Choose Files gets the directory of each file selected, returns a tuple


directory = "/Lottie_AI/files"

#img_path = "/Desktop/Lottie/fileimg.jpg"
#images = []

class Page(ttk.Frame):
    def __init__(self, master, **kw):
        super().__init__(master, **kw)

class FilesPage(Page):
    def __init__(self, master, **kw):
        super().__init__(master, **kw)

        self.create_frame_content()

    def create_frame_content(self) -> ttk.Frame:
        self.frame_content = ttk.Frame(self)

        self.frame_content = customtkinter.CTkFrame(self, fg_color="#ffffff", 
                                                    width = 800, height = 400, 
                                                    border_width = 2, border_color = 
                                                    "#c7c7c7").grid(row = 1, column = 0, columnspan = 2, rowspan = 2, sticky = "ew")

        return self.frame_content  

class UploadsPage(Page):
    def __init__(self, master, **kw):
        super().__init__(master, **kw)

        self.create_frame_content()
    
    

    def create_frame_content(self) -> ttk.Frame:

        #self.frame_content = ttk.Frame(self)

        self.frame_content = customtkinter.CTkFrame(self, fg_color="#ffffff", 
                                                    width = 800, height = 400, 
                                                    border_width = 2, border_color = 
                                                    "#c7c7c7").grid(row = 1, column = 0, columnspan = 2, rowspan = 2, sticky = "ew")
        cf_button = customtkinter.CTkButton(self, command = self.choose_files, fg_color="#436448", hover_color = "#628E69", font =("Open Sans", 10))
        cf_button.configure(text="Choose Files")
        cf_button.grid(row=0, column=0, sticky = "w")

        ex_button = customtkinter.CTkButton(self, command = self.extract, fg_color="#436448", hover_color = "#628E69", font =("Open Sans", 10))
        ex_button.configure(text="Extract")
        ex_button.grid(row=0, column=1, sticky = "e")
        

        return self.frame_content      
    
    def extract(self):
        #extract engine place here
        print("hi")   

    def choose_files(self):
        #gets directories for files
        self.filenames = fd.askopenfilenames(initialdir=directory, title="Select File/s")
        
        '''
        for i in range(len(self.filenames)):
            image = ImageTk.PhotoImage(Image.open("fileimg.jpg"))
            label = tk.Label(image = image)
            label.photo = image

            label.grid(row=1, column = i)
            images.append(label)
        '''
        

        for i in range(len(self.filenames)):
            print(self.filenames[i])
 

class ScanPage(Page):
    def __init__(self, master, **kw):
        super().__init__(master, **kw)

        self.create_frame_content()

    def create_frame_content(self) -> ttk.Frame:


        self.frame_content = customtkinter.CTkFrame(self, fg_color="#436448", width = 800, height = 400)
        button = customtkinter.CTkButton(self, command=self.create_popout, fg_color="#436448", hover_color = "#628E69", font =("Open Sans", 10))
        button.configure(text="Scan")
        button.grid(row=1, column=4, sticky = "ne")
        
        
        return self.frame_content               


    def create_popout(self):
        window = customtkinter.CTkToplevel(self)
        window.title("Choose")
        window.geometry("400x200+520+250")
        window.grid_columnconfigure(1, weight=1)
        window.grid_columnconfigure((2, 3), weight=0)
        window.grid_rowconfigure((0, 1, 2), weight=1)

        label = customtkinter.CTkLabel(window, text="Kind of title document")
        label.grid (row=0, column=0, padx=20, pady=2)  

        label = customtkinter.CTkLabel(window, text="Output File")
        label.grid (row=0, column=1, padx=20, pady=2)  

        window.titledocu = customtkinter.CTkOptionMenu(window, values=["TCT", "LDC", "ETC"], fg_color = "#436448", button_color = "#436448", button_hover_color="#628E69" )
        window.titledocu.grid(row=1, column=0, padx=20, pady=10)

        window.output = customtkinter.CTkOptionMenu(window, values=["csv", "txt", "ETC"], fg_color = "#436448", button_color = "#436448", button_hover_color="#628E69" )
        window.output.grid(row=1, column=1, padx=20, pady=10)

        window.confirm = customtkinter.CTkButton(window, fg_color = "#436448", hover_color="#628E69", text = "Confirm")
        window.confirm.grid(row=2, column=0, padx=20, pady=10)
        window.confirm = customtkinter.CTkButton(window, fg_color = "#436448", hover_color="#628E69", text = "Cancel" )
        window.confirm.grid(row=2, column=1, padx=20, pady=10)   