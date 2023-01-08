import tkinter as tk
from tkinter import ttk
import customtkinter
from page_view import FilesPage, UploadsPage, ScanPage
import tkinter.messagebox

class LottieView(ttk.Frame):
    def __init__(self, master, **kw):
        super().__init__(master, **kw)

        # Key: Tab name (Files, Upload, Scan)
        # Value: Page object (ttk.Frame)
        self.pages = {}

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)


        self.create_frame_treeview().grid(row = 0, column = 0, sticky = "ens")

        self.create_frame_page().grid(row = 0, column = 1)

    def create_frame_page(self) -> ttk.Frame:
        """
        Create the frame that will show the current treeview page
        """

        self.frame_page = ttk.Frame(self)
        return self.frame_page

    def create_frame_treeview(self) -> ttk.Frame:
        """
        Create the frame that will hold the treeview widget and instantiate the LottieTreeview
        """

        self.frame_treeview = ttk.Frame(self)

        self.treeview_lottie = LottieTreeview(self.frame_treeview)
        self.treeview_lottie.bind("<<TreeviewSelect>>", self.on_treeview_selection_changed)
        self.treeview_lottie.pack(fill=tk.BOTH, expand = True)

        return self.frame_treeview

    def on_treeview_selection_changed(self, event):
        """
        Switch frames
        """

        selected_item = self.treeview_lottie.focus()

        lottie_name = self.treeview_lottie.item(selected_item).get("text")

        self.show_page(lottie_name)

    def show_page(self, lottie_name: str):
        """
        pack_forget() all pages and pack the given page name
        """

        for page_name in self.pages.keys():
            self.pages[page_name].pack_forget()

        self.pages[lottie_name].pack(fill=tk.BOTH, expand = True)

    def add_page(self, lottie_name: str, page):
        """
        Instantiate a page frame and add it to the pages dictionary
        """

        self.pages[lottie_name] = page(self.frame_page)

        self.treeview_lottie.add_title(section_text = lottie_name)


class LottieTreeview(ttk.Treeview):
    def __init__(self, master, **kw):
        super().__init__(master, **kw)

        self.heading("#0", text = "Lottie AI")
    
    def add_title(self, section_text: str):
        """
        Insert a row
        """

        self.insert(parent="",
                    index=tk.END,
                    text=section_text)

if __name__ == "__main__":
    app = tk.Tk()

    app.geometry(f"{1100}x{580}")
    app.title("Lottie AI")

    style= ttk.Style()
    style.configure("Treeview", background = "#436448")
    style.configure("Treeview.Heading", background = "#436448", relief = "flat")
    style.configure("Treeview", rowheight = 30)
    style.map("Treeview",
             foreground = [("selected", "white")],
             background = [("selected", "#628E69")])

    lottie = LottieView(app, relief = "flat")
    lottie.add_page(lottie_name = "Files",
                    page=FilesPage)
    lottie.add_page(lottie_name = "Upload",
                    page=UploadsPage)
    lottie.add_page(lottie_name = "Scan",
                    page=ScanPage)

    lottie.pack(fill = tk.BOTH, expand = True)
    
    app.mainloop()