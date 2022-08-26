from tkinter import *
from tkinter import messagebox

class MenuBar(Menu):
    def __init__(self, ad):
        Menu.__init__(self, ad)
        
        #File options (Open, Exit)
        file = Menu(self, tearoff=False)
        file.add_command(label="Open")  
        file.add_separator()
        file.add_command(label="Exit", underline=1, command=self.quit)
        self.add_cascade(label="File",underline=0, menu=file)
        
        #Tools options (Cut/Select subregion, Search region on image)
        tools = Menu(self, tearoff=0)  
        tools.add_command(label="Cut/Select")  
        tools.add_command(label="Search...")  
        self.add_cascade(label="Tools", menu=tools) 

        #Help options (About the program)
        help = Menu(self, tearoff=0)  
        help.add_command(label="About", command=self.about)  
        self.add_cascade(label="Help", menu=help)  

    def exit(self):
        self.exit

    def about(self):
            messagebox.showinfo(title='ArthritisDetec', 
            message='''Software made for the class of Processing and Analysis of Images in PUC Minas.
            
            By Danniel Vieira, João Amorim and Maria Fernanda''')


class MainApp(Tk):
    def __init__(self):
        Tk.__init__(self)
        menubar = MenuBar(self) # Declaring menu bar
        self.config(menu=menubar)

if __name__ == "__main__":
    ad=MainApp()
    ad.title('ArthritisDetec')
    ad.geometry('600x600') # Initial Resolution
    ad.mainloop()