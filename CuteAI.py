"""
    Main File for the GUI
"""
import tkinter
from tkinter import Label, ttk

from ttkbootstrap import Style
import icon
from dndRunner import Tester
from SinGAN import SinGAN
from mnistGAN import mnistGAN
import _thread

import warnings
warnings.filterwarnings("ignore")

class Application(tkinter.Tk):

    def __init__(self):
        super().__init__()
        self.title('CuteAI')
        self.style = Style('minty')
        self.GUI = GUI(self)

        self.GUI.pack(fill='both', expand='yes')

        # custom styles
        self.style.configure('header.TLabel', background=self.style.colors.secondary, foreground=self.style.colors.info)
        
        # do not allow window resizing
        self.resizable(False, False)
    
        


class GUI(ttk.Frame):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # application images
        self.plotPath = 'images/SinGAN.png'
        self.logo_img = tkinter.PhotoImage(name='logo', file='images/cuteAI_1_10.png')
        self.plotImage = tkinter.PhotoImage(name=self.plotPath, file=self.plotPath)
        self.privacy_img = tkinter.PhotoImage(name='privacy', file='images/pipeLine.png')
        
        self.inThread = False
        self.process = None
        # header
        header_frame = ttk.Frame(self, padding=20, style='secondary.TFrame')
        header_frame.grid(row=0, column=0, columnspan=3, sticky='ew')
        ttk.Label(header_frame, image='logo', style='header.TLabel').pack(side='left')

        # option notebook
        notebook = ttk.Notebook(self)
        notebook.grid(row=1, column=1, sticky='nsew', pady=(20, 0))

        radio_options = [
            'Internet Cache', 'Internet History', 'Cookies', 'Download History', 'Last Download Location',
            'Session', 'Set Aside Tabs', 'Recently Typed URLs', 'Saved Form Information', 'Saved Password']

        ## simple tab
        simple_tab = ttk.Frame(notebook, padding=10)
        wt_scrollbar_s = tkinter.Scrollbar(simple_tab)
        wt_scrollbar_s.pack(side='right', fill='y')
        wt_canvas_s = tkinter.Canvas(simple_tab, border=0, highlightthickness=0, yscrollcommand=wt_scrollbar_s.set)
        wt_canvas_s.pack(side='left', fill='both')

        wt_canvas_s.bind('<Configure>', lambda e: wt_canvas_s.configure(scrollregion=wt_canvas_s.bbox('all')))
        wt_scrollbar_s.configure(command=wt_canvas_s.yview)
        scroll_frame = ttk.Frame(wt_canvas_s)
        wt_canvas_s.create_window((0, 0), window=scroll_frame, anchor='nw')

        edge = ttk.Labelframe(scroll_frame, text='ME', padding=(20, 5))
        edge.pack(fill='both')

        explorer = ttk.Labelframe(scroll_frame, text='IE', padding=(20, 5))
        explorer.pack(fill='both', pady=10)

        s_in = ttk.Label(wt_canvas_s, text="Inputs:")
        s_in.place(x = 10, y = 10)

        s_alg = ttk.Label(wt_canvas_s, text="Algorithms:")
        s_alg.place(x = 10, y = 140)

        ## intermediate tab
        intermediate_tab = ttk.Frame(notebook, padding=10)
        wt_scrollbar_i = tkinter.Scrollbar(intermediate_tab)
        wt_scrollbar_i.pack(side='right', fill='y')
        wt_canvas_i = tkinter.Canvas(intermediate_tab, border=0, highlightthickness=0, yscrollcommand=wt_scrollbar_i.set)
        wt_canvas_i.pack(side='left', fill='both')

        wt_canvas_i.bind('<Configure>', lambda e: wt_canvas_i.configure(scrollregion=wt_canvas_i.bbox('all')))
        wt_scrollbar_i.configure(command=wt_canvas_i.yview)
        scroll_frame = ttk.Frame(wt_canvas_i)
        wt_canvas_i.create_window((0, 0), window=scroll_frame, anchor='nw')

        edge = ttk.Labelframe(scroll_frame, text='ME', padding=(20, 5))
        edge.pack(fill='both')

        explorer = ttk.Labelframe(scroll_frame, text='IE', padding=(20, 5))
        explorer.pack(fill='both', pady=10)

        i_in = ttk.Label(wt_canvas_i, text="Inputs:")
        i_in.place(x = 10, y = 10)

        i_alg = ttk.Label(wt_canvas_i, text="Algorithms:")
        i_alg.place(x = 10, y = 140)

        i_tsize_label = ttk.Label(wt_canvas_i, text="Training Size:")
        i_tsize_entry = ttk.Entry(wt_canvas_i)
        # a_tsize_entry.insert("1024")

        i_bsize_label = ttk.Label(wt_canvas_i, text="Batch Size:")
        i_bsize_entry = ttk.Entry(wt_canvas_i)
        # a_bsize_entry.insert("32")

        i_lr_label = ttk.Label(wt_canvas_i, text="Learning Rate:")
        i_lr_entry = ttk.Entry(wt_canvas_i)
        # a_lr_entry.insert("0.001")

        i_epochs_label = ttk.Label(wt_canvas_i, text="Number of Epochs:")
        i_epochs_entry = ttk.Entry(wt_canvas_i)
        # a_epochs_entry.insert("300")

        i_tsize_label.place(x = 10, y = 300)
        i_tsize_entry.place(x = 130, y = 300)

        i_bsize_label.place(x = 10, y = 350)
        i_bsize_entry.place(x = 130, y = 350)

        i_lr_label.place(x = 10, y = 400)
        i_lr_entry.place(x = 130, y = 400)

        i_epochs_label.place(x = 10, y = 450)
        i_epochs_entry.place(x = 130, y = 450)

        self.intermediate_parameters = [i_tsize_entry, i_bsize_entry, i_lr_entry, i_epochs_entry]

        # advanced tab
        advanced_tab = ttk.Frame(notebook, padding=10)
        wt_scrollbar_a = tkinter.Scrollbar(advanced_tab)
        wt_scrollbar_a.pack(side='right', fill='y')
        wt_canvas_a = tkinter.Canvas(advanced_tab, border=0, highlightthickness=0, yscrollcommand=wt_scrollbar_a.set)
        wt_canvas_a.pack(side='left', fill='both')

        wt_canvas_a.bind('<Configure>', lambda e: wt_canvas_a.configure(scrollregion=wt_canvas_a.bbox('all')))
        wt_scrollbar_a.configure(command=wt_canvas_a.yview)
        scroll_frame = ttk.Frame(wt_canvas_a)
        wt_canvas_a.create_window((0, 0), window=scroll_frame, anchor='nw')

        edge = ttk.Labelframe(scroll_frame, text='ME', padding=(20, 5))
        edge.pack(fill='both')

        explorer = ttk.Labelframe(scroll_frame, text='IE', padding=(20, 5))
        explorer.pack(fill='both', pady=10)

        a_in = ttk.Label(wt_canvas_a, text="Inputs:")
        a_in.place(x = 10, y = 10)

        a_alg = ttk.Label(wt_canvas_a, text="Algorithms:")
        a_alg.place(x = 10, y = 140)

        a_tsize_label = ttk.Label(wt_canvas_a, text="Training Size:")
        a_tsize_entry = ttk.Entry(wt_canvas_a)
        # a_tsize_entry.insert("1024")

        a_bsize_label = ttk.Label(wt_canvas_a, text="Batch Size:")
        a_bsize_entry = ttk.Entry(wt_canvas_a)
        # a_bsize_entry.insert("32")

        a_lr_label = ttk.Label(wt_canvas_a, text="Learning Rate:")
        a_lr_entry = ttk.Entry(wt_canvas_a)
        # a_lr_entry.insert("0.001")

        a_epochs_label = ttk.Label(wt_canvas_a, text="Number of Epochs:")
        a_epochs_entry = ttk.Entry(wt_canvas_a)
        # a_epochs_entry.insert("300")

        a_tsize_label.place(x = 10, y = 300)
        a_tsize_entry.place(x = 130, y = 300)

        a_bsize_label.place(x = 10, y = 350)
        a_bsize_entry.place(x = 130, y = 350)

        a_lr_label.place(x = 10, y = 400)
        a_lr_entry.place(x = 130, y = 400)

        a_epochs_label.place(x = 10, y = 450)
        a_epochs_entry.place(x = 130, y = 450)

        self.advanced_parameters = [a_tsize_entry, a_bsize_entry, a_lr_entry, a_epochs_entry]

        a_nn_label = ttk.Label(wt_canvas_a, text="Neural Network Design:")
        a_nn_label.place(x = 10, y = 500)

        notebook.add(simple_tab, text='Simple')
        notebook.add(intermediate_tab, text='Intermediate')
        notebook.add(advanced_tab, text='Advanced')

        # results frame
        results_frame = ttk.Frame(self)
        results_frame.grid(row=1, column=2, sticky='nsew')

        ## result cards
        cards_frame = ttk.Frame(results_frame, name='cards-frame', style='secondary.TFrame')
        cards_frame.pack(fill='both', expand='yes')

        ### privacy card
        pipLine = tkinter.PhotoImage(name='pipe', file='images/pipeLine.png')

        priv_card = ttk.Frame(cards_frame, padding=1, style='secondary.TButton')
        priv_card.pack(side='left', fill='both', padx=(10, 5), pady=10)
        priv_container = ttk.Frame(priv_card, padding=40)
        priv_container.pack(fill='both', expand='yes')

        priv_canvas = tkinter.Canvas(priv_container, border=0, highlightthickness=0, width = 575, height = 600)
        priv_canvas.pack(side='left', fill='both')

        priv_canvas.create_image(300, 300, image='privacy', anchor='center')

        self.canvas = priv_canvas
        # resultImage = priv_canvas.create_image(100, 100, image='images/SinGAN.png', anchor='center')
        self.label = Label(image=self.plotImage)
        # self.window = priv_canvas.create_window(280, 498, window=self.label, anchor='center')

        runButton = tkinter.Button(priv_canvas, width = 15, height = 3, text='RUN', anchor="c", command = self.runAI)
        runButton.place(x = 450, y = 550)

        # moveL = ttk.Label(header_frame, image=self.logo_img)
        # moveL.place(x=10, y=10)
        
        self.testMain = Tester(self, priv_canvas)

        t_s = Tester(self, wt_canvas_s)
        i_s_cl = icon.Icon("s_CLAS", 'images/scale.png')
        i_s_mnist = icon.Icon("s_MNIST", 'images/mnist.png')
        i_s_mnist.attach(t_s.canvas, y = 40)
        i_s_cl.attach(t_s.canvas, y = 170)
        i_s_sin = icon.Icon("s_SIN", 'images/sin.png')
        i_s_sin.attach(t_s.canvas, x = 100, y = 40)
        

        t_i = Tester(self, wt_canvas_i)
        i_i_gen = icon.Icon("i_GAN", 'images/gen1.png')
        i_i_sin = icon.Icon("i_SIN", 'images/sin.png')
        i_i_gen.attach(t_i.canvas, x = 100, y = 170)
        i_i_sin.attach(t_i.canvas, x = 100, y = 40)
        i_i_mnist = icon.Icon("i_MNIST", 'images/mnist.png')
        i_i_mnist.attach(t_i.canvas, y = 40)
        i_i_cl = icon.Icon("i_CLAS", 'images/scale.png')
        i_i_cl.attach(t_i.canvas, y = 170)

        t_a = Tester(self, wt_canvas_a)
        i_a_gen = icon.Icon("a_GAN", 'images/gen1.png')
        i_a_sin = icon.Icon("a_SIN", 'images/sin.png')
        i_a_mnist = icon.Icon("a_MNIST", 'images/mnist.png')
        i_a_gen.attach(t_a.canvas, x = 100, y = 170)
        i_a_sin.attach(t_a.canvas, x = 100, y = 40)
        i_a_mnist.attach(t_a.canvas, y = 40)
        i_a_cl = icon.Icon("a_CLAS", 'images/scale.png')
        i_a_cl.attach(t_a.canvas, y = 170)

        i_a_relu = icon.Icon("a_RELU", 'images/relu.png')
        i_a_relu.attach(t_a.canvas, x = 10, y = 530)

        i_a_dropout = icon.Icon("a_DROPOUT", 'images/dropout.png')
        i_a_dropout.attach(t_a.canvas, x = 100, y = 530)

        i_a_weights = icon.Icon("a_weights", 'images/weights.png')
        i_a_weights.attach(t_a.canvas, x = 190, y = 530)
        
    def runAI(self):
        icons = self.testMain.currentIcons
        if len(icons) == 0:
            return

        if "i_GAN" in icons and "i_SIN" in icons:
            self.plotPath = 'images/SinGAN.png'

            tsize = 1024
            bsize = 32
            lr = 0.001
            epochs = 300

            self.setRecommendedParameters(tsize, bsize, lr, epochs)
            self.update_plot()
            _thread.start_new_thread(self.singan, (tsize, bsize, lr, epochs))
        elif "i_GAN" in icons and "i_MNIST" in icons:
            self.plotPath = 'images/SinGAN.png'

            bsize = 32
            lr = 0.0001
            epochs = 50

            self.setRecommendedParameters("Not Used", bsize, lr, epochs)
            self.update_plot()
            _thread.start_new_thread(self.mnistgan, (bsize, lr, epochs))
        elif "a_GAN" in icons and "a_MNIST" in icons:
            self.advanced_parameters[0].insert(0, "Not Used: ")
            bsize = int(self.advanced_parameters[1].get())
            lr = float(self.advanced_parameters[2].get())
            epochs = int(self.advanced_parameters[3].get())
            
            self.plotPath = 'images/mnistGAN.png'
            self.update_plot()
            _thread.start_new_thread(self.mnistgan, (bsize, lr, epochs))
        elif "a_GAN" in icons and "a_SIN" in icons:
            tsize = int(self.advanced_parameters[0].get())
            bsize = int(self.advanced_parameters[1].get())
            lr = float(self.advanced_parameters[2].get())
            epochs = int(self.advanced_parameters[3].get())

            self.plotPath = 'images/SinGAN.png'
            self.update_plot()
            _thread.start_new_thread(self.singan, (tsize, bsize, lr, epochs))
        
        self.testMain.clearIcons()

        
    def setRecommendedParameters(self, tsize, bsize, lr, epochs):
        self.intermediate_parameters[0].insert(0, str(tsize))
        self.intermediate_parameters[1].insert(0, str(bsize))
        self.intermediate_parameters[2].insert(0, str(lr))
        self.intermediate_parameters[3].insert(0, str(epochs))
    
    def singan(self, train_size, batch_size, lr, epochs):
        gan = SinGAN(train_size, batch_size)
        gan.train(lr, epochs)

    def mnistgan(self, batch_size, lr, epochs):
        gan = mnistGAN(batch_size)
        gan.train(lr, epochs)

    def update_plot(self):
        self.plotImage = tkinter.PhotoImage(file=self.plotPath)
        self.label['image'] = self.plotImage
        self.plotImage.image = self.plotImage
        self.window = self.canvas.create_window(280, 498, window=self.label, anchor='center')
        a = self.after(1000, lambda: self.update_plot())


if __name__ == '__main__':
    Application().mainloop()