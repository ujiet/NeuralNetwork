# encoding: utf-8

import hopfield as hop
import preprocess as pre
import matplotlib.pyplot as plt

import tkinter as tk
from tkinter import ttk
from PIL import ImageTk, Image

class gui:
    
    def __init__(self, window):
        
        self.dataset = ['Basic', 'Bonus']
        self.init_condition = {}

        # window
        self.window = window
        window.title('Neural Network Hw3')
        window.geometry('1280x720')
        window.resizable(False, False)

        # image
        self.img = ImageTk.PhotoImage(Image.open("start.png"))
        self.panel = ttk.Label(window, image = self.img)
        self.panel.place(x = 10, y = 10)

        # combolist
        self.comvalue = tk.StringVar()
        self.comboxlist = ttk.Combobox(window, textvariable = self.comvalue) 
        self.comboxlist['values'] = self.dataset
        self.comboxlist.current()  
        self.comboxlist.bind("<<ComboboxSelected>>", self.combo_select) 
        self.comboxlist.place(x = 10, y = 670)

        # box
        self.var_n = tk.IntVar()
        self.var_n.set(5)
        self.entry_n = ttk.Entry(window, textvariable = self.var_n)
        self.entry_n.place(x = 210, y = 670)

        self.var_noise_ratio = tk.DoubleVar()
        self.var_noise_ratio.set(0.25)
        self.entry_noise_ratio = ttk.Entry(window, textvariable = self.var_noise_ratio)
        self.entry_noise_ratio.place(x = 610, y = 670)

        # checkbutton
        self.var_ckb = tk.IntVar()
        self.noise_ckb = ttk.Checkbutton(window, text = 'Noise', variable = self.var_ckb)
        self.noise_ckb.place(x = 510, y = 670)

        self.var_ckb_converge = tk.IntVar()
        self.converge_ckb = ttk.Checkbutton(window, text = 'Converge', variable = self.var_ckb_converge)
        self.converge_ckb.place(x = 410, y = 670)

        # label
        ttk.Label(window, text='選擇資料集').place(x = 10, y = 640)
        ttk.Label(window, text='設定回想次數').place(x = 210, y = 640)
        ttk.Label(window, text='設定雜訊比例').place(x = 610, y = 640)

        # button
        self.btn_train_start = ttk.Button(window, text = 'Start', 
                                      command = lambda: self.start(self.panel))
        self.btn_train_start.place(x = 910, y = 665)

    def combo_select(self, dataset_name): 
        self.init_condition['dataset_name'] = self.comboxlist.get()
        self.init_condition['train_name'] = self.comboxlist.get() + '_Training.txt'
        self.init_condition['test_name'] = self.comboxlist.get() + '_Testing.txt'

    def start(self, panel):

        # get initial condition
        self.init_condition['n'] = int(self.entry_n.get())
        self.converge = self.var_ckb_converge.get()
        self.ratio = float(self.entry_noise_ratio.get())
        if self.ratio > 1: 
            self.ratio = 1
        elif self.ratio < 0:
            self.ratio = 0
        
        # training
        pre_result = pre.preprocess(self.init_condition['train_name']) 
        train_inputs = pre_result['data']

        nn = hop.Hopfield(pre_result)
        nn.calculate_W(train_inputs)
        nn.calculate_threshold()

        # testing
        if self.var_ckb.get():
            noise_test_name = pre.noise(self.init_condition['train_name'], self.ratio)
            test_result = pre.preprocess(noise_test_name)
        else:
            test_result = pre.preprocess(self.init_condition['test_name'])
        test_inputs = test_result['data']

        # ploting
        if self.init_condition['dataset_name'] == 'Basic':
            flag = 'basic'
            graph_num = 3
        else:
            flag = 'bonus'
            graph_num = 15

        plt.figure(figsize = (12.5, 6))

        for j in range(graph_num):
            if flag == 'basic':
                axs = plt.subplot(1, 3, j + 1)
            else: 
                axs = plt.subplot(3, 5, j + 1) 

            x = test_inputs[j]

            if self.converge:
                i = 0
                while True:
                    i += 1
                    y = nn.update_x(x)
                    if nn.equalty(x, y):
                        break
                    x = y
            else:
                for i in range(self.init_condition['n']):
                    y = nn.update_x(x)
                    x = y

            nn.calculate_accuracy(train_inputs[j], x)
            nn.graph(x)

            if self.converge:
                axs.set_title('accuracy = %.2f, converge at %d' % (nn.accuracy, i), fontsize = 10)
            else:
                axs.set_title('accuracy = %.2f, converge at %d' % (nn.accuracy, i + 1), fontsize = 10)
        plt.tight_layout()
        plt.savefig('result.png')
        plt.close()  
        
        # show image
        self.img = ImageTk.PhotoImage(Image.open('result.png'))
        panel.configure(image = self.img)
        panel.image = self.img
        

root = tk.Tk()
my_gui = gui(root)
root.mainloop()