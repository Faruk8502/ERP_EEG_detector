import csv
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter import *
#_________________________________________________________________________
class LPFilter(object):
    def __init__(self, Ch, N):
        filtred = np.zeros(N)
        C = Filter_designer()
        print(C.b)
        for i in range(6, N - 8):
            filtred[i] = C.b[0] * Ch[i - 6] + C.b[1] * Ch[i - 5] +\
                         C.b[2] * Ch[i - 4] + C.b[3] * Ch[i - 3] + \
                         C.b[4] * Ch[i - 2] + C.b[5] * Ch[i - 1] +\
                         C.b[6] * Ch[i] + C.b[7] * Ch[i + 1] + \
                         C.b[8] * Ch[i + 2] + C.b[9] * Ch[i + 3] +\
                         C.b[10] * Ch[i + 4] + C.b[11] * Ch[i + 5] + \
                         C.b[12] * Ch[i + 6]
        self.filtred = filtred
class Filter_designer:
    def __init__(self):
        self.b = np.array((-0.01, -0.01, 0, 0.09, 0.15, 0.22, 0.25,
                           0.22, 0.15, 0.09, 0, -0.01, -0.01), dtype=np.single)
#_________________________________________________________________________
# class Test(Tk):
#     def __init__(self, f, N):
#         Tk.__init__(self, None)
#         self.frame=Frame(None)
#         self.frame.columnconfigure(0,weight=1)
#         self.frame.rowconfigure(0,weight=1)
#         self.frame.grid(row=0,column=0)
#
#         fig = Figure()
#         self.N = N
#         norm_arr = []
#         min0 = min(f[7: int(self.N - 8)])
#         max0 = max(f[7: int(self.N - 8)])
#         diff = max0 - min0
#         self.data_f = f
#         for i in range(6, int(self.N - 8)):
#             self.data_f[i] = (f[i] - min0 - diff/2)/diff
#         xval = np.arange(N)/10.
#         yval = self.data_f
#
#         ax1 = fig.add_subplot(111)
#         ax1.plot(xval, yval)
#
#         self.hbar = Scrollbar(self.frame,orient=HORIZONTAL)
#         self.hbar.grid(row=1, column=0)
#
#         self.canvas = FigureCanvasTkAgg(fig, master=self.frame)
#         self.canvas.get_tk_widget().config(bg='#FFFFFF', scrollregion=(0, 0, 100000, 500))
#         self.canvas.get_tk_widget().config(width=40000, height=300)
#         self.canvas.get_tk_widget().config(xscrollcommand=self.hbar.set)
#         self.canvas.get_tk_widget().grid(row=0, column=1)
#
#         self.hbar.config(command=self.canvas.get_tk_widget().xview)
class FormBuild(Tk):

    def __init__(self, window, f1, f2, f3, f4, N):
        Tk.__init__(self, None)
        self.window = window
        self.N = N
        # min0 = min(f1[7: int(self.N - 8)])
        # max0 = max(f1[7: int(self.N - 8)])
        # diff = max0 - min0
        # self.data_f = f1
        # self.Ch1 = Ch1
        # for i in range(6, int(self.N - 8)):
        #     self.data_f[i] = (f[i] - min0 - diff/2)/diff
        self.data_f1 = f1/500
        self.data_f2 = f2/500
        self.data_f3 = f3/500
        self.data_f4 = f4/500
        self.initialize_user_interface()

    def initialize_user_interface(self):
        self.width = self.window.winfo_screenwidth()
        print(self.width)
        self.height = self.window.winfo_screenheight()
        self.window.state('zoomed')
        self.window.title("EEG analys")

        self.frame1 = Frame(None)
        self.frame1.pack(ipady=0)
        self.canv1 = Canvas(self.frame1, bg="white", width=int(self.width * 2),
                             height=int(self.height/2), scrollregion=(0, 0, int(self.width * 2), 500))

        self.canv1.create_line(0, 1000, 0, 0, width=2, arrow=LAST)
        self.canv1.create_line(0, int(self.height / 16), int(self.N),
                               int(self.height / 16), width=2, arrow=LAST)

        self.canv1.create_line(0, 3 * int(self.height / 16), int(self.N),
                               3 * int(self.height / 16), width=2, arrow=LAST)

        self.canv1.create_line(0, 5 * int(self.height / 16), int(self.N),
                               5 * int(self.height / 16), width=2, arrow=LAST)

        self.canv1.create_line(0, 7 * int(self.height / 16), int(self.N),
                               7 * int(self.height / 16), width=2, arrow=LAST)

        print(self.height/4)
        for i in range(50, int(self.N - 100)):
            self.canv1.create_line(i - 1, int(self.height/16) - self.data_f1[i - 1] *
                                   int(self.height/8), i, int(self.height / 16) - self.data_f1[i] *
                                   int(self.height / 8), width=2, fill="blue")
            self.canv1.create_line(i - 1, 3 * int(self.height / 16) - self.data_f2[i - 1] *
                                   int(self.height / 8), i, 3 * int(self.height / 16) - self.data_f2[i] *
                                   int(self.height / 8), width=2, fill="blue")
            self.canv1.create_line(i - 1, 5 * int(self.height / 16) - self.data_f3[i - 1] *
                                   int(self.height / 8), i, 5 * int(self.height / 16) - self.data_f3[i] *
                                   int(self.height / 8), width=2, fill="blue")
            self.canv1.create_line(i - 1, 7 * int(self.height / 16) - self.data_f4[i - 1] *
                                   int(self.height / 8), i, 7 * int(self.height / 16) - self.data_f4[i] *
                                   int(self.height / 8), width=2, fill="blue")
        self.canv1.pack(side='top')
        # f = Figure(figsize=(1, 1), dpi=100)
        # a = f.add_subplot(111)
        # a.plot(self.data_f)
        #
        # self.canv2 = FigureCanvasTkAgg(f, master=self.frame1)
        # self.canv2.get_tk_widget().pack(side=TOP)
        # self.canv2.get_tk_widget().config(width=int(self.width * 2), height=int(self.height/5))
        # self.canv2.get_tk_widget().config(bg='#FFFFFF', scrollregion=(0, 0, int(self.width * 2), 500))
        #
        self.sb = Scrollbar(self.frame1, orient='horizontal')
        self.canv1.config(xscrollcommand=self.sb.set)
        self.sb.pack(side='bottom', fill='x')
        self.sb.config(command=self.canv1.xview)

    def normalize(self, arr, t_min, t_max):
        norm_arr = []
        diff = t_max - t_min
        diff_arr = max(arr[7: int(self.N - 8)]) - min(arr[7: int(self.N - 8)])
        for i in arr:
            temp = (((i - min(arr)) * diff) / diff_arr) + t_min
            norm_arr.append(temp)
        return norm_arr
#_________________________________________________________________________
with open("s01_ex01_s01.csv", 'r', newline='') as r_file:
    file_reader = csv.reader(r_file, delimiter=",")
    data = np.empty([24002, 5], dtype=object)
    k = 0
    for row in file_reader:
        k = k + 1
        data[k] = row
print(data)

N = np.shape(data)
Fs = 200
T = 1/Fs
tmax = (N[0] - 2) * T
t = np.arange(0, tmax, T)
print(N[0])

Ch1 = np.zeros(N[0] - 2)
Ch2 = np.zeros(N[0] - 2)
Ch3 = np.zeros(N[0] - 2)
Ch4 = np.zeros(N[0] - 2)

for i in range(0, N[0] - 2):
    Ch1[i] = float(data[i + 2, 1])
    Ch2[i] = float(data[i + 2, 2])
    Ch3[i] = float(data[i + 2, 3])
    Ch4[i] = float(data[i + 2, 4])

Filtred = LPFilter(Ch1, N[0])
f1 = Filtred.filtred

Filtred = LPFilter(Ch2, N[0])
f2 = Filtred.filtred

Filtred = LPFilter(Ch3, N[0])
f3 = Filtred.filtred

Filtred = LPFilter(Ch4, N[0])
f4 = Filtred.filtred

app = FormBuild(Tk(), f1, f2, f3, f4, N[0])
app.window.mainloop()

# if __name__ == '__main__':
#     app = Test(f, N[0])
#     app.mainloop()
# plt.subplot(211), plt.plot(t[7:1000], Ch1[7:1000]), plt.title('Исходный')
# plt.xticks([]), plt.yticks([])
# plt.subplot(212), plt.plot(t[7:1000], Filtred.filtred[7:1000]), plt.title('Отфильтрованный')
# plt.xticks([]), plt.yticks([])
# plt.show()

