import PySimpleGUI as sg
import matplotlib.pyplot as plt
from program1 import GA

class InputWindow:

    def __init__(self):
        self.layout = [[sg.Text('Shape Classification', size=(40, 1), justification='center', font='Helvetica 20')],
                       [sg.Text("There are 4 shapes:")],
                       [sg.Text("circle, square, triangle, star")],
                       [sg.Input(default_text="square", key="-SHAPE-")],
                       [sg.Spin([i for i in range(0, 500, 1)], initial_value=20000, key="-GENERATIONS-")],
                       [sg.Button('Start', size=(10, 1), font='Helvetica 14')],
                       [sg.Button('Show', size=(10, 1), font='Helvetica 14')]]
        self.interrupted = False
    def show(self):
        window = sg.Window('Main Window', self.layout, location=(800, 400))

        while True:
            event, values = window.read()
            if event == 'Show' or event == sg.WIN_CLOSED:
                self.interrupted = True
                break
            if event == "Start":
                self.generations = int(values["-GENERATIONS-"])
                self.shape = values["-SHAPE-"]
                algorithm = GA(self.generations, self.shape)
                algorithm.run()
                break
        window.close()
