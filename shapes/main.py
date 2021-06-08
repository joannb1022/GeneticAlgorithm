from gui import InputWindow

inputWindow = InputWindow()
inputWindow.show()

if inputWindow.interrupted is True:
    exit(1)
