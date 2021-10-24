'''
    Adapted from python 3.11.0 alpha
'''

import tkinter
from dnd import *
from icon import *

class Tester:
    def __init__(self, root, canvas):
        self.top = root
        self.canvas = canvas
        self.canvas.pack(fill="both", expand=1)
        self.canvas.dnd_accept = self.dnd_accept
        self.currentIcons = []
    
    def whichIcon(self):
        return list(set(self.currentIcons))
    
    def clearIcons(self):
        self.currentIcons = []

    # def addIcon(self, name):
    #     self.currentIcons.append(name)
    
    # def removeIcon(self, name):
    #     self.currentIcons.remove(name)

    def dnd_accept(self, source, event):
        return self

    def dnd_enter(self, source, event):
        
        self.currentIcons.append(source.name)

        self.canvas.focus_set() # Show highlight border
        x, y = source.where(self.canvas, event)
        x1, y1, x2, y2 = source.canvas.bbox(source.id)
        dx, dy = x2-x1, y2-y1
        self.dndid = self.canvas.create_rectangle(x, y, x+dx, y+dy)
        self.dnd_motion(source, event)

    def dnd_motion(self, source, event):
        x, y = source.where(self.canvas, event)
        x1, y1, x2, y2 = self.canvas.bbox(self.dndid)
        self.canvas.move(self.dndid, x-x1, y-y1)

    def dnd_leave(self, source, event):
        # print(source.name)
        # self.currentIcons.remove(source.name)

        self.top.focus_set() # Hide highlight border
        self.canvas.delete(self.dndid)
        self.dndid = None

    def dnd_commit(self, source, event):
        self.dnd_leave(source, event)
        x, y = source.where(self.canvas, event)
        source.attach(self.canvas, x, y)


# def test():
#     root = tkinter.Tk()
#     root.geometry("+1+1")
#     tkinter.Button(command=root.quit, text="Quit").pack()
#     t1 = Tester(root)
#     t1.top.geometry("+1+60")
#     t2 = Tester(root)
#     t2.top.geometry("+120+60")
#     t3 = Tester(root)
#     t3.top.geometry("+240+60")
#     i1 = Icon("ICON1", 'images/Body.png')
#     i2 = Icon("ICON2", 'images/Body.png')
#     i3 = Icon("ICON3", 'images/Body.png')
#     i1.attach(t1.canvas)
#     i2.attach(t2.canvas)
#     i3.attach(t3.canvas)
#     root.mainloop()


# if __name__ == '__main__':
#     test()