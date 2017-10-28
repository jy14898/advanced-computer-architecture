from Tkinter import *
from functools import partial

class Component(Frame):
    def __init__(self, parent, component, x, y):
        Frame.__init__(self, parent, bd=1, relief="solid", background="white")

        self.parent = parent

        self.pos = (x,y)
        self.place(x=x,y=y)
        self.bindtags((component.name(),))

        name = Label(self, text=component.name(), bg ="white", font=("TkDefaultFont",8))
        name.grid(row=0,columnspan=2, ipadx=1, ipady=1, sticky="nsew")
        name.bindtags((component.name(),))

        type_ = Label(self, text="({})".format(type(component).__name__), bg ="white", font=("TkDefaultFont",8))
        type_.grid(row=1,columnspan=2, ipadx=1, ipady=1, sticky="nsew")
        type_.bindtags((component.name(),))

        self.input_labels = {}
        for row, input_key in enumerate(component.input_keys):
            self.input_labels[input_key] = Label(self, text=input_key, anchor="w", bg="light sky blue", font=("TkDefaultFont",8))
            self.input_labels[input_key].grid(row=row+2, column=0, ipadx=1, ipady=1, sticky="nsew")
            self.input_labels[input_key].bindtags((component.name(),))

        self.output_labels = {}
        for row, output_key in enumerate(component.output_keys):
            self.output_labels[output_key] = Label(self, text=output_key, anchor="w", bg="sandy brown", font=("TkDefaultFont",8))
            self.output_labels[output_key].grid(row=row+2, column=1, ipadx=1, ipady=1, sticky="nsew")
            self.output_labels[output_key].bindtags((component.name(),))


        self.drag_start_x= 0
        self.drag_start_y= 0
        self.bind_class(component.name(), "<Button-1>", self.drag_start)
        self.bind_class(component.name(), "<ButtonRelease-1>", self.drag_stop)
        self.bind_class(component.name(), "<B1-Motion>", self.drag_move)

        self.connections = []
        self.dragging = False

        self.drag_preview = parent.create_rectangle([0,0,0,0],state="hidden")

    def drag_start(self, event):
        self.drag_start_x = event.x
        self.drag_start_y = event.y
        self.dragging = True

        self.parent.itemconfig(self.drag_preview,state="normal")

    def drag_stop(self, event):
        self.dragging = False

        x= self.winfo_x()-self.drag_start_x+event.x
        y= self.winfo_y()-self.drag_start_y+event.y
        self.place(x=x, y=y)
        self.pos = (x,y)
        for connection in self.connections:
            connection()

        self.parent.itemconfig(self.drag_preview,state="hidden")

    def drag_move(self, event):
        if self.dragging:
            x= self.winfo_x()-self.drag_start_x+event.x
            y= self.winfo_y()-self.drag_start_y+event.y
            w = self.winfo_width()
            h = self.winfo_height()
            self.parent.coords(self.drag_preview, x, y, x + w, y + h)


    def subscribe_connection(self, update):
        self.connections.append(update)



def display(processor_cco):

    root = Tk()
    canvas = Canvas(root,width=1600, height=800, highlightthickness=0)
    canvas.pack(fill=BOTH, expand=YES)

    x = 100
    component_widgets = {}
    for component in processor_cco.components:
        c = Component(canvas, component, x=x, y=x)

        component_widgets[component] = c
        x += 10

    root.update()

    for ((out_component,out_key),(in_component,in_key)) in processor_cco.connections:
        l = canvas.create_line(0,0,0,0, smooth=1, arrow="last")

        def update(line, bi, bo, in_, out, end):
            ox = bo.pos[0] + bo.winfo_width()
            oy = bo.pos[1] + out.winfo_y() + 8
            ix = bi.pos[0]
            iy = bi.pos[1] + in_.winfo_y() + 8
            if end == "start":
                canvas.coords(line, ox, oy, ox + 10, oy, ix - 10, iy, ix, iy)
            else:
                canvas.coords(line, ox, oy, ox + 10, oy, ix - 10, iy, ix, iy)

        component_widgets[out_component].subscribe_connection(partial(update, l, component_widgets[in_component], component_widgets[out_component], component_widgets[in_component].input_labels[in_key], component_widgets[out_component].output_labels[out_key], "start"))
        component_widgets[in_component].subscribe_connection(partial(update, l, component_widgets[in_component], component_widgets[out_component], component_widgets[in_component].input_labels[in_key], component_widgets[out_component].output_labels[out_key], "end"))

        # .grid_info()

    mainloop()
