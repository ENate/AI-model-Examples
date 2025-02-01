# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 01:35:31 2015

@author: nath
"""

from Tkinter import *
import Pmw
from scipy import *


class Window(Tk):
    def __init__(self, parent):
        Tk.__init__(self, parent)
        self.parent = parent
        self.initialize()

    def initialize(self):
        self.geometry("600x600")
        # self.pack(expand = YES, fill = BOTH)
        self.button1 = Button(
            self,
            text="CLASSIC SIR ",
            width=35,
            height=2,
            bg="blue",
            fg="green",
            font=("georgia, 18"),
            command=self.classic_SIR,
        )
        self.button1.pack(side=TOP)
        self.button2 = Button(
            self,
            text="SIR WITH VITAL DYNAMICS",
            width=35,
            height=2,
            bg="blue",
            fg="green",
            font=("georgia, 18"),
            command=self.Vital_Dynamics,
        )
        self.button2.pack(side=TOP)
        self.button3 = Button(
            self,
            text="SIR WITH VERTICAL TRANSMISSION",
            width=35,
            height=2,
            bg="blue",
            fg="green",
            font=("georgia, 18"),
            command=self.Vertical_Transmission,
        )
        self.button3.pack(side=TOP)
        self.button4 = Button(
            self,
            text="SIR WITH CARRIERS",
            width=35,
            height=2,
            bg="blue",
            fg="green",
            font=("georgia, 18"),
            command=self.Carriers,
        )
        self.button4.pack(side=TOP)
        self.button5 = Button(
            self,
            text="SIR WITH TEMPORARY IMMUNITY ",
            width=35,
            height=2,
            bg="blue",
            fg="green",
            font=("georgia, 18"),
            command=self.Temporary_Immunity,
        )
        self.button5.pack(side=TOP)
        self.button6 = Button(
            self,
            text="SIR WITH HOST AND VECTORS",
            width=35,
            height=2,
            bg="blue",
            fg="green",
            font=("georgia, 18"),
            command=self.Host_Vector,
        )
        self.button6.pack(side=TOP)
        self.button7 = Button(
            self,
            text="EXIT WINDOW",
            width=35,
            height=2,
            bg="red",
            fg="green",
            font=("georgia, 18"),
            command=quit,
        )
        self.button7.pack(side=TOP)

    def hide(self):
        self.withdraw()
        # self.wButton = Button(self, text='text', command = self.OnButtonClick)
        # self.wButton.pack()

    # -------------------------------------------------------------------------------------------------------------------------------------------------------
    #  SIR with Vertical Dynamics
    # -------------------------------------------------------------------------------------------------------------------------------------------------------
    def classic_SIR(self):
        self.hide()
        cLFrame = Toplevel()
        cLFrame.title("THE CLASSIC SIR")
        cLFrame.geometry("700x800")
        # self.top.transient(self)
        g = Pmw.Blt.Graph(cLFrame)
        g.pack(expand=1, fill="both")
        E = zeros([2, 1])
        A = zeros([2, 2])
        C = zeros([2, 1])
        A = zeros([2, 2])
        B = zeros([2, 2])
        E = zeros([2, 1])
        C = zeros([2, 1])
        # -------------------------------------------------------------------------------------------------------------------------------------------------------
        # Text entry
        # -------------------------------------------------------------------------------------------------------------------------------------------------------

        sus = Pmw.EntryField(
            cLFrame,
            labelpos="w",
            value="0.0",
            label_text="Initial Susceptibles :",
            validate={"validator": "real"},
        )
        sus.pack(expand=1, padx=5, pady=5)
        Infec = Pmw.EntryField(
            cLFrame,
            labelpos="w",
            value="0.0",
            label_text="Initial Infectives Io:",
            validate={"validator": "real"},
        )
        Infec.pack(expand=1, padx=7, pady=5)
        contact = Pmw.EntryField(
            cLFrame,
            labelpos="w",
            value="0.0",
            label_text="Contact Rate (k) :",
            validate={"validator": "real"},
        )
        contact.pack(expand=1, padx=5, pady=5)
        Removal = Pmw.EntryField(
            cLFrame,
            labelpos="w",
            value="0.0",
            label_text=" Removal  Rate (l) :",
            validate={"validator": "real"},
        )
        Removal.pack(expand=1, padx=10, pady=5)
        time_interval = Pmw.EntryField(
            cLFrame,
            labelpos="w",
            value="0",
            label_text=" Time    Interval (t) :",
            validate={"validator": "numeric"},
        )
        time_interval.pack(expand=1, padx=5, pady=5)
        # -------------------------------------------------------------------------------------------------------------------------------------------------------

        # self.top.transient(self)
        # self.wButton.config(state='disabled')
        def retrive():
            k1 = contact.get()
            k = float(k1)
            ggamma = float(Removal.get())
            N = int(time_interval.get())
            s0 = float(sus.get())
            I0 = float(Infec.get())
            A[0][0] = 0
            A[0][1] = -k
            A[1][0] = k
            A[1][1] = 0
            B[0][0] = 0
            B[0][1] = 0
            B[1][0] = 0
            B[1][1] = 0
            E[0][0] = 0
            E[1][0] = -ggamma
            C[0][0] = 0
            C[1][0] = 0

            def f1(s1, I1):
                s_dot = (
                    s1 * E[0][0]
                    + A[0][0] * (s1 ** 2)
                    + A[0][1] * s1 * I1
                    + B[0][0] * s1
                    + B[0][1] * I1
                    + C[0][0]
                )
                return s_dot

            def f2(s1, I1):
                I_dot = (
                    I1 * E[1][0]
                    + A[1][1] * (I1 ** 2)
                    + A[1][0] * s1 * I1
                    + B[1][0] * s1
                    + B[1][1] * I1
                    + C[1][0]
                )
                return I_dot

            # =========RATE OF CHANGE OF THE MODEL WITH RESPECT TO TIME AND DECLARATON OF CONSTANTS==============================================================
            t = arange(0, N, 0.02455)
            s1 = zeros(len(t))
            I1 = zeros(len(t))
            s1[0] = s0
            I1[0] = I0
            h = 0.01

            # ==========ITERATION FOR A GIVEN TIME INTERVAL=======================================================================================================

            for n in range(len(t) - 1):
                s1[n + 1] = s1[n] + h * (f1(s1[n], I1[n]))
                I1[n + 1] = I1[n] + h * (f2(s1[n], I1[n]))
            x_ = tuple(t)
            y_ = tuple(s1)
            z_ = tuple(I1)
            g.configure(
                title="A graph of Susceptible and Infectives  against time for the classic SIR"
            )
            g.line_create(
                "S(t) Vs t", xdata=x_, color="red", ydata=y_, linewidth=2, symbol=""
            )
            g.line_create(
                "I(t) Vs t", xdata=x_, color="orange", ydata=z_, linewidth=2, symbol=""
            )

        def postscript():
            g.postscript_output(fileName="HelloUser1.ps", decorations="no")

        def newFile():
            for name in g.element_names():
                g.element_delete(name)
            contact.setentry("")
            Removal.setentry("")
            time_interval.setentry("")
            Infec.setentry("")
            sus.setentry("")

        # -------------------------------------------------------------------------------------------------------------------------------------------------------
        buttons = Pmw.ButtonBox(cLFrame, labelpos="w", label_text="Options")
        # buttons = Pmw.ButtonBox(self, labelpos='w', label_text='Options')
        buttons.pack(fill="both", expand=1, padx=10, pady=10)
        buttons.add("GRID", bg="green", command=g.grid_toggle)
        buttons.add("PLOT", bg="green", command=retrive)
        buttons.add("CLEAR", bg="green", command=newFile)
        buttons.add("SAVE", bg="green", command=postscript)
        handler = lambda: self.onCloseOtherFrame(cLFrame)
        buttons.add("BACK", bg="red", command=handler)

    # -------------------------------------------------------------------------------------------------------------------------------------------------------
    #  SIR with Vertical Dynamics
    # -------------------------------------------------------------------------------------------------------------------------------------------------------
    def Vital_Dynamics(self):
        self.hide()
        vDFrame = Toplevel()
        vDFrame.title("SIR WITH VITAL DYNAMICS")
        vDFrame.geometry("700x800")
        # self.top.transient(self)
        g = Pmw.Blt.Graph(vDFrame)
        g.pack(expand=1, fill="both")
        E = zeros([2, 1])
        A = zeros([2, 2])
        C = [2, 1]
        A = zeros([2, 2])
        B = zeros([2, 2])
        E = zeros([2, 1])
        C = zeros([2, 1])

        # ===============CREATE, PACK AND VALIDATE THE TEXTFIELD FOR DATA ENTRY================================================================================

        sus = Pmw.EntryField(
            vDFrame,
            labelpos="w",
            value="0.0",
            label_text="Initial Susceptibles :",
            validate={"validator": "real"},
        )
        sus.pack(expand=1, padx=5, pady=5)
        Infec = Pmw.EntryField(
            vDFrame,
            labelpos="w",
            value="0.0",
            label_text="Initial Infectives :",
            validate={"validator": "real"},
        )
        Infec.pack(expand=1, padx=5, pady=5)
        contact = Pmw.EntryField(
            vDFrame,
            labelpos="w",
            value="0.0",
            label_text="Contact Rate (k) :",
            validate={"validator": "real"},
        )
        contact.pack(expand=1, padx=5, pady=5)
        Removal = Pmw.EntryField(
            vDFrame,
            labelpos="w",
            value="0.0",
            label_text=" Removal  Rate  :",
            validate={"validator": "real"},
        )
        Removal.pack(expand=1, padx=10, pady=5)
        time_interval = Pmw.EntryField(
            vDFrame,
            labelpos="w",
            value="0",
            label_text=" Time    Interval :",
            validate={"validator": "numeric"},
        )
        time_interval.pack(expand=1, padx=5, pady=5)
        death_rate = Pmw.EntryField(
            vDFrame,
            labelpos="w",
            value="0.0",
            label_text=" Death  Rate (d) :",
            validate={"validator": "real"},
        )
        death_rate.pack(expand=1, padx=10, pady=5)
        immunity = Pmw.EntryField(
            vDFrame,
            labelpos="w",
            value="0.00",
            label_text="  Immunity Rate:",
            validate={"validator": "real"},
        )
        immunity.pack(expand=1, padx=10, pady=10)
        # -------------------------------------------------------------------------------------------------------------------------------------------------------

        def retrive():
            k1 = contact.get()
            k = float(k1)
            ddelta = float(death_rate.get())
            aalpha = float(immunity.get())
            ggamma = float(Removal.get())
            N = int(time_interval.get())
            s0 = float(sus.get())
            I0 = float(Infec.get())
            A[0][0] = 0
            A[0][1] = -k
            A[1][0] = k
            A[1][1] = 0
            B[0][0] = 0
            B[0][1] = 0
            B[1][0] = 0
            B[1][1] = 0
            E[0][0] = -ddelta
            E[1][0] = -(ggamma + ddelta)
            C[0][0] = ddelta
            C[1][0] = 0

            # =========GENERAL DECLARATION SECTION FOR THE BILINEAR FORM=============================================================================================

            def f1(s1, I1):
                s_dot = (
                    s1 * E[0][0]
                    + A[0][0] * (s1 ** 2)
                    + A[0][1] * s1 * I1
                    + B[0][0] * s1
                    + B[0][1] * I1
                    + C[0][0]
                )
                return s_dot

            def f2(s1, I1):
                I_dot = (
                    I1 * E[1][0]
                    + A[1][1] * (I1 ** 2)
                    + A[1][0] * s1 * I1
                    + B[1][0] * s1
                    + B[1][1] * I1
                    + C[1][0]
                )
                return I_dot

            # =========RATE OF CHANGE OF THE MODEL WITH RESPECT TO TIME AND DECLARATON OF CONSTANTS==================================================================
            t = arange(0, N, 0.02455)
            s1 = zeros(len(t))
            I1 = zeros(len(t))
            s1[0] = s0
            I1[0] = I0
            h = 0.01

            # ==========ITERATION FOR A GIVEN TIME INTERVAL========================================================================================================

            for n in range(len(t) - 1):
                s1[n + 1] = s1[n] + h * (f1(s1[n], I1[n]))
                I1[n + 1] = I1[n] + h * (f2(s1[n], I1[n]))

            # =========PLOTTING AND FORMATTING=====================================================================================================================

            x_ = tuple(t)
            y_ = tuple(s1)
            z_ = tuple(I1)
            g.configure(
                title="Susceptible and Infectives  against time for SIR with vital dynamics"
            )
            g.line_create(
                "S(t) Vs t", xdata=x_, color="blue", ydata=y_, linewidth=2, symbol=""
            )
            g.line_create(
                "I(t) Vs t", xdata=x_, color="orange", ydata=z_, linewidth=2, symbol=""
            )

        def postscript():
            g.postscript_output(fileName="HelloUser2.eps", decorations="no")

        def newFile():
            for name in g.element_names():
                g.element_delete(name)
            contact.setentry("")
            Removal.setentry("")
            time_interval.setentry("")
            Infec.setentry("")
            sus.setentry("")
            immunity.setentry("")
            death_rate.setentry("")

        # =========CREATE AND PACK BUTTONS, LABELS USING Pmw PACKAGE IN PYTHON===============================================================================

        buttons = Pmw.ButtonBox(vDFrame, labelpos="w", label_text="Options")
        buttons = Pmw.ButtonBox(vDFrame, labelpos="w", label_text="Options")
        buttons.pack(fill="both", expand=1, padx=10, pady=10)
        buttons.add("GRID", bg="green", command=g.grid_toggle)
        buttons.add("PLOT", bg="green", command=retrive)
        buttons.add("CLEAR", bg="green", command=newFile)
        buttons.add("SAVE", bg="green", command=postscript)
        handler = lambda: self.onCloseOtherFrame1(vDFrame)
        buttons.add("QUIT", bg="red", command=handler)

    # --------------------------------------------------------------------------------------------------------------------------------------------------------
    # SIR with Vertical_Transmission
    # --------------------------------------------------------------------------------------------------------------------------------------------------------
    def Vertical_Transmission(self):
        self.hide()
        vTFrame = Toplevel()
        vTFrame.title("SIR WITH VERTICAL TRANSMISSION")
        vTFrame.geometry("700x800")
        # self.top.transient(self)
        g = Pmw.Blt.Graph(vTFrame)
        g.pack(expand=1, fill="both")
        E = zeros([2, 1])
        A = zeros([2, 2])
        C = zeros([2, 1])
        A = zeros([2, 2])
        B = zeros([2, 2])
        E = zeros([2, 1])
        C = zeros([2, 1])
        # --------------------------------------------------------------------------------------------------------------------------------------------------------
        sus = Pmw.EntryField(
            vTFrame,
            labelpos="w",
            value="0.0",
            label_text="Initial Susceptible",
            validate={"validator": "real"},
        )
        sus.pack(expand=1, padx=5, pady=5)
        Infec = Pmw.EntryField(
            vTFrame,
            labelpos="w",
            value="0.0",
            label_text="Initial Infective Io:",
            validate={"validator": "real"},
        )
        Infec.pack(expand=1, padx=5, pady=5)
        contact = Pmw.EntryField(
            vTFrame,
            labelpos="w",
            value="0.0",
            label_text="Contact Rate - k ",
            validate={"validator": "real"},
        )
        contact.pack(expand=1, padx=5, pady=5)
        Removal = Pmw.EntryField(
            vTFrame,
            labelpos="w",
            value="0.0",
            label_text=" Removal  Rate ",
            validate={"validator": "real"},
        )
        Removal.pack(expand=1, padx=10, pady=5)
        time_interval = Pmw.EntryField(
            vTFrame,
            labelpos="w",
            value="0",
            label_text=" Time    Interval",
            validate={"validator": "numeric"},
        )
        time_interval.pack(expand=1, padx=5, pady=5)
        birth_rate = Pmw.EntryField(
            vTFrame,
            labelpos="w",
            value="0.0",
            label_text=" Birth  Rate - b :",
            validate={"validator": "real"},
        )
        birth_rate.pack(expand=1, padx=10, pady=5)
        immunity1 = Pmw.EntryField(
            vTFrame,
            labelpos="w",
            value="0.00",
            label_text="  Immunity Rate ",
            validate={"validator": "real"},
        )
        immunity1.pack(expand=1, padx=10, pady=10)
        infected_rate = Pmw.EntryField(
            vTFrame,
            labelpos="w",
            value="0.0",
            label_text="Infected Birth - Ib",
            validate={"validator": "real"},
        )
        infected_rate.pack(expand=1, padx=10, pady=5)
        vertical = Pmw.EntryField(
            vTFrame,
            labelpos="w",
            value="0.00",
            label_text="  Vertical Rate - v",
            validate={"validator": "real"},
        )
        vertical.pack(expand=1, padx=10, pady=10)
        probability = Pmw.EntryField(
            vTFrame,
            labelpos="w",
            value="0.00",
            label_text="Probability loss -p",
            validate={"validator": "real"},
        )
        probability.pack(expand=1, padx=10, pady=10)
        # --------------------------------------------------------------------------------------------------------------------------------------------------------
        # self.top.transient(self)
        # self.wButton.config(state='disabled')
        def retrive():
            k1 = contact.get()
            k = float(k1)
            ggamma = float(Removal.get())
            N = int(time_interval.get())
            s0 = float(sus.get())
            I0 = float(Infec.get())
            m = float(immunity1.get())
            v = float(vertical.get())
            b_prime = float(infected_rate.get())
            b = float(birth_rate.get())
            p = float(probability.get())
            A[0][0] = 0
            A[0][1] = -k
            A[1][0] = k
            A[1][1] = 0
            B[0][0] = (m - 1) * b + (p * b_prime + ggamma)
            B[0][1] = 0
            B[1][0] = 0
            B[1][1] = 0
            E[0][0] = -b - ggamma
            E[1][0] = p * b_prime - v
            C[0][0] = b * (1 - m) + (ggamma)
            C[1][0] = 0

            def f1(s1, I1):
                s_dot = (
                    s1 * E[0][0]
                    + A[0][0] * (s1 ** 2)
                    + A[0][1] * s1 * I1
                    + B[0][0] * s1
                    + B[0][1] * I1
                    + C[0][0]
                )
                return s_dot

            def f2(s1, I1):
                I_dot = (
                    I1 * E[1][0]
                    + A[1][1] * (I1 ** 2)
                    + A[1][0] * s1 * I1
                    + B[1][0] * s1
                    + B[1][1] * I1
                    + C[1][0]
                )
                return I_dot

            # =========RATE OF CHANGE OF THE MODEL WITH RESPECT TO TIME AND DECLARATON OF CONSTANTS==================================
            t = arange(0, N, 0.02455)
            s1 = zeros(len(t))
            I1 = zeros(len(t))
            s1[0] = s0
            I1[0] = I0
            h = 0.01

            # ==========ITERATION FOR A GIVEN TIME INTERVAL=========================

            for n in range(len(t) - 1):
                s1[n + 1] = s1[n] + h * (f1(s1[n], I1[n]))
                I1[n + 1] = I1[n] + h * (f2(s1[n], I1[n]))
            x_ = tuple(t)
            y_ = tuple(s1)
            z_ = tuple(I1)
            g.configure(
                title="Susceptible and Infectives  against time for SIR with Vertical Transmission"
            )
            g.line_create(
                "S(t) Vs t", xdata=x_, color="red", ydata=y_, linewidth=2, symbol=""
            )
            g.line_create(
                "I(t) Vs t", xdata=x_, color="orange", ydata=z_, linewidth=2, symbol=""
            )

        def postscript():
            g.postscript_output(fileName="HelloUser1.ps", decorations="no")

        def newFile():
            for name in g.element_names():
                g.element_delete(name)
            contact.setentry("")
            Removal.setentry("")
            time_interval.setentry("")
            Infec.setentry("")
            sus.setentry("")
            immunity1.setentry("")
            birth_rate.setentry("")
            vertical.setentry("")
            infected_rate.setentry("")
            probability.setentry("")

        # --------------------------------------------------------------------------------------------------------------------------------------------------------
        buttons = Pmw.ButtonBox(vTFrame, labelpos="w", label_text="Options")
        # buttons = Pmw.ButtonBox(self, labelpos='w', label_text='Options')
        buttons.pack(fill="both", expand=1, padx=10, pady=10)
        buttons.add("GRID", bg="green", command=g.grid_toggle)
        buttons.add("PLOT", bg="green", command=retrive)
        buttons.add("CLEAR", bg="green", command=newFile)
        buttons.add("SAVE", bg="green", command=postscript)
        handler = lambda: self.onCloseOtherFrame2(vTFrame)
        buttons.add("QUIT", bg="red", command=handler)

    # --------------------------------------------------------------------------------------------------------------------------------------------------------
    # SIR with Carriers
    # --------------------------------------------------------------------------------------------------------------------------------------------------------
    def Carriers(self):
        self.hide()
        cFrame = Toplevel()
        cFrame.title("SIR WITH CARRIERS")
        cFrame.geometry("700x800")
        # self.top.transient(self)
        g = Pmw.Blt.Graph(cFrame)
        g.pack(expand=1, fill="both")
        E = zeros([2, 1])
        A = zeros([2, 2])
        C = zeros([2, 1])
        A = zeros([2, 2])
        B = zeros([2, 2])
        E = zeros([2, 1])
        C = zeros([2, 1])

        # --------------------------------------------------------------------------------------------------------------------------------------------------------
        sus = Pmw.EntryField(
            cFrame,
            labelpos="w",
            value="0.0",
            label_text="Initial Susceptible ",
            validate={"validator": "real"},
        )
        sus.pack(expand=1, padx=5, pady=5)
        Infec = Pmw.EntryField(
            cFrame,
            labelpos="w",
            value="0.0",
            label_text="Initial Infective Io",
            validate={"validator": "real"},
        )
        Infec.pack(expand=1, padx=5, pady=5)
        contact = Pmw.EntryField(
            cFrame,
            labelpos="w",
            value="0.0",
            label_text="Contact Rate -k ",
            validate={"validator": "real"},
        )
        contact.pack(expand=1, padx=5, pady=5)
        Removal = Pmw.EntryField(
            cFrame,
            labelpos="w",
            value="0.0",
            label_text=" Removal  Rate  ",
            validate={"validator": "real"},
        )
        Removal.pack(expand=1, padx=10, pady=5)
        time_interval = Pmw.EntryField(
            cFrame,
            labelpos="w",
            value="0",
            label_text=" Time    Interval ",
            validate={"validator": "numeric"},
        )
        time_interval.pack(expand=1, padx=5, pady=5)
        death_rate = Pmw.EntryField(
            cFrame,
            labelpos="w",
            value="0.0",
            label_text=" Death  Rate - d",
            validate={"validator": "real"},
        )
        death_rate.pack(expand=1, padx=10, pady=10)
        carriers = Pmw.EntryField(
            cFrame,
            labelpos="w",
            value="0.00",
            label_text="Initial Carriers C",
            validate={"validator": "real"},
        )
        carriers.pack(expand=1, padx=10, pady=10)
        # ---------------------------------------------------------------------------------------------------------------------------------------------------------

        # self.top.transient(self)
        # self.wButton.config(state='disabled')
        def retrive():
            k1 = contact.get()
            k = float(k1)
            ddelta = float(death_rate.get())
            C1 = float(carriers.get())
            ggamma = float(Removal.get())
            N = int(time_interval.get())
            s0 = float(sus.get())
            I0 = float(Infec.get())
            A[0][0] = 0
            A[0][1] = -k
            A[1][0] = k
            A[1][1] = 0
            B[0][0] = 0
            B[0][1] = 0
            B[1][0] = 0
            B[1][1] = 0
            E[0][0] = -ddelta
            E[1][0] = -(ggamma + ddelta)
            C[0][0] = ddelta
            C[1][0] = (ddelta + ggamma) * C1

            def f1(s1, I1):
                s_dot = (
                    s1 * E[0][0]
                    + A[0][0] * (s1 ** 2)
                    + A[0][1] * s1 * I1
                    + B[0][0] * s1
                    + B[0][1] * I1
                    + C[0][0]
                )
                return s_dot

            def f2(s1, I1):
                I_dot = (
                    I1 * E[1][0]
                    + A[1][1] * (I1 ** 2)
                    + A[1][0] * s1 * I1
                    + B[1][0] * s1
                    + B[1][1] * I1
                    + C[1][0]
                )
                return I_dot

            # =========RATE OF CHANGE OF THE MODEL WITH RESPECT TO TIME AND DECLARATON OF CONSTANTS=============================================================
            t = arange(0, N, 0.02455)
            s1 = zeros(len(t))
            I1 = zeros(len(t))
            s1[0] = s0
            I1[0] = I0
            h = 0.01

            # ==========ITERATION FOR A GIVEN TIME INTERVAL======================================================================================================

            for n in range(len(t) - 1):
                s1[n + 1] = s1[n] + h * (f1(s1[n], I1[n]))
                I1[n + 1] = I1[n] + h * (f2(s1[n], I1[n]))
            x_ = tuple(t)
            y_ = tuple(s1)
            z_ = tuple(I1)
            g.configure(
                title="Susceptible and Infectives  against time for SIR with Carriers"
            )
            g.line_create(
                "S(t) Vs t", xdata=x_, color="red", ydata=y_, linewidth=2, symbol=""
            )
            g.line_create(
                "I(t) Vs t", xdata=x_, color="orange", ydata=z_, linewidth=2, symbol=""
            )

        def postscript():
            g.postscript_output(fileName="HelloUser1.ps", decorations="no")

        def newFile():
            for name in g.element_names():
                g.element_delete(name)
            contact.setentry("")
            Removal.setentry("")
            time_interval.setentry("")
            Infec.setentry("")
            sus.setentry("")
            carriers.setentry("")
            death_rate.setentry("")

        # ------------------------------------------------------------------------------------------------------------------------------------------------------
        buttons = Pmw.ButtonBox(cFrame, labelpos="w", label_text="Options")
        # buttons = Pmw.ButtonBox(self, labelpos='w', label_text='Options')
        buttons.pack(fill="both", expand=1, padx=10, pady=10)
        buttons.add("GRID", bg="green", command=g.grid_toggle)
        buttons.add("PLOT", bg="green", command=retrive)
        buttons.add("CLEAR", bg="green", command=newFile)
        buttons.add("SAVE", bg="green", command=postscript)
        handler = lambda: self.onCloseOtherFrameC(cFrame)
        buttons.add("QUIT", bg="red", command=handler)

    # ------------------------------------------------------------------------------------------------------------------------------------------------------
    # SIR with Temporary_Immunity
    # ------------------------------------------------------------------------------------------------------------------------------------------------------
    def Temporary_Immunity(self):
        self.hide()
        tIFrame = Toplevel()
        tIFrame.title("SIR WITH TEMPORARY IMMUNITY")
        tIFrame.geometry("700x800")
        # self.top.transient(self)
        g = Pmw.Blt.Graph(tIFrame)
        g.pack(expand=1, fill="both")
        E = zeros([2, 1])
        A = zeros([2, 2])
        C = zeros([2, 1])
        A = zeros([2, 2])
        B = zeros([2, 2])
        E = zeros([2, 1])
        C = zeros([2, 1])
        # -------------------------------------------------------------------------------------------------------------------------------------------------------
        sus = Pmw.EntryField(
            tIFrame,
            labelpos="w",
            value="0.0",
            label_text="Initial Susceptibles :",
            validate={"validator": "real"},
        )
        sus.pack(expand=1, padx=5, pady=5)
        Infec = Pmw.EntryField(
            tIFrame,
            labelpos="w",
            value="0.0",
            label_text="Initial Infectives :",
            validate={"validator": "real"},
        )
        Infec.pack(expand=1, padx=5, pady=5)
        contact = Pmw.EntryField(
            tIFrame,
            labelpos="w",
            value="0.0",
            label_text="Contact Rate (k) :",
            validate={"validator": "real"},
        )
        contact.pack(expand=1, padx=5, pady=5)
        Removal = Pmw.EntryField(
            tIFrame,
            labelpos="w",
            value="0.0",
            label_text=" Removal  Rate  :",
            validate={"validator": "real"},
        )
        Removal.pack(expand=1, padx=10, pady=5)
        time_interval = Pmw.EntryField(
            tIFrame,
            labelpos="w",
            value="0",
            label_text=" Time    Interval :",
            validate={"validator": "numeric"},
        )
        time_interval.pack(expand=1, padx=5, pady=5)
        death_rate = Pmw.EntryField(
            tIFrame,
            labelpos="w",
            value="0.0",
            label_text=" Death  Rate (d) :",
            validate={"validator": "real"},
        )
        death_rate.pack(expand=1, padx=10, pady=5)
        immunity = Pmw.EntryField(
            tIFrame,
            labelpos="w",
            value="0.00",
            label_text="  Immunity Rate:",
            validate={"validator": "real"},
        )
        immunity.pack(expand=1, padx=10, pady=10)
        # --------------------------------------------------------------------------------------------------------------------------------------------------------

        # self.top.transient(self)
        # self.wButton.config(state='disabled')
        def retrive():
            k1 = contact.get()
            k = float(k1)
            ddelta = float(death_rate.get())
            aalpha = float(immunity.get())
            ggamma = float(Removal.get())
            N = int(time_interval.get())
            s0 = float(sus.get())
            I0 = float(Infec.get())
            A[0][0] = 0
            A[0][1] = -k
            A[1][0] = k
            A[1][1] = 0
            B[0][0] = 0
            B[0][1] = 0
            B[1][0] = 0
            B[1][1] = 0
            E[0][0] = -(ddelta + aalpha)
            E[1][0] = -(ggamma + ddelta + +aalpha)
            C[0][0] = (ddelta + aalpha) * (1 + aalpha / k)
            C[1][0] = 0

            def f1(s1, I1):
                s_dot = (
                    s1 * E[0][0]
                    + A[0][0] * (s1 ** 2)
                    + A[0][1] * s1 * I1
                    + B[0][0] * s1
                    + B[0][1] * I1
                    + C[0][0]
                )
                return s_dot

            def f2(s1, I1):
                I_dot = (
                    I1 * E[1][0]
                    + A[1][1] * (I1 ** 2)
                    + A[1][0] * s1 * I1
                    + B[1][0] * s1
                    + B[1][1] * I1
                    + C[1][0]
                )
                return I_dot

            # =========RATE OF CHANGE OF THE MODEL WITH RESPECT TO TIME AND DECLARATON OF CONSTANTS================================================================
            t = arange(0, N, 0.02455)
            s1 = zeros(len(t))
            I1 = zeros(len(t))
            s1[0] = s0
            I1[0] = I0
            h = 0.01

            # ==========ITERATION FOR A GIVEN TIME INTERVAL=======================================================================================================

            for n in range(len(t) - 1):
                s1[n + 1] = s1[n] + h * (f1(s1[n], I1[n]))
                I1[n + 1] = I1[n] + h * (f2(s1[n], I1[n]))
            x_ = tuple(t)
            y_ = tuple(s1)
            z_ = tuple(I1)
            g.configure(
                title="Susceptible and Infectives  against time for SIR with Temporary Immunity"
            )
            g.line_create(
                "S(t) Vs t", xdata=x_, color="red", ydata=y_, linewidth=2, symbol=""
            )
            g.line_create(
                "I(t) Vs t", xdata=x_, color="orange", ydata=z_, linewidth=2, symbol=""
            )

        def postscript():
            g.postscript_output(fileName="HelloUser1.ps", decorations="no")

        def newFile():
            for name in g.element_names():
                g.element_delete(name)
            contact.setentry("")
            Removal.setentry("")
            time_interval.setentry("")
            Infec.setentry("")
            sus.setentry("")
            immunity.setentry("")
            death_rate.setentry("")

        # --------------------------------------------------------------------------------------------------------------------------------------------------------
        buttons = Pmw.ButtonBox(tIFrame, labelpos="w", label_text="Options")
        # buttons = Pmw.ButtonBox(self, labelpos='w', label_text='Options')
        buttons.pack(fill="both", expand=1, padx=10, pady=10)
        buttons.add("GRID", bg="green", command=g.grid_toggle)
        buttons.add("PLOT", bg="green", command=retrive)
        buttons.add("CLEAR", bg="green", command=newFile)
        buttons.add("SAVE", bg="green", command=postscript)
        handler = lambda: self.onCloseOtherFrameT(tIFrame)
        buttons.add("QUIT", bg="red", command=handler)

    # --------------------------------------------------------------------------------------------------------------------------------------------------------
    # SIR with Host vectors
    # --------------------------------------------------------------------------------------------------------------------------------------------------------

    def Host_Vector(self):
        self.hide()
        hVFrame = Toplevel()
        hVFrame.title("SIR WITH HOST VECTORS")
        hVFrame.geometry("700x800")
        # self.top.transient(self)
        g = Pmw.Blt.Graph(hVFrame)
        g.pack(expand=1, fill="both")
        E = zeros([3, 1])
        A = zeros([3, 3])
        C = zeros([3, 1])
        A = zeros([3, 3])
        B = zeros([2, 2])
        E = zeros([3, 1])
        C = zeros([3, 1])
        # --------------------------------------------------------------------------------------------------------------------------------------------------------
        # Textfields
        # --------------------------------------------------------------------------------------------------------------------------------------------------------
        global sus1, sus2, sus3, contact1, contact2, contact3, time_interval, death_rate1, death_rate2
        global death_rate3, recovery1, recovery2, recovery3
        sus1 = Pmw.EntryField(
            hVFrame,
            labelpos="w",
            value="0.0",
            label_text="Susceptibles1-S1",
            validate={"validator": "real"},
        )
        sus1.pack(expand=1, padx=5, pady=5)
        sus2 = Pmw.EntryField(
            hVFrame,
            labelpos="w",
            value="0.0",
            label_text="Susceptibles2-S2",
            validate={"validator": "real"},
        )
        sus2.pack(expand=1, padx=1, pady=2)
        sus3 = Pmw.EntryField(
            hVFrame,
            labelpos="w",
            value="0.0",
            label_text="Susceptibles3-S3 ",
            validate={"validator": "real"},
        )
        sus3.pack(expand=1, padx=1, pady=2)
        contact1 = Pmw.EntryField(
            hVFrame,
            labelpos="w",
            value="0.0",
            label_text="Contact Rate 1-k12",
            validate={"validator": "real"},
        )
        contact1.pack(expand=1, padx=1, pady=2)
        contact2 = Pmw.EntryField(
            hVFrame,
            labelpos="w",
            value="0.0",
            label_text="Contact Rate 2-k21",
            validate={"validator": "real"},
        )
        contact2.pack(expand=1, padx=1, pady=2)
        contact3 = Pmw.EntryField(
            hVFrame,
            labelpos="w",
            value="0.0",
            label_text="Contact Rate 3-k23",
            validate={"validator": "real"},
        )
        contact3.pack(expand=1, padx=1, pady=2)
        contact4 = Pmw.EntryField(
            hVFrame,
            labelpos="w",
            value="0.0",
            label_text="Contact Rate 4-k32",
            validate={"validator": "real"},
        )
        contact4.pack(expand=1, padx=1, pady=2)
        time_interval = Pmw.EntryField(
            hVFrame,
            labelpos="w",
            value="0",
            label_text=" Time    Interval-t ",
            validate={"validator": "numeric"},
        )
        time_interval.pack(expand=1, padx=1, pady=2)
        death_rate1 = Pmw.EntryField(
            hVFrame,
            labelpos="w",
            value="0.0",
            label_text=" Death  Rate 1-d1 ",
            validate={"validator": "real"},
        )
        death_rate1.pack(expand=1, padx=1, pady=2)
        death_rate2 = Pmw.EntryField(
            hVFrame,
            labelpos="w",
            value="0.0",
            label_text=" Death  Rate 2-d2 ",
            validate={"validator": "real"},
        )
        death_rate2.pack(expand=1, padx=1, pady=2)
        death_rate3 = Pmw.EntryField(
            hVFrame,
            labelpos="w",
            value="0.0",
            label_text=" Death  Rate 3-d3 ",
            validate={"validator": "real"},
        )
        death_rate3.pack(expand=1, padx=1, pady=2)
        recovery1 = Pmw.EntryField(
            hVFrame,
            labelpos="w",
            value="0.00",
            label_text="Recovery Rate 1",
            validate={"validator": "real"},
        )
        recovery1.pack(expand=1, padx=1, pady=2)
        recovery2 = Pmw.EntryField(
            hVFrame,
            labelpos="w",
            value="0.00",
            label_text="Recovery Rate 2",
            validate={"validator": "real"},
        )
        recovery2.pack(expand=1, padx=1, pady=2)
        recovery3 = Pmw.EntryField(
            hVFrame,
            labelpos="w",
            value="0.00",
            label_text="Recovery Rate 3",
            validate={"validator": "real"},
        )
        recovery3.pack(expand=1, padx=1, pady=2)

        # ------------------------------------------------------------------------------------------------------------------------------------------------------
        # self.top.transient(self)
        # self.wButton.config(state='disabled')
        def retrive():
            k12 = float(contact1.get())
            k21 = float(contact2.get())
            k23 = float(contact3.get())
            k32 = float(contact4.get())
            ddelta1 = float(death_rate1.get())
            ddelta2 = float(death_rate2.get())
            ddelta3 = float(death_rate3.get())
            ggamma1 = float(recovery1.get())
            ggamma2 = float(recovery2.get())
            ggamma3 = float(recovery3.get())
            N = int(time_interval.get())
            s0 = float(sus1.get())
            I0 = float(sus2.get())
            R0 = float(sus3.get())
            A[0][0] = 0
            A[0][1] = k12
            A[0][2] = 0
            A[1][0] = k21
            A[1][1] = 0
            A[1][2] = k23
            A[2][0] = 0
            A[2][1] = k32
            A[2][2] = 0
            B[0][0] = 0
            B[0][1] = 0
            B[1][0] = 0
            B[1][1] = 0
            E[0][0] = -k12 - (ggamma1 + ddelta1)
            E[1][0] = -k21 - k23 - (ggamma2 + ddelta2)
            E[2][0] = -k32 - (ggamma3 + ddelta3)
            C[0][0] = ddelta1 + ggamma1
            C[1][0] = ddelta2 + ggamma2
            C[2][0] = ddelta3 + ggamma3

            def f1(s1, I1, R1):
                s_dot = s1 * E[0][0] + A[0][1] * s1 * I1 + C[0][0]
                return s_dot

            def f2(s1, I1, R1):
                I_dot = I1 * E[1][0] + A[1][0] * s1 * I1 + A[1][2] * I1 * R1 + C[1][0]
                return I_dot

            def f3(s1, I1, R1):
                R_dot = E[2][0] * R1 + A[2][1] * I1 * R1 + C[2][0]
                return R_dot

            # =========RATE OF CHANGE OF THE MODEL WITH RESPECT TO TIME AND DECLARATON OF CONSTANTS==============================================================
            t = arange(0, N, 0.02455)
            s1 = zeros(len(t))
            I1 = zeros(len(t))
            R1 = zeros(len(t))
            s1[0] = s0
            I1[0] = I0
            R1[0] = R0
            h = 0.01

            # ==========ITERATION FOR A GIVEN TIME INTERVAL======================================================================================================

            for n in range(len(t) - 1):
                s1[n + 1] = s1[n] + h * (f1(s1[n], I1[n], R1[n]))
                I1[n + 1] = I1[n] + h * (f2(s1[n], I1[n], R1[n]))
                R1[n + 1] = R1[n] + h * (f3(s1[n], I1[n], R1[n]))
            x_ = tuple(t)
            y_ = tuple(s1)
            z_ = tuple(I1)
            w_ = tuple(R1)
            g.configure(
                title="A graph of Susceptible and Infective  against time for SIR with Host-Vector"
            )
            g.line_create(
                "S1(t) Vs t", xdata=x_, color="blue", ydata=y_, linewidth=2, symbol=""
            )
            g.line_create(
                "S2(t) Vs t", xdata=x_, color="orange", ydata=z_, linewidth=2, symbol=""
            )
            g.line_create(
                "S3(t) Vs t", xdata=x_, color="red", ydata=w_, linewidth=2, symbol=""
            )

        def postscript():
            g.postscript_output(fileName="HelloUser1.ps", decorations="no")

        def newFile():
            for name in g.element_names():
                g.element_delete(name)
            contact1.setentry("")
            contact2.setentry("")
            contact3.setentry("")
            contact4.setentry("")
            recovery1.setentry("")
            recovery2.setentry("")
            recovery3.setentry("")
            time_interval.setentry("")

            sus1.setentry("")
            sus2.setentry("")
            sus3.setentry("")
            death_rate1.setentry("")
            death_rate2.setentry("")
            death_rate3.setentry("")

        # --------------------------------------------------------------------------------------------------------------------------------------------------------
        buttons = Pmw.ButtonBox(hVFrame, labelpos="w", label_text="Options")
        buttons = Pmw.ButtonBox(hVFrame, labelpos="w", label_text="Options")
        buttons.pack(fill="both", expand=1, padx=10, pady=10)
        buttons.add("GRID", bg="green", command=g.grid_toggle)
        buttons.add("PLOT", bg="green", command=retrive)
        buttons.add("CLEAR", bg="green", command=newFile)
        buttons.add("SAVE", bg="green", command=postscript)
        handler = lambda: self.onCloseOtherFrameV(hVFrame)
        buttons.add("QUIT", bg="red", command=handler)

    # --------------------------------------------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------------------------------------------------
    def onCloseOtherFrame(self, cLFrame):
        cLFrame.destroy()
        self.show()

    # --------------------------------------------------------------------------------------------------------------------------------------------------------

    def onCloseOtherFrame1(self, vDFrame):
        vDFrame.destroy()
        self.show()

    # --------------------------------------------------------------------------------------------------------------------------------------------------------
    def onCloseOtherFrame2(self, vTFrame):
        vTFrame.destroy()
        self.show()

    # --------------------------------------------------------------------------------------------------------------------------------------------------------
    def onCloseOtherFrameC(self, cFrame):
        cFrame.destroy()
        self.show()

    # --------------------------------------------------------------------------------------------------------------------------------------------------------
    def onCloseOtherFrameT(self, tIFrame):
        tIFrame.destroy()
        self.show()

    # --------------------------------------------------------------------------------------------------------------------------------------------------------
    def onCloseOtherFrameV(self, hVFrame):
        hVFrame.destroy()
        self.show()

    # --------------------------------------------------------------------------------------------------------------------------------------------------------
    def show(self):
        self.update()
        self.deiconify()
        # self.topButton = Button(self.top, text="CLOSE", command = quit)
        # self.topButton.pack()

    def quit():
        self.quit()

    # def OnChildClose(self):
    # self.wButton.config(state='normal')
    # self.top.destroy()


if __name__ == "__main__":
    window = Window(None)

    window.title("BILINEAR EPIDEMOLOGICAL MODELS")
    window.wm_resizable(width=None, height=None)
    window.mainloop()
