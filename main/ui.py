from champ_select import champSelect
from data import cass_champs, champ_names, num_champs
import pickle
import tkinter as tk
from PIL import ImageTk, Image, ImageOps


class champPortrait(tk.Button):
    def __init__(self, master, champ, **kw):
        self.name = champ
        img = Image.open('../assets/champ_images/{}.png'.format(champ)).resize((75, 75))
        self.portrait = ImageTk.PhotoImage(img)
        self.gray_portrait = ImageTk.PhotoImage(ImageOps.grayscale(img))
        tk.Button.__init__(self, master=master, relief=tk.FLAT, image=self.portrait, **kw)
        self.available = True

    def update_portrait(self):
        if not self.available:
            self.config(image=self.gray_portrait)
        else:
            self.config(image=self.portrait)

    def on_click(self):
        self.available = not self.available
        self.update_portrait()


class playerChamp(tk.Button):
    def __init__(self, master, team, pick, **kw):
        img = Image.open('../assets/champ_images/Empty.png').resize((50, 50))
        self.current = ImageTk.PhotoImage(img)
        tk.Button.__init__(self, master=master, image=self.current, relief=tk.FLAT, **kw)
        self.team = team
        self.pick = pick
        self.champ = tk.StringVar()
        self.champ.set('SELECT')
        self.toggled = False

    def update_toggle(self):
        self.toggled = not self.toggled
        if self.toggled:
            self.configure(relief=tk.SUNKEN)
        else:
            self.configure(relief=tk.FLAT)

    def set(self, champ):
        self.current = ImageTk.PhotoImage(Image.open('../assets/champ_images/{}.png'.format(champ)).resize((50, 50)))
        self.champ.set(champ)
        self.configure(image=self.current)


class scrollFrame(tk.Frame):
    def __init__(self, master, *args, **kw):
        tk.Frame.__init__(self, master=master, **kw)
        tk.Grid.columnconfigure(self, 0, weight=1)
        tk.Grid.rowconfigure(self, 0, weight=1)
        canvas = tk.Canvas(self)
        tk.Grid.columnconfigure(canvas, 0, weight=1)
        tk.Grid.rowconfigure(canvas, 0, weight=1)
        scrollbar = tk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.scrollable_frame = tk.Frame(canvas)
        tk.Grid.columnconfigure(self.scrollable_frame, tk.ALL, weight=1)
        tk.Grid.rowconfigure(self.scrollable_frame, tk.ALL, weight=1)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.grid(column=0, row=0, sticky=tk.NSEW)
        scrollbar.grid(column=1, row=0, sticky=tk.NS)


def toggle_pick(pick, all_picks):
    for p in all_picks:
        if p.toggled:
            p.update_toggle()
    pick.update_toggle()


def make_toggle(pick, all_picks):
    return lambda: toggle_pick(pick, all_picks)


def update_prob_bar(bar, blue, red):
    pred = predict([pick.champ.get() for pick in blue], [pick.champ.get() for pick in red])
    bwin = pred['B']
    rwin = pred['R']
    bar.delete('all')
    width = bar.winfo_width()
    height = bar.winfo_height()
    blue_rect = bar.create_rectangle(0, 0, bwin*width, height, fill='blue')
    red_rect = bar.create_rectangle(width, 0, width-rwin*width, height, fill='red')
    blue_prob = bar.create_text(bwin*width/2, height/2, text='{}%'.format(str(bwin*100)[:4]), fill='white')
    red_prob = bar.create_text(width-rwin*width/2, height/2, text='{}%'.format(str(rwin*100)[:4]), fill='white')


def select_champ(champ, all_picks, all_portraits, update):
    for p in all_picks:
        if p.toggled:
            # if the champion is available
            if champ.available:
                # if we need to update the previously selected champion
                if p.champ.get() != 'SELECT':
                    ind = champ_names.tolist().index(p.champ.get())
                    all_portraits[ind].on_click()
                champ.on_click()
                p.set(champ.name)
                print('{}{} selected {}'.format(p.team, p.pick+1, champ.name))
                update()


def make_select_champ(champ, all_picks, all_portraits, update):
    return lambda: select_champ(champ, all_picks, all_portraits, update)


def predict(blue, red):
    blue_team = [pick for pick in blue]
    red_team = [pick for pick in red]
    cs = champSelect()
    cs.load(blue=blue_team, red=red_team)
    print(cs.predict())
    return cs.predict()


def get_size(master):
    print(master.winfo_width(), master.winfo_height())


# constructs the program
if __name__ == '__main__':
    root = tk.Tk()
    root.title('Champion Select')
    tk.Grid.rowconfigure(root, 0, weight=1)
    tk.Grid.columnconfigure(root, 0, weight=1)
    root.minsize(695, 398)
    root.maxsize(695, 398)

    main = tk.Frame(root)
    tk.Grid.rowconfigure(main, 0, weight=1)
    tk.Grid.columnconfigure(main, [0, 2], weight=1)
    tk.Grid.columnconfigure(main, 1, weight=3)

    left = tk.Frame(main)
    tk.Grid.columnconfigure(left, 0, weight=1)
    tk.Grid.rowconfigure(left, tk.ALL, weight=1)

    center = tk.Frame(main)
    tk.Grid.columnconfigure(center, 0, weight=1)
    tk.Grid.rowconfigure(center, 1, weight=1)

    right = tk.Frame(main)
    tk.Grid.columnconfigure(right, 0, weight=1)
    tk.Grid.rowconfigure(right, tk.ALL, weight=1)

    blue_label = tk.Label(left, text='Blue Team')
    red_label = tk.Label(right, text='Red Team')
    blue_team = [playerChamp(left, 'B', i) for i in range(5)]
    red_team = [playerChamp(right, 'R', i) for i in range(5)]
    for pick in blue_team+red_team:
        command = make_toggle(pick, blue_team+red_team)
        pick.configure(command=command)

    champ_label = tk.Label(center, text='Choose Champion')
    champ_holder = scrollFrame(center)

    select = champSelect()
    select.load(blue=[pick for pick in blue_team], red=[pick for pick in red_team])

    prob_label = tk.Label(center, text='Predicted Probability')
    prob_bar = tk.Canvas(center, bg='gray', width=400, height=26)
    update_bar = lambda: update_prob_bar(prob_bar, blue_team, red_team)
    #size = tk.Button(center, text='size', command=lambda: get_size(center))

    portraits = []
    for champ in champ_names:
        portraits.append(champPortrait(champ_holder.scrollable_frame, champ))

    main.grid(column=0, row=0, sticky=tk.NSEW)
    left.grid(column=0, row=0, sticky=tk.NSEW)
    center.grid(column=1, row=0, sticky=tk.NSEW)
    right.grid(column=2, row=0, sticky=tk.NSEW)

    blue_label.grid(column=0, row=0, sticky=tk.N)
    red_label.grid(column=0, row=0, sticky=tk.N)

    champ_label.grid(column=0, row=0, sticky=tk.N)
    champ_holder.grid(column=0, row=1, sticky=tk.NSEW)
    for i in range(num_champs):
        portraits[i].configure(command=make_select_champ(portraits[i], blue_team+red_team, portraits, update_bar))
        portraits[i].grid(column=i%6, row=i//6)
    for i in range(5):
        blue_team[i].grid(column=0, row=i+1)
        red_team[i].grid(column=0, row=i+1)
    prob_label.grid(column=0, row=2)
    prob_bar.grid(column=0, row=3, sticky=tk.NS)
    #size.grid(column=0, row=4, sticky=tk.S)

    root.mainloop()
