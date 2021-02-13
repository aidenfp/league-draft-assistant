import tkinter as tk
from loldraftassist.champ_select import champSelect
from loldraftassist.soloq.data import get_champ_role
from loldraftassist.champ_data import all_roles, champ_names
from PIL import ImageTk, Image, ImageOps

all_portraits = []
role_buttons = []
blue_team = []
red_team = []
active_role_filter = None
active_search_query = ''
model_path = 'soloq/models/11_3soloq'


class champPortrait(tk.Button):
    def __init__(self, master, champ, **kw):
        self.name = champ
        img = Image.open('assets/champ_images/{}.png'.format(champ)).resize((75, 75))
        self.portrait = ImageTk.PhotoImage(img)
        self.role = get_champ_role(champ)
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
        img = Image.open('assets/champ_images/Empty.png').resize((50, 50))
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

    def set(self, champion):
        self.current = ImageTk.PhotoImage(Image.open('assets/champ_images/{}.png'.format(champion)).resize((50, 50)))
        self.champ.set(champion)
        self.configure(image=self.current)


class roleButton(tk.Button):
    def __init__(self, master, role, **kw):
        self.role = role
        img = Image.open('assets/champ_select/{}.png'.format(role))
        self.image = ImageTk.PhotoImage(img)
        tk.Button.__init__(self, master=master, relief=tk.FLAT, image=self.image, **kw)
        self.active = False

    def update_relief(self):
        if self.active:
            self.config(relief=tk.SUNKEN)
        else:
            self.config(relief=tk.FLAT)

    def on_click(self):
        self.active = not self.active
        self.update_relief()


class scrollFrame(tk.Frame):
    def __init__(self, master, **kw):
        tk.Frame.__init__(self, master=master, **kw)
        tk.Grid.columnconfigure(self, 0, weight=1)
        tk.Grid.rowconfigure(self, 0, weight=1)
        self.canvas = canvas = tk.Canvas(self)
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


def toggle_pick(player_pick):
    global blue_team, red_team
    all_picks = blue_team + red_team
    for p in all_picks:
        if p.toggled:
            p.update_toggle()
    player_pick.update_toggle()


def make_toggle(player_pick):
    return lambda: toggle_pick(player_pick)


def update_prob_bar(bar):
    pred = predict()
    blue_win = pred['B']
    red_win = pred['R']
    bar.delete('all')
    width = bar.winfo_width()
    height = bar.winfo_height()
    bar.create_rectangle(0, 0, blue_win * width, height, fill='blue')
    bar.create_rectangle(width, 0, width - red_win * width, height, fill='red')
    bar.create_text(blue_win * width / 2, height / 2, text='{}%'.format(str(blue_win * 100)[:4]), fill='white')
    bar.create_text(width - red_win * width / 2, height / 2, text='{}%'.format(str(red_win * 100)[:4]), fill='white')


def select_champ(champ_portrait):
    global blue_team, red_team
    all_picks = blue_team + red_team
    for p in all_picks:
        # if we are currently picking for this player
        if p.toggled:
            # if the champion is available
            if champ_portrait.available:
                # if we need to update the previously selected champion
                if p.champ.get() != 'SELECT':
                    ind = champ_names.tolist().index(p.champ.get())
                    all_portraits[ind].on_click()
                champ_portrait.on_click()
                p.set(champ_portrait.name)
                print('{}{} selected {}'.format(p.team, p.pick + 1, champ_portrait.name))
                update_bar_func()


def make_select_champ(champ_portrait):
    return lambda: select_champ(champ_portrait)


def jump_to_top(scroll_frame):
    scroll_frame.canvas.yview_moveto(0)


def predict():
    global blue_team, red_team
    cs = champSelect(model_path)
    cs.load(blue=[p.champ.get() for p in blue_team], red=[p.champ.get() for p in red_team])
    print(cs.predict())
    return cs.predict()


def get_size(master):
    print(master.winfo_width(), master.winfo_height())


def role_button_command(role_button):
    global active_search_query, active_role_filter
    if role_button.role == active_role_filter:
        role_button.on_click()
        active_role_filter = None
        filter_portraits(query=active_search_query, role=active_role_filter)
    else:
        active_role_filter = role_button.role
        filter_portraits(query=active_search_query, role=active_role_filter)
        for b in role_buttons:
            if b.active:
                b.on_click()
        role_button.on_click()


def make_role_button(role_button):
    return lambda: role_button_command(role_button)


def update_search_bar(search_bar, event):
    global active_search_query, active_role_filter
    search_bar.configure(state='normal')
    current_length = len(search_bar.get('1.0', '1.end'))
    if event.char != '\x08':  # if its not a backspace
        search_bar.insert('1.end', event.char)
    else:
        search_bar.delete(f'1.{current_length - 1}')
    active_search_query = search_bar.get('1.0', '1.end')
    search_bar.configure(state='disabled')
    filter_portraits(query=active_search_query, role=active_role_filter)


def filter_portraits(**kwargs):
    query = '' if 'query' not in kwargs else kwargs['query']
    role = None if 'role' not in kwargs else kwargs['role']
    filtered_portraits = [portrait for portrait in all_portraits if
                          query.lower() in portrait.name.lower() and (role == portrait.role or role is None)]
    for portrait in all_portraits:
        if portrait not in filtered_portraits:
            portrait.grid_remove()
    for i in range(len(filtered_portraits)):
        filtered_portraits[i].grid(column=i % 6, row=i // 6)
    champ_holder_reset_yview()


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
    tk.Grid.rowconfigure(center, 2, weight=1)

    right = tk.Frame(main)
    tk.Grid.columnconfigure(right, 0, weight=1)
    tk.Grid.rowconfigure(right, tk.ALL, weight=1)

    blue_label = tk.Label(left, text='Blue Team')
    red_label = tk.Label(right, text='Red Team')
    blue_team = [playerChamp(left, 'B', i) for i in range(5)]
    red_team = [playerChamp(right, 'R', i) for i in range(5)]
    for pick in blue_team + red_team:
        pick.configure(command=make_toggle(pick))

    champ_label = tk.Label(center, text='Choose Champion')
    filter_holder = tk.Frame(center, pady=5)
    tk.Grid.columnconfigure(filter_holder, 0, weight=2)
    tk.Grid.rowconfigure(filter_holder, 0, weight=1)

    role_holder = tk.Frame(filter_holder)
    tk.Grid.columnconfigure(role_holder, [i for i in range(5)], weight=1)
    role_buttons = [roleButton(role_holder, role) for role in all_roles]
    for button in role_buttons:
        button.config(command=make_role_button(button))

    search_holder = tk.Frame(filter_holder)
    tk.Grid.columnconfigure(search_holder, 0, weight=1)
    search_label = tk.Label(search_holder, text='Search:', anchor=tk.W)
    search_field = tk.Text(search_holder, height=1, width=25, background='#12191a', foreground='#ffffff',
                           state='disabled', font='Helvetica, 10')
    search_field.bind('<Key>', lambda e: update_search_bar(search_field, e))
    search_field.bind('<BackSpace>', lambda e: update_search_bar(search_field, e))

    champ_holder = scrollFrame(center)
    champ_holder_reset_yview = lambda: jump_to_top(champ_holder)

    prob_label = tk.Label(center, text='Predicted Outcome')
    prob_bar = tk.Canvas(center, bg='gray', width=400, height=26)
    update_bar_func = lambda: update_prob_bar(prob_bar)
    # size = tk.Button(center, text='size', command=lambda: get_size(center))

    for champ in champ_names:
        all_portraits.append(champPortrait(champ_holder.scrollable_frame, champ))

    main.grid(column=0, row=0, sticky=tk.NSEW)
    left.grid(column=0, row=0, sticky=tk.NSEW)
    center.grid(column=1, row=0, sticky=tk.NSEW)
    right.grid(column=2, row=0, sticky=tk.NSEW)

    blue_label.grid(column=0, row=0, sticky=tk.N)
    red_label.grid(column=0, row=0, sticky=tk.N)

    champ_label.grid(column=0, row=0, sticky=tk.N)
    filter_holder.grid(column=0, row=1, sticky=tk.NSEW)
    role_holder.grid(column=0, row=0, sticky=tk.NSEW)

    for i in range(len(role_buttons)):
        role_buttons[i].grid(column=i, row=0)

    search_holder.grid(column=1, row=0, sticky=tk.NSEW)
    search_label.grid(column=0, row=0, sticky=tk.EW)
    search_field.grid(column=0, row=1, sticky=tk.EW + tk.S)

    champ_holder.grid(column=0, row=2, sticky=tk.NSEW)
    for i in range(len(all_portraits)):
        all_portraits[i].configure(command=make_select_champ(all_portraits[i]))
        all_portraits[i].grid(column=i % 6, row=i // 6)
    for i in range(5):
        blue_team[i].grid(column=0, row=i + 1)
        red_team[i].grid(column=0, row=i + 1)
    prob_label.grid(column=0, row=3)
    prob_bar.grid(column=0, row=4, sticky=tk.NS)
    # size.grid(column=0, row=4, sticky=tk.S)

    root.mainloop()
