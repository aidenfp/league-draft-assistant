import difflib as diff
import numpy as np
from ml import t, champs_to_vec
from data import champ_names
import torch.nn.functional as F


class champSelect:
    def __init__(self, load_model=True):
        self.picks = {'B': [0 for i in range(5)], 'R': [0 for i in range(5)]}
        self.bans = {'B': [0 for i in range(5)], 'R': [0 for i in range(5)]}
        self.model = False
        if load_model:
            self.model = t.load('../assets/pkl/select_predict_updated.pkl')
            self.model.eval()
            self.transfer = t.load('../assets/pkl/probability_model.pkl')
            self.transfer.eval()

    def in_select(self, champ):
        print(champ)
        select = self.picks['B']+self.picks['R']+self.bans['B']+self.bans['R']
        return True if champ in select else False

    def ban(self, champ, side):
        champ = diff.get_close_matches(champ, champ_names)[0]
        if not self.in_select(champ):
            try:
                ind = self.bans[side].index('SELECT')
            except ValueError:
                print('No more bans for %s' % ('Blue' if side == 'B' else 'Red'))
                return 0
            # get pick with error correction
            try:
                self.bans[side][ind] = champ
                return 1
            # if error correction fails, print error
            except IndexError:
                print('Error: Unable to find champion')
                return 0
        else:
            print("Champion unavailable")
            return 0

    def pick(self, champ, side):
        champ = diff.get_close_matches(champ, champ_names)[0]
        if not self.in_select(champ):
            try:
                ind = self.picks[side].index('SELECT')
            except ValueError:
                print('No more picks for {}'.format('Blue' if side == 'B' else 'Red'))
                return 0
            try:
                self.picks[side][ind] = champ
                return 1
            except IndexError:
                print('Error: Unable to find champion')
                return 0
        else:
            print('Champion unavailable')
            return 0

    def load(self, **kwargs):
        if 'picks' in kwargs:
            self.picks = kwargs['picks']
        if 'bans' in kwargs:
            self.bans = kwargs['bans']
        if 'blue' in kwargs:
            self.picks['B'] = kwargs['blue']
        if 'red' in kwargs:
            self.picks['R'] = kwargs['red']

    def manual(self):
        # Ban phase 1
        for side in ['B', 'R', 'B', 'R', 'B', 'R']:
            update = 0
            while not update:
                update = self.ban(input('{} Ban: '.format(side)), side)
        # Pick phase 1
        for r, side in zip([0, 0, 1, 1, 2, 2], ['B', 'R', 'R', 'B', 'B', 'R']):
            update = 0
            while not update:
                update = self.pick(input('{}{}: '.format(side, r + 1)), side)
        # Ban phase 2
        for side in ['B', 'R', 'B', 'R']:
            update = 0
            while not update:
                update = self.ban(input('{} Ban: '.format(side)), side)
        # Pick phase 2
        for r, side in zip([3, 3, 4, 4], ['R', 'B', 'B', 'R']):
            update = 0
            while not update:
                update = self.pick(input('{}{}: '.format(side, r + 1)), side)

    def predict(self):
        if self.model:
            print('Predicting: ', self.picks)
            blue_vec = champs_to_vec(self.picks['B'])
            red_vec = champs_to_vec(self.picks['R'])
            x_vec = np.vstack((blue_vec, red_vec))
            x = t.from_numpy(x_vec).float()
            vals = F.softmax(self.model(x.cuda().T).cpu(), dim=1)[0].detach().numpy()
            win = np.argmax(vals)
            win_prob = round(self.transfer(t.tensor([[np.max(vals)]])).data.numpy()[0][0], 3)
            lose_prob = round(1-win_prob, 3)
            if win == 0:
                return {'B': win_prob, 'R': lose_prob}
            else:
                return {'B': lose_prob, 'R': win_prob}


if __name__ == '__main__':
    test = champSelect()
    test.load(picks={'B': ["Nasus", 'Master Yi', 'Anivia', 'Draven', 'Twitch'], 'R': ['Camille', 'Udyr', 'Vladimir', 'Xayah', 'Blitzcrank']})

    print(test.predict())
