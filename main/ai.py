import torch as t
import numpy as np
from ml import champs_to_vec, num_champs, device
from data import champs, champ_names, num_champs
from champ_select import champSelect, InSelectError


current_model = t.load('../assets/pkl/select_predict_updated.pkl', map_location=device)
current_transfer = t.load('../assets/pkl/probability_model.pkl')


# computes the change in win percentage based on the initial state and the next state
# returns column vector of size 2 * num_champs with each entry corresponding to net change
def win_deltas(blue, red, active_team=None):
    initial = champSelect()
    initial.load(blue=blue, red=red)

    predicts = initial.predict(False)
    b_initial = predicts['B']
    r_initial = predicts['R']

    b_deltas = np.zeros((num_champs, 1))
    r_deltas = np.zeros((num_champs, 1))
    if active_team == 'B':
        for champ in champ_names:
            if champ not in blue:
                blue.append(champ)
                next = champSelect()
                next.load(blue=blue, red=red)
                new_predicts = next.predict(False)
                predict = new_predicts['B']
                delta = predict - b_initial
                b_deltas[np.where(champs == champ)] = delta
                blue.remove(champ)
        return b_deltas
    elif active_team == 'R':
        for champ in champ_names:
            if champ not in blue:
                red.append(champ)
                next = champSelect()
                next.load(blue=blue, red=red)
                new_predicts = next.predict(False)
                predict = new_predicts['R']
                delta = predict - r_initial
                r_deltas[np.where(champs == champ)] = delta
                red.remove(champ)
        return r_deltas
    else:
        for champ in champ_names:
            for team, label in zip([blue, red], ['B', 'R']):
                if champ not in team:
                    team.append(champ)
                    next = champSelect()
                    next.load(blue=blue, red=red)
                    new_predicts = next.predict(False)
                    b_predict = new_predicts['B']
                    r_predict = new_predicts['R']
                    b_delta = b_predict - b_initial
                    r_delta = r_predict - r_initial
                    if label == 'B':
                        b_deltas[np.where(champs == champ)] = b_delta
                    else:
                        r_deltas[np.where(champs == champ)] = r_delta
                    team.remove(champ)
    return b_deltas, r_deltas


class botSelect(champSelect):
    def _ai_choose(self, team, type):
        # picks the champion with highest win delta
        if type == 'pick':
            deltas = win_deltas(self.picks['B'], self.picks['R'], team).tolist()
            zipped_deltas = zip(deltas, champ_names.tolist())
            sorted_deltas = sorted(zipped_deltas, key=lambda x: x[0], reverse=True)
            num = 0
            while True:
                choice = sorted_deltas[num][1]
                num += 1
                try:
                    return self.pick(choice, team)
                except InSelectError:
                    continue
        # bans the champion with opposing team's highest win delta
        elif type == 'ban':
            deltas = win_deltas(self.picks['B'], self.picks['R'], 'B' if team == 'R' else 'R').tolist()
            zipped_deltas = zip(deltas, champ_names.tolist())
            sorted_deltas = sorted(zipped_deltas, key=lambda x: x[0], reverse=True)
            num = 0
            while True:
                choice = sorted_deltas[num][1]
                num += 1
                try:
                    return self.ban(choice, team)
                except InSelectError:
                    continue

    def vs_ai(self, ai_team='R'):
        # Ban phase 1
        for side in ['B', 'R', 'B', 'R', 'B', 'R']:
            update = 0
            while not update:
                if side != ai_team:
                    update = self.ban(input('{} Ban: '.format(side)), side)
                else:
                    update = self._ai_choose(ai_team, 'ban')
        # Pick phase 1
        for r, side in zip([0, 0, 1, 1, 2, 2], ['B', 'R', 'R', 'B', 'B', 'R']):
            update = 0
            while not update:
                if side != ai_team:
                    update = self.pick(input('{}{}: '.format(side, r + 1)), side)
                else:
                    update = self._ai_choose(ai_team, 'pick')
        # Ban phase 2
        for side in ['B', 'R', 'B', 'R']:
            update = 0
            while not update:
                if side != ai_team:
                    update = self.ban(input('{} Ban: '.format(side)), side)
                else:
                    update = self._ai_choose(ai_team, 'ban')
        # Pick phase 2
        for r, side in zip([3, 3, 4, 4], ['R', 'B', 'B', 'R']):
            update = 0
            while not update:
                if side != ai_team:
                    update = self.pick(input('{}{}: '.format(side, r + 1)), side)
                else:
                    update = self._ai_choose(ai_team, 'pick')
        print(f'Select complete!\nAll bans: {self.bans}\n'
              f'All picks: {self.picks}\nComputed probabilities: {self.predict()}')


if __name__ == '__main__':
    select = botSelect()
    select.vs_ai()
