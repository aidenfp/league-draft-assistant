import torch as t
import numpy as np
from ml import champs_to_vec, num_champs
from data import champs, champ_names, num_champs
from champ_select import champSelect


current_model = t.load('../assets/pkl/select_predict_updated.pkl')
current_transfer = t.load('../assets/pkl/probability_model.pkl')


# computes the change in win percentage based on the initial state and the next state
# returns column vector of size 2 * num_champs with each entry corresponding to net change
def win_deltas(blue, red, active_team=None):
    initial = champSelect()
    initial.load(blue=blue, red=red)

    predicts = initial.predict()
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


if __name__ == '__main__':
    b = win_deltas(['Amumu'], ['Kalista', 'Alistar'], 'B')
    print(b)
