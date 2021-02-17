import sqlite3 as sql
import numpy as np
import csv
import torch as t
import requests as r
from loldraftassist.champ_data import num_champs, all_roles, all_champs_vec

oe_db_url = 'https://oracleselixir-downloadable-match-data.s3-us-west-2.amazonaws.com' \
            '/2021_LoL_esports_match_data_from_OraclesElixir_20210215.csv'
assets_path = '../assets/champ_images'
db_path = 'db/2021pro.db'


def format_role(role):
    if role == 'top':
        return 'TOP'
    if role == 'jng':
        return 'JUNGLE'
    if role == 'mid':
        return 'MIDDLE'
    if role == 'bot':
        return 'BOTTOM'
    if role == 'sup':
        return 'UTILITY'


def csv_to_dict(rows):
    assert len(rows) == 12

    out = {'ID': None, 'League': None, 'Patch': None,
           'B': {'TOP': None, 'JUNGLE': None, 'MIDDLE': None, 'BOTTOM': None, 'UTILITY': None},
           'R': {'TOP': None, 'JUNGLE': None, 'MIDDLE': None, 'BOTTOM': None, 'UTILITY': None},
           'W': None}

    for row in rows:
        id = row[0]
        league = row[3]
        patch = row[9]
        side = row[11][0]
        position = format_role(row[12])
        champion = row[15]
        winner = int(row[22])

        if out['ID'] is None:
            out['ID'] = id

        if out['League'] is None:
            out['League'] = league

        if out['Patch'] is None:
            out['Patch'] = patch

        if position is not None and out[side][position] is None:
            out[side][position] = champion

        if out['W'] is None:
            out['W'] = 'B' if winner == 1 and side == 'B' else 'R'

    # make sure all of our data is filled in
    for one, two in zip(out['B'].values(), out['R'].values()):
        try:
            assert one is not None and two is not None
        except AssertionError:
            return None

    return out


def get_pro_data(csv_db):
    with open(csv_db) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        all_rows = [row for row in csv_reader]
        line_count = 0
        row_buffer = []
        out = []
        for row in all_rows[1:]:
            if line_count % 12 == 0 and len(row_buffer) > 0:
                to_append = csv_to_dict(row_buffer)
                if to_append is not None:
                    out.append(to_append)
                row_buffer = []
                line_count = 0
            row_buffer.append(row)
            line_count += 1
        return out


def data_to_db(db, data):
    conn = sql.connect(db)
    c = conn.cursor()
    c.execute('''DROP TABLE IF EXISTS matches_table''')
    c.execute(
        '''CREATE TABLE IF NOT EXISTS matches_table (ID text, League text, Patch text, B_TOP text, B_JUNGLE text, B_MIDDLE text, B_BOTTOM text, B_UTILITY text, R_TOP text, R_JUNGLE text, R_MIDDLE text, R_BOTTOM text, R_UTILITY text, W text);''')
    for match in data:
        c.execute('''INSERT into matches_table VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
              (match['ID'], match['League'], match['Patch'],
               match['B']['TOP'], match['B']['JUNGLE'], match['B']['MIDDLE'], match['B']['BOTTOM'], match['B']['UTILITY'],
               match['R']['TOP'], match['R']['JUNGLE'], match['R']['MIDDLE'], match['R']['BOTTOM'], match['R']['UTILITY'],
               match['W']))
    conn.commit()
    conn.close()


def pick_to_vec(champ, tag, input_vec=None):
    team, role = tuple(tag.split('_'))
    if input_vec is None:
        out = np.zeros((1, 10 * num_champs))
    team_offset = 0 if team == 'B' else 5 * num_champs
    role_offset = all_roles.index(role)
    champ_ind = all_champs_vec.tolist().index([champ]) * 5
    total_ind = team_offset + role_offset + champ_ind
    if input_vec is None:
        out[:, total_ind] = 1
        return out
    else:
        input_vec[:, total_ind] = 1
        return input_vec


# be sure to adjust to correct patches for now
def db_to_tensor(db, patches):
    conn = sql.connect(db)
    c = conn.cursor()
    matches = c.execute(f'''SELECT * FROM matches_table WHERE Patch IN {patches};''').fetchall()
    x = np.zeros((len(matches), 10 * num_champs))
    y = np.zeros((len(matches), 1))
    i = 0
    for match in matches:
        match_dict = {'B_TOP': match[3], 'B_JUNGLE': match[4], 'B_MIDDLE': match[5], 'B_BOTTOM': match[6],
                      'B_UTILITY': match[7], 'R_TOP': match[8], 'R_JUNGLE': match[9], 'R_MIDDLE': match[10],
                      'R_BOTTOM': match[11], 'R_UTILITY': match[12]}
        win = match[13]
        match_vec = None
        for tag, champ in match_dict.items():
            match_vec = pick_to_vec(champ, tag, match_vec)
        x[i:i+1, :] = match_vec
        y[i:i + 1, :] = 1 if win == 'B' else 0
        i += 1
    print(f'x shape: {x.shape}')
    print(f'y shape: {y.shape}')
    return t.from_numpy(x), t.from_numpy(y)


def update_data():
    new_csv = r.get(oe_db_url)

    with open('db/2021pro.csv', 'wb') as new_data:
        new_data.write(new_csv.content)

    new_dict = get_pro_data('db/2021pro.csv')
    data_to_db('db/2021pro.db', new_dict)


if __name__ == '__main__':
    update_data()
