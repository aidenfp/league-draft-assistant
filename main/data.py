import cassiopeia as cass
import sqlite3 as sql
import numpy as np
import arrow
import random
import torch as t
from roleidentification import pull_data
from sortedcontainers import SortedList

assets_path = '../assets/champ_images'
cass.set_riot_api_key("RGAPI-bb8e0309-666c-4862-8932-0f171119a130")
cass.set_default_region("NA")
cass_champs = cass.get_champions()
champ_names = np.array([champ.name for champ in cass_champs])
num_champs = champ_names.shape[0]
champs = champ_names.reshape((num_champs, 1))


def id_to_name(data):
    new_data = {}
    for champ_id in data.keys():
        roles = data[champ_id]
        champ = cass.Champion(id=champ_id)
        new_data.update({champ.name: roles})
    return new_data


role_data = id_to_name(pull_data())


def get_image_assets():
    for champ in cass_champs:
        champ.image.image.save(f'../assets/champ_images/{champ.name}.png', 'PNG')


# from Cassiopeia examples
# filters a summoner's match history based on patch and queue type
def filter_match_history(summoner, patch):
    end_time = patch.end
    if end_time is None:
        end_time = arrow.now()
    match_history = cass.MatchHistory(summoner=summoner, queues={cass.Queue.ranked_solo_fives, cass.Queue.clash},
                                      begin_time=patch.start, end_time=end_time)
    return match_history


# extracts winner, champion from match data
def get_match_info(win, bt, rt, patch):
    winner = 'B' if win else 'R'
    blue_side = [champ_names[cass_champs.index(c.champion)] for c in bt.participants]
    red_side = [champ_names[cass_champs.index(c.champion)] for c in rt.participants]
    return blue_side, red_side, winner, patch.name


# takes in list of Cassiopeia match class as input and appends relevant info to database
def add_to_db(c, match_info):
    c.execute(
        '''CREATE TABLE IF NOT EXISTS matches_table (B_1 text, B_2 text, B_3 text, B_4 text, B_5 text, R_1 text, R_2 text, R_3 text, R_4 text, R_5 text, W text, PATCH text);''')
    blue, red, winner, patch = match_info
    c.execute('''INSERT into matches_table VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
              (blue[0], blue[1], blue[2], blue[3], blue[4], red[0], red[1], red[2], red[3], red[4], winner, patch))


# Converts list of champion names into one-hot encoded vector
def champs_to_vec(champ_list):
    return np.isin(champs, champ_list).astype(dtype=np.uint8)


# pull match data from database and converts to torch tensors
def db_to_tensor(db):
    conn = sql.connect(db)
    c = conn.cursor()
    matches = c.execute('''SELECT * FROM matches_table;''').fetchall()
    x = np.zeros((2 * num_champs, len(matches)))
    y = np.zeros((1, len(matches)))
    i = 0
    for match in matches:
        blue = [match[i] for i in range(5)]
        red = [match[i + 5] for i in range(5)]
        win = match[10]
        blue_vec = champs_to_vec(blue)
        red_vec = champs_to_vec(red)
        x_vec = np.vstack((blue_vec, red_vec))
        x[:, i:i + 1] = x_vec
        y[:, i:i + 1] = 1 if win == 'B' else 0
        i += 1
    return t.from_numpy(x).T, t.from_numpy(y).T


# From Cassiopeia examples
# modified
def collect_matches(initial_summoner_name, match_db, patch, db_size=50):
    region = "NA"
    summoner = cass.Summoner(name=initial_summoner_name, region=region)
    patch = cass.Patch.from_str(patch, region=region)

    unpulled_summoner_ids = SortedList([summoner.id])
    pulled_summoner_ids = SortedList()

    unpulled_match_ids = SortedList()
    pulled_match_ids = SortedList()

    count = 1

    conn = sql.connect(match_db)
    c = conn.cursor()
    while unpulled_summoner_ids:
        if len(pulled_match_ids) >= db_size:
            break
        # Get a random summoner from our list of unpulled summoners and pull their match history
        new_summoner_id = random.choice(unpulled_summoner_ids)
        try:
            new_summoner = cass.Summoner(id=new_summoner_id, region=region)
            matches = filter_match_history(new_summoner, patch)
            unpulled_match_ids.update([match.id for match in matches])
        except Exception as e:
            print("Riot API broke when pulling new summoner, going to next iteration")
            print(e)
        unpulled_summoner_ids.remove(new_summoner_id)
        pulled_summoner_ids.add(new_summoner_id)

        while unpulled_match_ids:
            if len(pulled_match_ids) >= db_size:
                break
            # Get a random match from our list of matches
            new_match_id = random.choice(unpulled_match_ids)

            # Make sure we haven't already inserted this match into our database
            while new_match_id in pulled_match_ids:
                print("Match is duplicate, pulling new match")
                if unpulled_match_ids:
                    unpulled_match_ids.remove(new_match_id)
                    new_match_id = random.choice(unpulled_match_ids)
                else:
                    break
            try:
                new_match = cass.Match(id=new_match_id, region=region)
                print('Progress: {}%'.format(count / db_size * 100))
                add_to_db(c,
                          get_match_info(new_match.blue_team.win, new_match.blue_team, new_match.red_team, patch))
                count += 1
                for participant in new_match.participants:
                    if participant.summoner.id not in pulled_summoner_ids and participant.summoner.id not in unpulled_summoner_ids:
                        unpulled_summoner_ids.add(participant.summoner.id)
            except Exception as e:
                print("Riot API broke when pulling new match, going to next iteration")
                print(e)
            unpulled_match_ids.remove(new_match_id)
            pulled_match_ids.add(new_match_id)
    print("Finishing up")
    conn.commit()
    conn.close()


def get_champ_role(champ_name):
    max_rate = 0
    max_role = 'TOP'
    for role, rate in role_data[champ_name].items():
        if rate > max_rate:
            max_role = role
            max_rate = rate
    return max_role


if __name__ == '__main__':
    collect_matches("Doublelift", '../db/11_3soloq.db', '11.3', 10000)
