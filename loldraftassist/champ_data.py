import cassiopeia as cass
import numpy as np

# cass.set_riot_api_key("insert key here")
cass.set_default_region("NA")
cass_champs = cass.get_champions()
champ_names = np.array([champ.name for champ in cass_champs])
num_champs = champ_names.shape[0]
all_champs_vec = champ_names.reshape((num_champs, 1))
all_roles = ['TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'UTILITY']


def get_image_assets():
    for champ in cass_champs:
        champ.image.image.save(f'assets/champ_images/{champ.name}.png', 'PNG')
