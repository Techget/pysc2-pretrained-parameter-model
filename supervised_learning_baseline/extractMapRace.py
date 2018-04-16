import os
from os.path import expanduser
import shutil
import pickle

map_used = "Odyssey LE"
race_used = "Terran"
home_dir = expanduser("~")
home_dir += '/'
parsed_directory = home_dir+'pysc2-replay/data_64/'
extracted_directory = home_dir+'pysc2-replay/data_'+map_used+'_'+race_used+'/'


for fn in os.listdir(parsed_directory):
	full_filename = parsed_directory+fn
	replay_data = pickle.load(open(full_filename, "rb"))

	if replay_data['info'].map_name != map_used:
		continue

	outer_loop_coutinue = False
	for pi in replay_data['info'].player_info:
		if pi.player_info.race_actaul != race_used:
			outer_loop_coutinue = True
			break
	if outer_loop_coutinue == True:
		continue

	shutil.copyfile(parsed_directory+fn, extracted_directory+fn)





