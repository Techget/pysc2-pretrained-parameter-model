import os
from os.path import expanduser
import shutil
import pickle
from google.protobuf.json_format import MessageToJson
import json

map_used = "Abyssal Reef LE" #"Odyssey LE"
race_used = "Terran"
home_dir = expanduser("~")
home_dir += '/'
parsed_directory = home_dir+'pysc2-replay/data/'
extracted_directory = home_dir+'pysc2-replay/data_'+map_used+'_'+race_used+'/'


counter = {}

for fn in os.listdir(parsed_directory):
	full_filename = parsed_directory+fn

	try:
		replay_data = pickle.load(open(full_filename, "rb"))
	except:
		continue
	

	loaded_replay_info_json = MessageToJson(replay_data['info'])
	info_dict = json.loads(loaded_replay_info_json)

	# if info_dict['mapName'] == map_used and info_dict['playerInfo'][0]['playerInfo']['raceActual'] == race_used and info_dict['playerInfo'][1]['playerInfo']['raceActual'] == race_used:
	# 	print("..")
	# 	counter+=1

	if info_dict['mapName'] in counter and race_used in counter[info_dict['mapName']]:
		counter[info_dict['mapName']][race_used] += 1
	else:
		counter[info_dict['mapName']][race_used] = 1


	# if replay_data['info'].map_name != map_used:
	# 	continue

	# outer_loop_coutinue = False
	# for pi in replay_data['info'].player_info:
	# 	if pi.player_info.race_actaul != race_used:
	# 		outer_loop_coutinue = True
	# 		break
	# if outer_loop_coutinue == True:
	# 	continue

	# shutil.copyfile(parsed_directory+fn, extracted_directory+fn)

print(counter)




