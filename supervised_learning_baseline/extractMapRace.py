import os
from os.path import expanduser
import shutil
import pickle
from tqdm import tqdm
from google.protobuf.json_format import MessageToJson
import json

# map_used = "Odyssey LE"
# race_used = "Terran"
home_dir = expanduser("~")
home_dir += '/'
parsed_directory = home_dir+'pysc2-replay/data_full/'
extracted_directory_base = home_dir+'pysc2-replay/map_race_data/'
# extracted_directory = home_dir+'pysc2-replay/data_'+map_used+'_'+race_used+'/'


if not os.path.exists(extracted_directory_base):
	os.mkdir(extracted_directory_base)

for fn in tqdm(os.listdir(parsed_directory)):
	full_filename = parsed_directory+fn
	temp_dir = extracted_directory_base

	try:
		replay_data = pickle.load(open(full_filename, "rb"))
	except:
		continue
	
	loaded_replay_info_json = MessageToJson(replay_data['info'])
	info_dict = json.loads(loaded_replay_info_json)

	temp_dir += info_dict['mapName']+'_'+info_dict['playerInfo'][0]['playerInfo']['raceActual']+'_'+info_dict['playerInfo'][1]['playerInfo']['raceActual']
	temp_dir += '/'

	if not os.path.exists(temp_dir):
		os.mkdir(temp_dir)

	shutil.move(parsed_directory+fn, temp_dir+fn)



# shutil.move(src, dst, copy_function=copy2)
# os.mkdir()
# os.path.exists(path)
