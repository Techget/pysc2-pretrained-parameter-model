import os
from os.path import expanduser
import numpy as np
import pickle
from google.protobuf.json_format import MessageToJson
import json
import pysc2.lib.actions as pysc2_actions


class batchGenerator(object):
	def __init__(self):
		self.home_dir = expanduser("~")
		self.parsed_directory = self.home_dir+'/pysc2-replay/data_full/' 
		self.parsed_filenames = os.listdir(self.parsed_directory)
		self.next_index = 0
		
		self.next_index_within_file = 0
		self.winner_id_within_file = -1
		self.BATCH_SIZE_LIMIT = 1000

		self.dimension = 64
		self.used_map = 'Abyssal Reef LE'
		self.player1_used_race = 'Terran' # Terran vs Terran only
		self.player2_used_race = 'Terran' # Terran vs Terran only
		self.reserve_validation_file()


	# reserve one file as validation data set, also warm up the validation 
	def reserve_validation_file(self):
		FIND_FLAG = False

		# warm up the self.next_index as well
		while FIND_FLAG == False:
			full_filename = self.parsed_directory+self.parsed_filenames[self.next_index]
			if os.path.getsize(full_filename) == 0:
				del self.parsed_filenames[self.next_index]
				full_filename = self.parsed_directory+self.parsed_filenames[self.next_index]
				if self.next_index >= len(self.parsed_filenames):
					self.next_index = 0
				continue

			self.next_index += 1
			try:
				replay_data = pickle.load(open(full_filename, "rb"))
			except:
				replay_data = []
				continue

			loaded_replay_info_json = MessageToJson(replay_data['info'])
			info_dict = json.loads(loaded_replay_info_json)

			winner_id = -1
			for pi in info_dict['playerInfo']:
				if pi['playerResult']['result'] == 'Victory':
					winner_id = int(pi['playerResult']['playerId'])
					break

			CONTAIN_FLAG = False
			for state in replay_data['state']:
				if state['actions'] == []:
					continue
				# player info
				pi_temp = np.array(state['player'])
				if pi_temp[0] == winner_id:
					CONTAIN_FLAG = True
					break

			# if info_dict['mapName'] == self.used_map and \
			# 	info_dict['playerInfo'][0]['playerInfo']['raceActual'] == self.player1_used_race and \
			# 	info_dict['playerInfo'][1]['playerInfo']['raceActual'] == self.player2_used_race and \
			if winner_id != -1 and CONTAIN_FLAG == True:

				del self.parsed_filenames[self.next_index - 1] # remove from training data set
				self.validation_file_name = full_filename
				self.validation_winner_id = winner_id
				# print(full_filename, replay_data)
				FIND_FLAG = True


	# every batch corresponding to 1 replay file
	def next_batch(self, get_validation_data = False):
		replay_data = []
		winner_id = -1

		if get_validation_data != False:
			replay_data = pickle.load(open(self.validation_file_name, "rb"))
			winner_id = self.validation_winner_id

		if self.next_index_within_file != 0:
			# read in file at index of 'self.next_index - 1'
			replay_data = pickle.load(open(self.parsed_directory+self.parsed_filenames[self.next_index-1], "rb"))
			winner_id = self.winner_id_within_file

		while replay_data == []:
			full_filename = self.parsed_directory+self.parsed_filenames[self.next_index]
			if full_filename == self.validation_file_name:
				continue

			if os.path.getsize(full_filename) == 0:
				del self.parsed_filenames[self.next_index]
				full_filename = self.parsed_directory+self.parsed_filenames[self.next_index]
				if self.next_index >= len(self.parsed_filenames):
					self.next_index = 0
				continue

			self.next_index += 1
			if self.next_index == len(self.parsed_filenames):
				self.next_index = 0	

			try:
				replay_data = pickle.load(open(full_filename, "rb"))
			except:
				replay_data = []
				continue

			loaded_replay_info_json = MessageToJson(replay_data['info'])
			info_dict = json.loads(loaded_replay_info_json)

			# if info_dict['mapName'] != self.used_map or \
			# 	info_dict['playerInfo'][0]['playerInfo']['raceActual'] == self.player1_used_race or \
			# 	info_dict['playerInfo'][1]['playerInfo']['raceActual'] == self.player2_used_race:
			# 	continue

			winner_id = -1
			for pi in info_dict['playerInfo']:
				if pi['playerResult']['result'] == 'Victory':
					winner_id = int(pi['playerResult']['playerId'])
					break

			if winner_id == -1:
				# 'Tie'
				replay_data = [] # release memory
				continue

		minimap_output = []
		screen_output = []
		action_output = []
		player_info_output = []
		available_actions_output = []


		output_counter = 0
		self.winner_id_within_file = winner_id
		for state in replay_data['state'][self.next_index_within_file:]:
			self.next_index_within_file += 1 

			if state['actions'] == []:
				continue

			# player info
			pi_temp = np.array(state['player'])
			if pi_temp[0] != winner_id:
				continue

			# minimap
			m_temp = np.array(state['minimap'])
			m_temp = np.reshape(m_temp, [self.dimension,self.dimension,5])
			# screen
			s_temp = np.array(state['screen'])
			s_temp = np.reshape(s_temp, [self.dimension,self.dimension,10])
			
			# one-hot action_id
			last_action = None
			for action in state['actions']:
				if last_action == action:
					# filter repeated action
					continue

				output_counter += 1
				one_hot = np.zeros((1, 524)) # shape will be 1*254
				one_hot[np.arange(1), [action[0]]] = 1

				# for param in action[2]:
				minimap_output.append(m_temp)
				screen_output.append(s_temp)
				action_output.append(one_hot[0])
				player_info_output.append(pi_temp)
				available_actions_output.append(np.array(state['available_actions']))

			if output_counter >= self.BATCH_SIZE_LIMIT:
				break

		if output_counter < self.BATCH_SIZE_LIMIT:
			# means finishing reading the current file
			self.next_index_within_file = 0
			self.winner_id_within_file = -1

		assert(len(minimap_output) == len(action_output))
		if len(minimap_output) == 0:
			# The replay file only record one person's operation, so if it is 
			# the defeated person, we need to skip the replay file
			return self.next_batch()

		return minimap_output, screen_output, player_info_output, available_actions_output, action_output


	# every batch corresponding to 1 replay file, action params
	def next_batch_params(self, get_validation_data = False):
		replay_data = []
		winner_id = -1

		if get_validation_data != False:
			replay_data = pickle.load(open(self.validation_file_name, "rb"))
			winner_id = self.validation_winner_id

		while replay_data == []:
			full_filename = self.parsed_directory+self.parsed_filenames[self.next_index]
			if os.path.getsize(full_filename) == 0:
				del self.parsed_filenames[self.next_index]
				full_filename = self.parsed_directory+self.parsed_filenames[self.next_index]
				if self.next_index >= len(self.parsed_filenames):
					self.next_index = 0
				continue

			self.next_index += 1
			if self.next_index == len(self.parsed_filenames):
				self.next_index = 0	

			try:
				replay_data = pickle.load(open(full_filename, "rb"))
			except:
				replay_data = []
				continue

			loaded_replay_info_json = MessageToJson(replay_data['info'])
			info_dict = json.loads(loaded_replay_info_json)

			winner_id = -1
			for pi in info_dict['playerInfo']:
				if pi['playerResult']['result'] == 'Victory':
					winner_id = int(pi['playerResult']['playerId'])
					break

			if winner_id == -1:
				# 'Tie'
				replay_data = [] # release memory
				continue

		minimap_output = []
		screen_output = []
		action_output = []
		player_info_output = []

		ground_truth_parameters = []

		for state in replay_data['state']:
			if state['actions'] == []:
				continue

			# player info
			pi_temp = np.array(state['player'])
			if pi_temp[0] != winner_id:
				continue

			# minimap
			m_temp = np.array(state['minimap'])
			m_temp = np.reshape(m_temp, [self.dimension,self.dimension,5])
			# screen
			s_temp = np.array(state['screen'])
			s_temp = np.reshape(s_temp, [self.dimension,self.dimension,10])
			
			# one-hot action_id
			last_action = None
			for action in state['actions']:
				if last_action == action:
					# filter repeated action
					continue

				one_hot = np.zeros((1, 524)) # shape will be 1*254
				one_hot[np.arange(1), [action[0]]] = 1

				action_param_type = pysc2.actions.FUNCTION_TYPES[pysc2_actions.FUNCTIONS[action[0]].function_type]				

				if action_param_type == 'no_op' or action_param_type == 'autocast':
					continue

				# for param in action[2]:
				minimap_output.append(m_temp)
				screen_output.append(s_temp)
				action_output.append(one_hot[0])
				player_info_output.append(pi_temp)
				ground_truth_parameters.append(param)
				function_types.append(action_param_type)

		assert(len(minimap_output) == len(ground_truth_parameters))

		if len(minimap_output) == 0:
			# The replay file only record one person's operation, so if it is 
			# the defeated person, we need to skip the replay file
			return self.next_batch()

		return minimap_output, screen_output, action_output, player_info_output, ground_truth_parameters, function_types












