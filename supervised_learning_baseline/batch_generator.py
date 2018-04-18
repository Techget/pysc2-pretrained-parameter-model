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
		self.parsed_directory = self.home_dir+'/pysc2-replay/data_64/'
		self.parsed_filenames = os.listdir(self.parsed_directory)
		self.next_index = 0
		self.dimension = 64
		self.used_map = 'Abyssal Reef LE'
		self.used_race = 'Terran' # Terran vs Terran only

	# every batch corresponding to 1 replay file
	def next_batch(self, get_action_id_only=False, param_type = -1):
		if get_action_id_only == True:
			assert(param_type == -1)

		replay_data = []
		winner_id = -1
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

				action_param_types = pysc2.actions.FUNCTION_TYPES[pysc2_actions.FUNCTIONS[action[0]].function_type]				



				for param in action[2]:
					minimap_output.append(m_temp)
					screen_output.append(s_temp)
					action_output.append(one_hot[0])
					player_info_output.append(pi_temp)
					ground_truth_parameters.append(np.array(param))

		assert(len(minimap_output) == len(ground_truth_parameters))

		if len(minimap_output) == 0:
			# The replay file only record one person's operation, so if it is 
			# the defeated person, we need to skip the replay file
			return self.next_batch(get_action_id_only)

		if get_action_id_only:
			return minimap_output, screen_output, player_info_output, action_output
		else:
			return minimap_output, screen_output, action_output, player_info_output, ground_truth_parameters

