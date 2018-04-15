import os
from os.path import expanduser
import numpy as np

class batchGenerator(object):
	def __init__(self):
		self.home_dir = expanduser("~")
		self.parsed_directory = self.home_dir+'pysc2-replay/data_64/'
		self.parsed_filenames = os.listdir(parsed_directory)
		self.next_index = 0
		self.dimension = 64

	# every batch corresponding to 1 replay file
	def next_batch():
		full_filename = self.parsed_directory+self.parsed_filenames[self.next_index]
		replay_data = pickle.load(open(full_filename, "rb"))

		winner_id = -1
		for pi in replay_data['info'].player_info:
			if pi.player_result.result == 'Victory':
				winner_id = pi.player_result.player_id
				break

		minimap_output = []
		screen_output = []
		action_output = []
		player_info_output = []
		ground_truth_coordinates = []

		for state in replay_data['state']:
			if state['actions'] == []:
				continue

			# minimap
			m_temp = np.array(state['minimap'])
			m_temp = np.reshape(m_temp, [self.dimension,self.dimension,5])
			# minimap_output.append(m_temp)
			# screen
			s_temp = np.array(state['screen'])
			s_temp = np.reshape(s_temp, [self.dimension,self.dimension,10])
			# screen_output.append(s_temp)
			# player info
			pi_temp = np.array(state['player'])
			# player_info_output.append(pi_temp)

			# one-hot action_id
			last_action = None
			for action in state['actions']:
				if last_action == action:
					continue

				one_hot = np.zeros((1, 524))
				one_hot[np.arange(1), [action[0]]] = 1

				for param in action[2]:
					if param == [0]:
						continue
					minimap_output.append(m_temp)
					screen_output.append(s_temp)
					action_output.append(one_hot)
					player_info_output.append(pi_temp)
					ground_truth_coordinates.append(np.array(param))

		assert(len(minimap_output) == len(ground_truth_coordinates))

		return minimap_output, screen_output, action_output. player_info_output. ground_truth_coordinates





