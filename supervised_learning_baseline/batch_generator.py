import os
from os.path import expanduser

class batchGenerator(object):
	def __init__(self):
		self.home_dir = expanduser("~")
		self.parsed_directory = self.home_dir+'pysc2-replay/data_64/'
		self.parsed_filenames = os.listdir(parsed_directory)
		self.next_index = 0

	# every batch corresponding to 1 replay file
	def next_batch():
		full_filename = self.parsed_directory+self.parsed_filenames[self.next_index]
		replay_data = pickle.load(open(full_filename, "rb"))

		winner_id = -1
		for pi in replay_data['info'].player_info:
			pi.





