import mpyq

# Using mpyq, load the replay file.
replay_name = 'e802955d1d65d3dcd2ec7831d90e0e8351c96770cf5a26e15500ff0f11a81a62.SC2Replay'
path = '/home/xuan/StarCraftII/Replays/'+replay_name
archive = mpyq.MPQArchive(path)
contents = archive.header['user_data_header']['content']

# Now parse the header information.
from s2protocol import versions
header = versions.latest().decode_replay_header(contents)