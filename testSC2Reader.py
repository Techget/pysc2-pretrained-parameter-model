# import os
# os.environ['SC2READER_CACHE_DIR'] = "path/to/local/cache"
# os.environ['SC2READER_CACHE_MAX_SIZE'] = "100"

# if you have imported sc2reader anywhere already this won't work
import sc2reader

# replay_name = 'e802955d1d65d3dcd2ec7831d90e0e8351c96770cf5a26e15500ff0f11a81a62.SC2Replay'
replay_name = '00149c4c0a6451ab831e0c570e6318ca21ba1fac26c2f1f45d6794dbe001ea2e.SC2Replay'
path = '/home/xuan/StarCraftII/Replays/'+replay_name
replay = sc2reader.load_replay(path, load_map=True)
replay.load_map()
