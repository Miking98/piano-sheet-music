from .SongToSheet import song_to_m21
from .SynthesiaToSong import synthesia_to_notes, FileVideoStream

log = logging.getLogger(__name__)

# Get video
fvs = FileVideoStream('billie.mp4').start()
# Get notes played in video
song = synethesia_to_notes(fvs, first_white_note = 'B', first_black_note = 'A#', animate = False)

# 	with open('song.pickle', 'wb') as fd:
# 		pickle.dump(song, fd)
# else:
# 	with open('song.pickle', 'rb') as fd:
# 		song = pickle.load(fd)

# Convert song to Music21 object
score = song_to_m21(song)

# Print sheet music
score.show()
