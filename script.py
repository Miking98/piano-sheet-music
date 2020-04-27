import sys

from SongToSheet import song_to_m21
from SynthesiaToSong import synthesia_to_notes, FileVideoStream

# Get system args
video_file = 'theoffice.mp4' if len(sys.argv) < 2 else sys.argv[1]
first_white_note = 'A' if len(sys.argv) < 3 else sys.argv[2]
first_black_note = 'A#' if len(sys.argv) < 4 else sys.argv[3]


# Get video
print("Loading video...")
fvs = FileVideoStream(video_file).start()
# Get notes played in video
print("Converting video to song...")
song = synthesia_to_notes(fvs, 'keyboard_template.png', 
			first_white_note = first_white_note, first_black_note = first_black_note,
			animate = False, logging = True)

# 	with open('song.pickle', 'wb') as fd:
# 		pickle.dump(song, fd)
# else:
# 	with open('song.pickle', 'rb') as fd:
# 		song = pickle.load(fd)

print("Converting song to MusicXML sheet music...")
# Convert song to Music21 object
score = song_to_m21(song)

# Print sheet music
print("Displaying sheet music in MuseScore...")
score.show()
