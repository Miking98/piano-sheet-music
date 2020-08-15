import sys, os, argparse, pickle

from SongToSheet import song_to_m21
from SynthesiaToSong import synthesia_to_notes, FileVideoStream

# Get system args
parser = argparse.ArgumentParser(description='Convert HD .mp4 of Synthesia -> Sheet Music')
parser.add_argument('file', type = str, help = 'Path to .mp4 video file')
parser.add_argument('first_white_note', type = str, help = 'First white note in keyboard, (e.g. "A", "B", "C", "D", "E", "F", "G")')
parser.add_argument('first_black_note', type = str, help = 'First black note in keyboard, must be sharp (e.g. "A#", "C#", "D#", "F#", "G#")')

args = parser.parse_args()

# Get video
print("Loading video...")
if not os.path.exists(args.file + ".pickle"):
	fvs = FileVideoStream(args.file).start()
	# Get notes played in video
	print("Converting video to song...")
	song = synthesia_to_notes(fvs, 'keyboard_template.png', 
				first_white_note = args.first_white_note, first_black_note = args.first_black_note,
				animate = False, logging = True)
	# Save song to .pickle file
	with open(args.file + '.pickle', 'wb') as fd:
		pickle.dump(song, fd)
else:
	with open(args.file + '.pickle', 'rb') as fd:
		song = pickle.load(fd)

print("Converting song to MusicXML sheet music...")
# Convert song to Music21 object
score = song_to_m21(song)

# Print sheet music
print("Displaying sheet music in MuseScore...")
score.show()
