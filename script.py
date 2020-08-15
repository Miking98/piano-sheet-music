import sys, os, argparse, pickle

from SongToSheet import song_to_m21
from SynthesiaToSong import synthesia_to_notes, FileVideoStream

# Get system args
parser = argparse.ArgumentParser(description='Convert HD .mp4 of Synthesia -> Sheet Music')
parser.add_argument('file', type = str, help = 'Name of .mp4 file (minus the .mp4). Must be trimmed so that keyboard is clearly visible in first frame. Right hand must be blue, left hand green')
parser.add_argument('-w', dest='first_white_note', type = str, default = "A", help = 'First white note in keyboard, (e.g. "A", "B", "C", "D", "E", "F", "G"). Default to "A"')
parser.add_argument('-b', dest='first_black_note', type = str, default = "A#", help = 'First black note in keyboard, must be sharp (e.g. "A#", "C#", "D#", "F#", "G#"). Default to "A#')
parser.add_argument('--a', dest='animate', action='store_true', help = 'If true, set animate flags to True')

args = parser.parse_args()

# Get video
print("Loading video...")
if not os.path.exists('song_pickles/' + args.file + ".pickle"):
	# If .pickle file hasn't been created, then arguments are necessary
	if None in [args.first_white_note, args.first_black_note, ]:
		print("ERROR. Without a pre-made .pickle file, you must provide all arguments to script.py")
		exit(1)
	fvs = FileVideoStream(args.file + '.mp4').start()
	# Get notes played in video
	print("Converting video to song...")
	lh_song, rh_song = synthesia_to_notes(fvs,
											first_white_note = args.first_white_note, first_black_note = args.first_black_note,
											animate = args.animate, logging = True)
	# Save song to .pickle file
	with open('song_pickles/' + args.file + '.pickle', 'wb') as fd:
		pickle.dump((lh_song, rh_song), fd)
else:
	with open('song_pickles/' + args.file + '.pickle', 'rb') as fd:
		(lh_song, rh_song) = pickle.load(fd)

print("Converting song to MusicXML sheet music...")
# Convert song to Music21 object
score = song_to_m21(lh_song, rh_song)

# Print sheet music
print("Saving sheet music to MusicXML...")
score.write(fmt = 'musicxml', fp = 'music_xmls/' + args.file + '.musicxml') # Options: musicxml text midi lily (or lilypond) lily.png lily.pdf lily.svg braille vexflow musicxml.png


