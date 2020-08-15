import numpy as np
import music21 as m21
import copy

#
# Convert Song -> Music21XML to display in MuseScore
#

def frames_to_16ths(frames, frames_per_16th):
	return int(np.round(frames / frames_per_16th)) # 1 if 16th note

def get_semitone_dist(note1, note2, return_low_high = False):
	interval = m21.interval.Interval(noteStart = m21.note.Note(note1), noteEnd = m21.note.Note(note2)).cents // 100
	if interval > 0:
		low, high = note1, note2
	else:
		low, high = note2, note1
	if return_low_high:
		return abs(interval), low, high
	return abs(interval)

def get_unique_notes_in_song(song):
	return set([ y for x in song for y in x ])

def song_to_timesteps(song):
	# Get all unique notes in song
	unique_notes = get_unique_notes_in_song(song)
	# Set up time_steps array
	held_notes = { n:0 for n in unique_notes } # key = note held, val = number of frames held
	time_steps = [ [] for s in song ] # time_steps[i] = length of notes that began to be pressed at frame i
	for i, s in enumerate(song):
		# Record which notes were held down during this timestep
		for note in s:
			held_notes[note] += 1
		# Record and remove unpressed notes (or if end of song, currently held notes)
		end_of_song = i == len(song) - 1
		for note in unique_notes:
			if (note not in s or end_of_song) and held_notes[note] > 0:
				# Record length of this notepress
				length_of_press = held_notes[note]
				start_of_press = i - length_of_press + (1 if end_of_song else 0) # Adjust up by 1 if end of song
				time_steps[start_of_press].append({ note: length_of_press })
				# Remove it
				held_notes[note] = 0
	return time_steps

def remove_time_step_noise(time_steps):
	time_steps = copy.deepcopy(time_steps)
	for i, t in enumerate(time_steps):
		## If a note was pressed down 1 timestep after another note one semi-tone away, then its likely an artifact of the
		## ...glowing animation at the top of the keyboard when a note is pressed, so ignore
		if i == 0: continue
		for n in t:
			if len(time_steps[i-1]) > 0:
				filtered_time_step = []
				note = list(n.keys())[0]
				min_dist = min([ get_semitone_dist(note, list(n.keys())[0]) for n in time_steps[i-1] ])
				if min_dist >= 2:
					## Keep this note b/c not an artifact
					filtered_time_step.append(n)
				time_steps[i] = filtered_time_step
	return time_steps

def get_frames_per_16th(time_steps):
	# Get # of frames each note is pressed
	frames_pressed = []
	for t in time_steps:
		for n in t:
			frames_pressed.append(list(n.values())[0])
	
	# Call most often-repeated frame_length an 8th note
	frames_per_16th = np.bincount(frames_pressed).argmax()/2
	return frames_per_16th

def frames_to_durations(time_steps, frames_per_16th):
	time_steps = copy.deepcopy(time_steps)
	# Quantize each note to closest 16th
	for i, t in enumerate(time_steps):
		notes = []
		for n in t:
			key = list(n.keys())[0]
			_16ths = list(n.values())[0]//frames_per_16th
			quarters = _16ths / 4
			if quarters == 0: quarters = 0.25
			notes.append({ key : m21.duration.Duration(quarters) })
		time_steps[i] = notes
	return time_steps, frames_per_16th

def convert_note_to_key(note, key):
	#
	# Convert note (e.g. A#) to proper note for this key (e.g. Bb for Eb Major)
	# NOTE: Does in-place edits
	#
	flat_order = [ m21.pitch.Pitch(x) for x in ['B-', 'E-', 'A-', 'D-', 'G-', 'C-', 'F-' ]]
	sharp_order = [ m21.pitch.Pitch(x) for x in ['F#', 'C#', 'G#', 'D#', 'A#', 'E#', 'B#' ] ]
	conversion_table = {
		'major' : {
			'flats' : [ 'C', 'F', 'B-', 'E-', 'A-', 'D-', 'G-', 'C-' ],
			'sharps' : [ 'C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#' ],
		},
		'minor' : {
			'flats' : [ 'A', 'D', 'G', 'C', 'F', 'B-', 'E-', 'A-' ],
			'sharps' : [ 'A', 'E', 'B', 'F#', 'C#', 'G#', 'D#', 'A#' ],
		}
	}
	rel_major = conversion_table[key.mode]
	is_flat = str(key.tonic) in rel_major['flats']
	circle_of_fifths = rel_major['flats' if is_flat else 'sharps']
	accidental_order = flat_order if is_flat else sharp_order
	accidentals = accidental_order[:circle_of_fifths.index(str(key.tonic))]
	for a in accidentals:
		if note.pitch.pitchClass == a.pitchClass:
			note.name = a.name
			note.pitch.accidental.displayStatus = False
			return

def merge_lh_rh(lh_song, rh_song):
	# 
	# Merge LH and RH aspects of song into one Song
	#
	assert(len(lh_song) == len(rh_song))
	song = []
	for i in range(len(lh_song)):
		notes = [ note for note in lh_song[i] ] + [ note for note in rh_song[i] ]
		song.append(notes)
	return song

def song_to_m21(lh_song, rh_song):
	# 1. Calculate frame duration of 16th note
	song = merge_lh_rh(lh_song, rh_song)
	time_steps = song_to_timesteps(song)
	## If empty song, return
	if len(time_steps) == 0:
		return None
	## Remove noisy artifacts
	time_steps = remove_time_step_noise(time_steps)
	## Convert # of frames pressed -> M21 Durations for each note
	frames_per_16th = get_frames_per_16th(time_steps)

	# 2. Set-up Music21 stream
	score = m21.stream.Score()
	treble = m21.stream.Part()
	treble.append(m21.clef.TrebleClef())
	bass = m21.stream.Part()
	bass.append(m21.clef.BassClef())

	# 3. Map song -> "time_steps", which stores how long each note is held for / when it begins
	for (song, staff) in [ (lh_song, bass), (rh_song, treble) ]:
		time_steps, _ = frames_to_durations(song_to_timesteps(song), frames_per_16th)
		# Look through every time step...
		for i, t in enumerate(time_steps):
			for n in t:
				note = list(n.keys())[0] # "F4" or "A#3"
				duration = list(n.values())[0] # Music21 Duration object
				# Place note
				staff.insert(frames_to_16ths(i, frames_per_16th)/4, m21.note.Note(note, duration = duration))

	# 4. Add treble/bass clefs to overall score
	score.append(treble)
	score.append(bass)
	
	# 5. Set key signature and change notes to correct key (e.g. if Eb Major, then "A#" -> "Bb")
	p = m21.analysis.discrete.TemperleyKostkaPayne()
	key = p.getSolution(treble)
	for n in score.recurse().notes:
		convert_note_to_key(n, key) # Done in-place
	treble.keySignature = key
	bass.keySignature = key

	return score