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

def split_into_treble_bass(time_steps, song):
	# If two notes are this many semitones apart and played at the same time, put them on opposite clefs
	OCTAVE_SEMITONE_THRESHOLD = 15
	# Get all unique notes in song
	unique_notes = get_unique_notes_in_song(song)
	# Find middle octave
	middle_octave = np.round(np.mean([ int(n[-1]) for n in unique_notes ])) # Octave is last char in note
	# Arrays to hold treble/bass notes
	treble_time_steps = [ [] for s in song ]
	bass_time_steps = [ [] for s in song ]

	# For each time step...
	for i, t in enumerate(time_steps):
		# Get 2 farthest notes currently being played
		max_note_dist, low_note, high_note = 0, None, None
		for a in song[i]:
			for b in song[i]:
				max_note_dist, low_note, high_note = get_semitone_dist(a, b, return_low_high = True) if get_semitone_dist(a, b) > max_note_dist else (max_note_dist, low_note, high_note)
		for n in t:
			# Get note info
			note = list(n.keys())[0] # "F4" or "A#3"
			octave = int(note[-1]) # "4"
			frames = n[note] # 4 (# of frames was pressed)
			# Determine if notes are treble/bass
			## (1) Guess based on octave: >= octave_middle -> treble
			clef = 'T' if octave >= middle_octave else 'B'
			## (2) Overwrite if low/high note are more than OCTAVE_SEMITONE_THRESHOLD apart, then separate into clefs based on proximity to these notes
			if OCTAVE_SEMITONE_THRESHOLD <= max_note_dist:
				clef = 'B' if get_semitone_dist(note, low_note) < get_semitone_dist(note, high_note) else 'T'
			# Add note to clef's time_step and song arrays
			ts = bass_time_steps if clef == 'B' else treble_time_steps
			ts[i].append(n)
	return treble_time_steps, bass_time_steps

def frames_to_durations(time_steps):
	time_steps = copy.deepcopy(time_steps)

	# Get # of frames each note is pressed
	frames_pressed = []
	for t in time_steps:
		for n in t:
			frames_pressed.append(list(n.values())[0])
	
	# Call most often-repeated frame_length an 8th note
	frames_per_16th = np.bincount(frames_pressed).argmax()/2

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

def song_to_m21(song):
	# Map song -> "time_steps", which stores how long each note is held for / when it begins
	time_steps = song_to_timesteps(song)
	## If empty song, return
	if len(time_steps) == 0:
		return None

	# Remove noisy artifacts
	time_steps = remove_time_step_noise(time_steps)

	# Convert # of frames pressed -> M21 Durations for each note
	time_steps, frames_per_16th = frames_to_durations(time_steps)

	# Split "time_steps" into treble/bass clefs
	treble_time_steps, bass_time_steps = split_into_treble_bass(time_steps, song)

	# Set-up Music21 stream
	score = m21.stream.Score()
	treble = m21.stream.Part()
	treble.append(m21.clef.TrebleClef())
	bass = m21.stream.Part()
	bass.append(m21.clef.BassClef())

	# Add notes
	for clef in ['T', 'B']:
		ts = bass_time_steps if clef == 'B' else treble_time_steps
		staff = bass if clef == 'B' else treble
		# Look through every time step...
		for i, t in enumerate(ts):
			######
			# IT FREAKS OUT B/C OF ADDITION OF THIS REST:
			# OffsetMap(element=<music21.note.Rest rest>, offset=2.0, endTime=79.25, voiceIndex=1)
			#
			# Changed music21package/stream/__init__.py:9471
			#####
			for n in t:
				note = list(n.keys())[0] # "F4" or "A#3"
				duration = list(n.values())[0] # Music21 Duration object
				# Place note
				staff.insert(frames_to_16ths(i, frames_per_16th)/4, m21.note.Note(note, duration = duration))
	# Add treble/bass clefs to overall score
	score.append(treble)
	score.append(bass)
	return score