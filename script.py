import cv2
import numpy as np
from threading import Thread
from queue import Queue
import sys
import collections
import music21 as m21
from time import time
from collections import Counter
import copy
import logging, pickle

np.random.seed(1)
log = logging.getLogger(__name__)

class FileVideoStream:
	def __init__(self, path, queueSize = 128):
		# initialize the file video stream along with the boolean
		# used to indicate if the thread should be stopped or not
		self.stream = cv2.VideoCapture(path)
		self.stopped = False
		# initialize the queue used to store frames read from
		# the video file
		self.Q = Queue(maxsize = queueSize)
	def start(self):
		# start a thread to read frames from the file video stream
		t = Thread(target = self.update, args = ())
		t.daemon = True
		t.start()
		return self
	def update(self):
		# keep looping infinitely
		while True:
			# if the thread indicator variable is set, stop the
			# thread
			if self.stopped:
				return
			# otherwise, ensure the queue has room in it
			if not self.Q.full():
				# read the next frame from the file
				(grabbed, frame) = self.stream.read()
				# if the `grabbed` boolean is `False`, then we have
				# reached the end of the video file
				if not grabbed:
					self.stop()
					return
				# add the frame to the queue
				self.Q.put(frame)
	def read(self):
		# Get next frame in queue
		return self.Q.get()
	def more(self):
		# return True if there are still frames in the queue
		return self.Q.qsize() > 0
	def stop(self):
		# indicate that the thread should be stopped
		self.stopped = True
		self.stream.release()

def get_before_black_marks(row):
	# Determine if "black" marks are "keys" or "dividers btwn white notes"
	kl = 0
	kls = [] 
	for i in range(len(row)):
		if row[i] == 255:
			# On black key or space between white keys
			kl += 1
		elif kl > 0:
			# On white key
			kls.append(kl)
			kl = 0 # Reset key length
	max_kl = max(kls)
	before_blacks = ''.join([ 'W' if k < max_kl/2 else 'B' for k in kls ])
	return before_blacks

def next_white_note(white_note):
	mapper = {
		'A' : 'B',
		'B' : 'C',
		'C' : 'D',
		'D' : 'E',
		'E' : 'F',
		'F' : 'G',
		'G' : 'A',
	}
	return mapper.get(white_note, None)

def next_black_note(black_note):
	mapper = {
		'A#' : 'C#',
		'C#' : 'D#',
		'D#' : 'F#',
		'F#' : 'G#',
		'G#' : 'A#',
	}
	return mapper.get(black_note, None)

def next_same_color_note(note):
	if next_white_note(note) is None:
		return next_black_note(note)
	return next_white_note(note)

def map_keys_to_notes(notes, keys, first_note):
	new_notes = notes.copy()
	for i in range(keys.shape[0]):
		current_note = first_note
		octave = 0 if current_note != 'C' else 1
		for j in range(keys.shape[1]):
			if keys[i,j] == 255:
				# Highlighted pixel (white), so must be a note...
				new_notes[i,j] = current_note + str(octave)
			elif j-1 >= 0 and keys[i,j-1] == 255:
				# Just switched to black, update note
				current_note = next_same_color_note(current_note)
				if current_note == 'C' or current_note == 'C#':
					octave += 1 # Octaves increase at each "C"
	return new_notes

def pixels_to_notes(grey, trim_start, trim_end):
	# Map pixels -> black keys (do black first b/c easier to isolate than white keys)
	_, black_keys = cv2.threshold(grey, 225, 255, cv2.THRESH_BINARY_INV)
	## Fill in black keys
	black_keys = cv2.erode(black_keys, np.ones((5,5), np.uint8), iterations = 1)
	black_keys[black_keys.shape[0]*3//4:,:] = 0 # Bottom 1/4 of keys must not be black keys

	# Map pixels -> white keys
	## Find distance between each white key
	_, white_keys = cv2.threshold(grey[grey.shape[0]*4//5], 220, 255, cv2.THRESH_BINARY)
	white_edges = cv2.Canny(white_keys, 100, 200) # Ensure thin edges
	length, lengths = 0, []
	for pixel in white_edges.flatten():
		if pixel == 255:
			lengths.append(length)
			length = 0
		else:
			length += 1
	white_key_dist = np.bincount(lengths).argmax()
	### Add lines to inverted black_keys
	white_keys = cv2.bitwise_not(black_keys)
	for i in range(0, white_keys.shape[1], white_key_dist+1):
		white_keys[:, i] = 0

	# Map keys -> notes
	## Figure out what note piano starts at
	first_white_note, first_black_note = 'A', 'A#'
	## Map keys -> notes
	notes = np.full((trim_end - trim_start, grey.shape[1]), None).astype("<U4")
	notes = map_keys_to_notes(notes, black_keys[trim_start:trim_end], first_black_note)
	notes = map_keys_to_notes(notes, white_keys[trim_start:trim_end], first_white_note)
	unique, counts = np.unique(notes, return_counts=True)
	total_note_counts = dict(zip(unique, counts))

	# Noise reduction 
	black_note_max_count = np.max([ val for key, val in total_note_counts.items() if '#' in key])
	white_note_max_count = np.max([ val for key, val in total_note_counts.items() if '#' not in key])
	## Filter out notes with low counts
	for key, val in list(total_note_counts.items()):
		if '#' in key and val < black_note_max_count //2: total_note_counts.pop(key, None)
		if '#' not in key and val < white_note_max_count //2: total_note_counts.pop(key, None)
	return notes, total_note_counts

def pressed_keys_to_notes(notes, frame, animate = False):
	tracker = { 'A' : 0, 'B' : 0, 'C' : 0, 'D' : 0, 'E' : 0, 'F' : 0, 'G' : 0 }
	start = time()
	def reset_time():
		nonlocal start
		a = start
		start = time()
		return a
	
	# Get all black/white pixels (greyscale means R = G + B)
	unpressed_pixels = frame.max(axis = 2) - frame.min(axis = 2) < 15
	tracker['A'] += time() - reset_time()
	# "colored_pixels" is an array the same size as frame, but with all grey pixels -> black and colored -> white
	colored_pixels = np.full(frame.shape[:2], 1).astype(np.uint8)
	tracker['B'] += time() - reset_time()
	colored_pixels[ unpressed_pixels ] = 0
	# Count up notes coresponding to each pixel that was pressed
	colored_notes = notes.copy()
	tracker['C'] += time() - reset_time()
	colored_notes[colored_pixels == 0] = None
	unique, counts = np.unique(colored_notes, return_counts=True)
	tracker['D'] += time() - reset_time()
	note_counts = dict(zip(unique, counts))
	tracker['E'] += time() - reset_time()
	# For visual display
	if animate:
		only_colored_pixels = np.full(frame.shape, 255).astype(np.uint8)
		tracker['F'] += time() - reset_time()
		only_colored_pixels[ unpressed_pixels ] = [0, 0, 0]
		tracker['G'] += time() - reset_time()
		cv2.imshow("Keys to Notes", only_colored_pixels)
		cv2.waitKey(0)
	return note_counts, tracker

def predict_pressed_notes(note_counts, total_note_counts, note_threshold = 0.8):
	# Calculate what percent of a note's pixels were hit
	percent_of_notes_hit = { n: note_counts[n]/total_note_counts[n] for n in note_counts.keys() }
	percent_of_notes_hit.pop('None', None)
	# Note is "hit" if >note_threshold percent of the pixels corresponding to it were pressed
	notes_hit = [ k for k, v in percent_of_notes_hit.items() if v > note_threshold ]
	return notes_hit

#
# Convert Synethesia video -> Notes played during video
#
def synethesia_to_notes(fvs, animate = False):
	NOTE_THRESHOLD = 0.8 # Percent of note's pixels that must be hit to "count"
	song = [] # List of notes hit at every frame
	firstLoop = True
	tracker = []
	while not fvs.stopped:
		orig_frame = fvs.read()
		(height, width, rgb) = orig_frame.shape
		# Crop image to bottom third
		frame = orig_frame[(height*2)//3:, :, :].copy()
		# Gray scale
		grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# Trim frame to focus on keyboard
		if firstLoop:
			## Find piano top y by detecting white key tops
			_, black_keys = cv2.threshold(grey, 240, 255, cv2.THRESH_BINARY_INV)
			black_keys = cv2.erode(black_keys, np.ones((5,5), np.uint8), iterations = 1)
			white_keys = cv2.bitwise_not(black_keys)
			edges = cv2.Canny(white_keys, 100, 200)
			piano_top_y = np.min(np.argmax(edges, axis = 0) + 10)
		## Trim to top of keyboard
		cut_frame = frame[piano_top_y:]
		cut_grey = grey[piano_top_y:]
		## Focus on middle half of keyboard (to reduce compute time needed to match notes)	
		mid_frame = cut_frame.shape[0]//2
		trim_start = mid_frame - 10
		trim_end = mid_frame + 10
		trim_frame = cut_frame[trim_start:trim_end]

		# Map pixels -> notes (only need to do this once)
		if firstLoop:
			firstLoop = False
			notes, total_note_counts = pixels_to_notes(cut_grey, trim_start, trim_end)

		# Map pressed keys -> notes
		note_counts, t = pressed_keys_to_notes(notes, trim_frame)
		# Map pressed notes -> overall estimate of pressed notes
		notes_hit = predict_pressed_notes(note_counts, total_note_counts, note_threshold = NOTE_THRESHOLD)
		## Append notes to overall song
		song.append(notes_hit)
		# Display the resulting grey
		tracker.append(Counter(t))
		if animate:
			cv2.imshow('frame', frame)
			cv2.waitKey(0)
	if animate:
		cv2.destroyAllWindows()
	return song

#
# Convert song -> Music21 Format
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

def print_clef(clef):
	for s in clef:
		if hasattr(s, 'pitch') and not s.isRest:
			print(s.pitch, s.duration, s.offset)
		elif hasattr(s, 'lyric') and s.isRest:
			print('Rest', s.duration, s.offset, s.lyric)

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

if False:
	# Get video
	fvs = FileVideoStream('highqual.mp4').start()
	# Get notes played in video
	song = synethesia_to_notes(fvs, animate = False)

	with open('song.pickle', 'wb') as fd:
		pickle.dump(song, fd)
else:
	with open('song.pickle', 'rb') as fd:
		song = pickle.load(fd)

# Convert song to Music21 object
score = song_to_m21(song)

# Print sheet music
score.show()
