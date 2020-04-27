import cv2
import numpy as np
from threading import Thread
from queue import Queue
import music21 as m21
import copy
import pickle


#
# Convert Synthesia video -> Notes played during video
#

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

def pixels_to_notes(grey, trim_start, trim_end, first_white_note, first_black_note):
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

	#  Map keys -> notes
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
	
	# Get all black/white pixels (greyscale means R = G + B)
	unpressed_pixels = frame.max(axis = 2) - frame.min(axis = 2) < 15
	# "colored_pixels" is an array the same size as frame, but with all grey pixels -> black and colored -> white
	colored_pixels = np.full(frame.shape[:2], 1).astype(np.uint8)
	colored_pixels[ unpressed_pixels ] = 0
	# Count up notes coresponding to each pixel that was pressed
	colored_notes = notes.copy()
	colored_notes[colored_pixels == 0] = None
	unique, counts = np.unique(colored_notes, return_counts=True)
	note_counts = dict(zip(unique, counts))
	# For visual display
	if animate:
		only_colored_pixels = np.full(frame.shape, 255).astype(np.uint8)
		only_colored_pixels[ unpressed_pixels ] = [0, 0, 0]
		cv2.imshow("Keys to Notes", only_colored_pixels)
		cv2.waitKey(0)
	return note_counts

def predict_pressed_notes(note_counts, total_note_counts, note_threshold = 0.8):
	# Calculate what percent of a note's pixels were hit
	percent_of_notes_hit = { n: note_counts[n]/total_note_counts[n] for n in note_counts.keys() }
	percent_of_notes_hit.pop('None', None)
	# Note is "hit" if >note_threshold percent of the pixels corresponding to it were pressed
	notes_hit = [ k for k, v in percent_of_notes_hit.items() if v > note_threshold ]
	return notes_hit

def synethesia_to_notes(fvs, first_white_note = 'A', first_black_note = 'A#', animate = False):
	NOTE_THRESHOLD = 0.8 # Percent of note's pixels that must be hit to "count"
	song = [] # List of notes hit at every frame
	firstLoop = True
	tracker = []
	while not fvs.stopped:
		orig_frame = fvs.read()
		(height, width, rgb) = orig_frame.shape
		# Crop image to bottom third
		frame = orig_frame[(height*2)//2:, :, :].copy()
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
			notes, total_note_counts = pixels_to_notes(cut_grey, trim_start, trim_end, first_white_note, first_black_note)

		# Map pressed keys -> notes
		note_counts = pressed_keys_to_notes(notes, trim_frame)
		# Map pressed notes -> overall estimate of pressed notes
		notes_hit = predict_pressed_notes(note_counts, total_note_counts, note_threshold = NOTE_THRESHOLD)
		## Append notes to overall song
		song.append(notes_hit)
		# Display the resulting grey
		if animate:
			cv2.imshow('frame', frame)
			cv2.waitKey(0)
	if animate:
		cv2.destroyAllWindows()
	return song
