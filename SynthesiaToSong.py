import cv2
import numpy as np
from threading import Thread
from queue import Queue
import music21 as m21
import copy
import pickle
import imutils

#
# Convert Synthesia video -> Notes played during video
#

class FileVideoStream:
	def __init__(self, path, queueSize = 128):
		# initialize the file video stream along with the boolean
		# used to indicate if the thread should be stopped or not
		self.stream = cv2.VideoCapture(path)
		# self.total_frames = count_frames(path, override = False)
		self.stopped = False
		self.frames_returned = 0 # Count how many frames have been returned
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
		self.frames_returned += 1
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
	white_start = None
	white_length, white_lengths = 0, [] # Track length of white keys
	white_sep_length, white_sep_lengths = 0, [] # Track length of black space between white keys
	for pixel in white_keys.flatten():
		if pixel == 255:
			white_length += 1
			if white_sep_length > 0:
				white_sep_lengths.append(white_sep_length)
				white_sep_length = 0
			if white_start == None: white_start = pixel
		else:
			white_sep_length += 1
			if white_length > 0:
				white_lengths.append(white_length)
				white_length = 0
	white_key_dist = np.bincount(white_lengths).argmax()
	white_sep_dist = np.bincount(white_sep_lengths).argmax()
	### Add lines to inverted black_keys
	white_keys = cv2.bitwise_not(black_keys)
	for i in range(0, white_keys.shape[1], white_key_dist + white_sep_dist):
		white_keys[:, i] = 0

	#  Map keys -> notes
	notes = np.full((trim_end - trim_start, grey.shape[1]), None).astype("<U4")
	notes = map_keys_to_notes(notes, black_keys[trim_start:trim_end], first_black_note)
	notes = map_keys_to_notes(notes, white_keys[trim_start:trim_end], first_white_note)
	unique, counts = np.unique(notes, return_counts=True)
	total_note_counts = dict(zip(unique, counts))
	total_note_counts['None'] = np.inf # Add "None" for corner cases
	
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
	unpressed_pixels = frame.max(axis = 2) - frame.min(axis = 2) < 40
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

def show_frame(frame):
	cv2.imshow('frame', frame)
	cv2.waitKey(0)

def detect_keyboard(edges, template_edges, last_val, animate = False):
	# Detects if a keyboard is present in this frame's edges
	## Loop over the scales of the image (start with 100%)...
	found = None
	for scale in np.linspace(0.2, 1.0, 5)[::-1]:
		## Resize the image according to the scale, and keep track
		## ...of the ratio of the resizing
		resized = imutils.resize(edges, width = int(edges.shape[1] * scale))
		r = edges.shape[1] / float(resized.shape[1])
		## If the resized image is smaller than the template, then break
		## ...from the loop
		if resized.shape[0] < template_edges.shape[0] or resized.shape[1] < template_edges.shape[1]:
			break
		# Detect edges in the resized, grayscale image and apply template
		# ...matching to find the template in the image
		result = cv2.matchTemplate(resized, template_edges, cv2.TM_CCORR_NORMED)
		(_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
		# If we have found a new maximum correlation value, then update
		# the bookkeeping variable
		if found is None or maxVal > found[0]:
			found = (maxVal, maxLoc, r)
	# Unpack the bookkeeping variable and compute the (x, y) coordinates
	# ...of the bounding box based on the resized ratio
	(maxVal, maxLoc, r) = found
	(startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
	(endX, endY) = (int((maxLoc[0] + template_edges.shape[0]) * r), int((maxLoc[1] + template_edges.shape[1]) * r))
	# Draw a bounding box around the detected result and display the image
	if True or animate:
		clone = np.dstack([edges, edges, edges])
		cv2.rectangle(clone, (startX, startY), (endX, endY), (0, 0, 255), 2)
		print(maxVal, (startX, startY, endX, endY), r)
		show_frame(clone)
	return ((maxVal > 0.995) or (maxVal > 0.9 and maxVal == last_val), maxVal)

def synthesia_to_notes(fvs, template_img, first_white_note = 'A', first_black_note = 'A#', animate = False, logging = False):
	#
	# NOTE: This ignores "template_img" for now and assumes video starts with keyboard showing
	#
	NOTE_THRESHOLD = 0.8 # Percent of note's pixels that must be hit to "count"
	song = [] # List of notes hit at every frame
	firstLoop = True
	tracker = []
	# Get template image (for matching to keyboard)
	template_edges = cv2.cvtColor(cv2.imread(template_img), cv2.COLOR_BGR2GRAY)
	template_max_val = 0
	first_note_found = False
	while not fvs.stopped:
		orig_frame = fvs.read()
		(height, width, rgb) = orig_frame.shape
		# Crop image to bottom third
		frame = orig_frame[height//2:, :, :].copy()
		# Gray scale
		grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		try:
			# Trim frame to focus on keyboard
			if firstLoop:
				## Find piano top y by detecting white key tops
				_, black_keys = cv2.threshold(grey, 240, 255, cv2.THRESH_BINARY_INV)
				black_keys = cv2.erode(black_keys, np.ones((5,5), np.uint8), iterations = 1)
				white_keys = cv2.bitwise_not(black_keys)
				edges = cv2.Canny(white_keys, 100, 200)
				piano_top_y = np.min(np.argmax(edges, axis = 0) + 10)
				piano_bottom_y = np.max(np.argmax(edges, axis = 0) - 10)
				## Ignore this frame if keyboard hasn't appeared yet (e.g. title screen or fade in)
				# (is_keyboard_present, template_max_val) = detect_keyboard(edges, template_edges, 
				# 											template_max_val, animate = False)
				# if not is_keyboard_present:	
				# 	continue
			# Trim frame to top of keyboard
			cut_frame = frame[piano_top_y:piano_bottom_y]
			cut_grey = grey[piano_top_y:piano_bottom_y]
			## Focus on middle half of keyboard (to reduce compute time needed to match notes)	
			mid_frame = cut_frame.shape[0]//2
			trim_start = mid_frame - 10
			trim_end = mid_frame + 10
			trim_frame = cut_frame[trim_start:trim_end]

			# Map pixels -> notes (only need to do this once)
			if firstLoop:
				if logging: print(" - Keyboard first detected @ frame:", fvs.frames_returned)
				notes, total_note_counts = pixels_to_notes(cut_grey, trim_start, trim_end, first_white_note, first_black_note)
				firstLoop = False

			# Map pressed keys -> notes
			note_counts = pressed_keys_to_notes(notes, trim_frame)
			# Map pressed notes -> overall estimate of pressed notes
			notes_hit = predict_pressed_notes(note_counts, total_note_counts, note_threshold = NOTE_THRESHOLD)
			## Append notes to overall song
			song.append(notes_hit)

			# Logging/Animating
			## Display the frame
			if animate:
				show_frame(frame)
			## Logging
			if len(notes_hit) > 0 and first_note_found is False: 
				if logging: print(" - Note first detected @ frame:", fvs.frames_returned)
				first_note_found = True
			if fvs.frames_returned % 1000 == 0:
				if logging: print("    * Done processing frame", fvs.frames_returned)
		except Exception as e:
			print("Error on frame:", fvs.frames_returned, str(e))
			continue
	if animate:
		cv2.destroyAllWindows()
	return song
