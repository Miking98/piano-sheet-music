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
		octave = 0 if not (current_note == 'C' or current_note == 'C#') else 1
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

def recenter_octaves(notes, total_note_counts):
	#
	# Recenter octaves around Middle C (C4)
	#
	new_notes = notes.copy()
	new_total_note_counts = {}
	# Get total number of C notes
	num_Cs = len([ n for n in total_note_counts.keys() if n[0] == "C" and len(n) == 2])
	# nth of C notes -> Middle C (C4)
	middle_C = int(np.ceil(num_Cs / 2))
	# Adjust 1st of C notes to proper octave, so that nth is Middle C
	first_C_octave = 4 - middle_C + 1
	# Adjust all notes
	## "total_note_counts"
	for key, val in total_note_counts.items():
		if key == "None": 
			new_total_note_counts[key] = val
			continue
		current_octave = int(key[-1])
		adjusted_octave = current_octave + first_C_octave - 1
		new_total_note_counts[key[:-1] + str(adjusted_octave)] = val
	## "notes"
	for i in range(new_notes.shape[0]):
		for j in range(new_notes.shape[1]):
			if new_notes[i,j] == "None": continue
			current_octave = int(new_notes[i,j][-1])
			adjusted_octave = current_octave + first_C_octave - 1
			new_notes[i,j] = new_notes[i,j][:-1]+ str(adjusted_octave)
	return new_notes, new_total_note_counts

def pixels_to_notes(grey, trim_start, trim_end, first_white_note, first_black_note, animate = False):
	# Map pixels -> black keys (do black first b/c easier to isolate than white keys)
	_, black_keys = cv2.threshold(grey, 225, 255, cv2.THRESH_BINARY_INV)
	## Fill in black keys
	black_keys = cv2.erode(black_keys, np.ones((5,5), np.uint8), iterations = 1)
	black_keys[black_keys.shape[0]*3//4:,:] = 0 # Bottom 1/4 of keys must not be black keys

	# Map pixels -> white keys
	_, black_marks = cv2.threshold(grey, 225, 255, cv2.THRESH_BINARY_INV)
	## Only keep black-marks that touch top of piano (otherwise might be artifacts or marks on keys that we should ignore)
	for col in range(black_marks.shape[1]):
		## Start from top of keyboard, and go down until hit a black pixel -- zero out everything else
		column = black_marks[:,col]
		for pixel_idx, pixel in enumerate(column):
			if pixel != 255:
				# Found black pixel, zero out everything below this pixel
				black_marks[pixel_idx:,col] = 0
				break
	white_keys = cv2.bitwise_not(black_marks)
	if animate:
		show_frames([white_keys, black_keys], "WHITE | BLACK")

	#  Map keys -> notes
	notes = np.full((trim_end - trim_start, grey.shape[1]), None).astype("<U4")
	## Get letter for each black/white key
	notes = map_keys_to_notes(notes, black_keys[trim_start:trim_end], first_black_note)
	notes = map_keys_to_notes(notes, white_keys[trim_start:trim_end], first_white_note)
	## Get count of each note
	unique, counts = np.unique(notes, return_counts=True)
	total_note_counts = dict(zip(unique, counts))
	total_note_counts['None'] = np.inf # Add "None" for corner cases
	## Noise reduction 
	black_note_max_count = np.max([ val for key, val in total_note_counts.items() if '#' in key])
	white_note_max_count = np.max([ val for key, val in total_note_counts.items() if '#' not in key])
	## Filter out notes with low counts
	for key, val in list(total_note_counts.items()):
		if '#' in key and val < black_note_max_count //2: total_note_counts.pop(key, None)
		if '#' not in key and val < white_note_max_count //2: total_note_counts.pop(key, None)
	## Recenter around Middle C
	notes, total_note_counts = recenter_octaves(notes, total_note_counts)

	return notes, total_note_counts

def mark_pixels_by_closest_color(frame, color1, color2):
	color1_dists = np.linalg.norm(frame - color1, axis = 2)
	color2_dists = np.linalg.norm(frame - color2, axis = 2)
	color1_notes = color1_dists < color2_dists
	color2_notes = color1_dists >= color2_dists
	return color1_notes, color2_notes

def pressed_keys_to_notes(notes, frame, animate = False):
	# 1. Zero out all unpressed keys 
	## Black/white keys will be greyscale, so R = G + B
	unpressed_pixels = frame.max(axis = 2) - frame.min(axis = 2) < 40
	# new_frame_colored = frame.copy() # Only pressed keys are not zero'd out
	# new_frame_colored[ unpressed_pixels ] = [0,0,0] # Set R to 1 so that np.argmax() doesn't break ties with Blue in BGR
	# # 2. Mark blue and green pressed keys (blue -> left hand, green -> right hand)
	# lh_pixels = np.argmax(new_frame_colored, axis = 2) == 0 # BGR, B = index 0
	# rh_pixels = np.argmax(new_frame_colored, axis = 2) == 1 # BGR, G = index 1
	# 3. Count up pressed notes corresponding to each hand

	lh_pixels, rh_pixels = mark_pixels_by_closest_color(frame.copy(), [1, 0, 0], [0, 1, 0])

	## Blue
	lh_notes = notes.copy()
	lh_notes[lh_pixels == False] = None
	lh_notes[unpressed_pixels] = None
	lh_unique, lh_counts = np.unique(lh_notes, return_counts = True)
	lh_counts = dict(zip(lh_unique, lh_counts))
	## Green
	rh_notes = notes.copy()
	rh_notes[rh_pixels == False] = None
	rh_notes[unpressed_pixels] = None
	rh_unique, rh_counts = np.unique(rh_notes, return_counts = True)
	rh_counts = dict(zip(rh_unique, rh_counts))
	# 4. For visual display
	if animate:
		lh_keys = np.full(frame.shape, 255).astype(np.uint8)
		lh_keys[lh_pixels == False] = [0,0,0]
		lh_keys[unpressed_pixels] = [0,0,0]
		rh_keys = np.full(frame.shape, 255).astype(np.uint8)
		rh_keys[rh_pixels == False] = [0,0,0]
		rh_keys[unpressed_pixels] = [0,0,0]
		show_frames([frame, lh_keys, rh_keys], 'Frame | LH Keys | RH Keys')
	return lh_counts, rh_counts

def predict_pressed_notes(note_counts, total_note_counts, note_threshold = 0.8):
	# Calculate what percent of a note's pixels were hit
	percent_of_notes_hit = { n: note_counts[n]/total_note_counts[n] for n in note_counts.keys() }
	percent_of_notes_hit.pop('None', None)
	# Note is "hit" if >note_threshold percent of the pixels corresponding to it were pressed
	notes_hit = [ k for k, v in percent_of_notes_hit.items() if v > note_threshold ]
	return notes_hit

def show_frames(frames, title = 'Title'):
	cv2.imshow(title, np.vstack(frames))
	cv2.waitKey(0)

def show_frame(frame, title = 'frame'):
	cv2.imshow(title, frame)
	cv2.waitKey(0)

def synthesia_to_notes(fvs, first_white_note = 'A', first_black_note = 'A#', animate = False, logging = False):
	#
	# Converts Synthesia video that starts with keyboard showing to Song
	#
	NOTE_THRESHOLD = 0.6 # Percent of note's pixels that must be hit to "count"
	lh_song, rh_song = [], [] # List of notes hit at every frame for each hand
	firstLoop = True
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
				piano_bottom_y = np.max(np.argmin(edges, axis = 0) - 10)
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
				notes, total_note_counts = pixels_to_notes(cut_grey, trim_start, trim_end, first_white_note, first_black_note, animate = animate)
				firstLoop = False

			if fvs.frames_returned < 15:
				continue
			# Map pressed keys -> notes for left/right hands
			lh_counts, rh_counts = pressed_keys_to_notes(notes, trim_frame, animate = animate)
			# Map pressed notes -> overall estimate of pressed notes
			lh_notes_hit = predict_pressed_notes(lh_counts, total_note_counts, note_threshold = NOTE_THRESHOLD)
			rh_notes_hit = predict_pressed_notes(rh_counts, total_note_counts, note_threshold = NOTE_THRESHOLD)
			## Append notes to overall song
			rh_song.append(rh_notes_hit)
			lh_song.append(lh_notes_hit)

			# Logging/Animating
			## Display the frame
			if animate:
				show_frame(frame)
				print(lh_notes_hit, "|", rh_notes_hit)
			## Logging
			if len(lh_notes_hit + rh_notes_hit) > 0 and first_note_found is False: 
				if logging: print(" - Note first detected @ frame:", fvs.frames_returned)
				first_note_found = True
			if fvs.frames_returned % 1000 == 0:
				if logging: print("    * Done processing frame", fvs.frames_returned)
		except Exception as e:
			import traceback
			print("Error on frame:", fvs.frames_returned, str(e))
			traceback.print_exc()
			continue
	if animate:
		cv2.destroyAllWindows()
	return lh_song, rh_song
