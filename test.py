from aubio import source, tempo
import numpy as np

def get_file_bpm(path, params=None):
	""" Calculate the beats per minute (bpm) of a given file.
		path: path to the file
		param: dictionary of parameters
	"""
	if params is None:
		params = {}
	# default:
	samplerate, win_s, hop_s = 44100, 1024, 512
	if 'mode' in params:
		if params.mode in ['super-fast']:
			# super fast
			samplerate, win_s, hop_s = 4000, 128, 64
		elif params.mode in ['fast']:
			# fast
			samplerate, win_s, hop_s = 8000, 512, 128
		elif params.mode in ['default']:
			pass
		else:
			raise ValueError("unknown mode {:s}".format(params.mode))
	# manual settings
	if 'samplerate' in params:
		samplerate = params.samplerate
	if 'win_s' in params:
		win_s = params.win_s
	if 'hop_s' in params:
		hop_s = params.hop_s

	s = source(path, samplerate, hop_s)
	samplerate = s.samplerate
	o = tempo("specdiff", win_s, hop_s, samplerate)
	# List of beats, in samples
	beats = []
	# Total number of frames read
	total_frames = 0

	while True:
		samples, read = s()
		is_beat = o(samples)
		if is_beat:
			this_beat = o.get_last_s()
			beats.append(this_beat)
			#if o.get_confidence() > .2 and len(beats) > 2.:
			#	break
		total_frames += read
		if read < hop_s:
			break

	def beats_to_bpm(beats, path):
		# if enough beats are found, convert to periods then to bpm
		if len(beats) > 1:
			if len(beats) < 4:
				print("few beats found in {:s}".format(path))
			bpms = 60./np.diff(beats)
			return np.median(bpms)
		else:
			print("not enough beats found in {:s}".format(path))
			return 0

	return beats_to_bpm(beats, path)

bpm = get_file_bpm('biglittlelies.mp4')

print(bpm)
