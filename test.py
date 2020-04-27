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

np.random.seed(1)

# Set-up Music21 stream
score = m21.stream.Score()
treble = m21.stream.Part()
treble.append(m21.clef.TrebleClef())
bass = m21.stream.Part()
bass.append(m21.clef.BassClef())

# Add notes
for i in range(1000):
	treble.insert(i/4, m21.note.Note('A3' if i%2 == 0 else 'D#2', duration = m21.duration.Duration(0.25)))
	if i%2 == 0:
		treble.insert(i/4, m21.note.Note('F3', duration = m21.duration.Duration(0.25)))
# Add treble/bass clefs to overall score
score.append(treble)

score.show()
