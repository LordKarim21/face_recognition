import cv2
from matplotlib import pyplot as pl
from tkinter import Tk
from tkinter.filedialog import askopenfilename

Tk().withdraw()

img = cv2.imread(askopenfilename())
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = cv2.CascadeClassifier('faces.xml')

res = faces.detectMultiScale(gray, scaleFactor=2, minNeighbors=1)

for (x, y, w, h) in res:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), thickness=3)

img = cv2.resize(img, (500, 500))

pl.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
pl.show()
