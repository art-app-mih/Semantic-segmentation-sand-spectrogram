# Semantic-segmentation-sand-spectrogram
Semantic segmentation sand spectrogram

This repository contains programs that form part of my master's work. Here I perform semantic segmentation of the spectrograms in order to identify areas with sand ejection (a burst of energy in the high-frequency region on the acoustic signals received from the sensor). Steps:

1- First, I collected a database from spectrograms where there is a signal with sand (using the library
Signal Processing Toolbox in MATLAB).

2 - Then I annotated the data using Labelme (it is a graphical image annotation tool inspired by http://labelme.csail.mit.edu.
It is written in Python and uses Qt for its graphical interface).

3- Using opencv, I resized the image to 512 x 512 (resized.py).

4 -Then I created a DataLoaderSegmentation.py, the data in which is converted to the TORCH.UTILS.DATA format (necessary to load data into a neural network written in pytorch).

5. Then I wrote a neural network of the Unet architecture, which shows some of the best results for image segmentation (Unet.py)

6- The neural network training process is described in traner.py.
