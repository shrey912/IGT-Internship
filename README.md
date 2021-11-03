# IGT-Internship
Personality based Psychometric Analysis using Deep Bimodal Regression

This is the solution to the problem "Psychometric Analysis using Audio-Visual Profiling" given to me. The solution is the embodiment of the theory presented in the paper [Deep Bimodal Regression for Apparent Personality Analysis](https://cs.nju.edu.cn/wujx/paper/eccvw16_APA.pdf)

This problem is a challenge on “first impressions”, in which I had to collate audio and visual inputs to come up with a solution that creates an analysis of the personality based on innumerable features like modulation, pitch, tempo, gestures, stutter, vocabulary etc. The project involves Deep Learning framework for incorporating all these features in proper weightage to achieve the best accuracy possible.

While cognitive tests are also a part of psychometric analysis, but they are flat and static. So this project addresses the other paradigm of psychometric test namely personality analysis.

The traits to be recognized will correspond to the “big five” personality traits used in psychology and well known of hiring managers using standardized personality profiling:
* Extroversion
* Agreeableness
* Conscientiousness
* Neuroticism
* Openness to experience.

While the solution is applied to short first impression videos it performs well enough when it comes to longer interviews as well. The model is trained for 1.5 Lakh seconds of footage and at almost a rate of 6.1 fps meaning 10 Lakh frames of interviews are taken into consideration.

The model used is called `Descriptor Aggregation Network` called DAN in short.

![Model Archi](/Images/DAN.png)

What distinguishes DAN from the traditional CNN is: the fully connected layers are discarded, and replaced by both average- and max-pooling following the last convolutional layers (Pool5). Meanwhile, each pooling operation is followed by the standard L2-normalization. After that, the obtained two 512-d feature vectors are concatenated as the final image representation. Thus, in DAN, the deep descriptors of the last convolutional layers are aggregated as a single visual feature. Finally, a regression (fc+sigmoid) layer is added for end-to-end training.



## Getting Started 

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

* [Python3](https://www.python.org/downloads/release/python-373/) - Python version 3.7.3
* [Numpy](http://www.numpy.org/) - Multidimensioanl Mathematical Computing
* [Tensorflow 1.14.0](https://www.tensorflow.org/) - Deep Learning python module
* [Pandas](https://pandas.pydata.org/) - Loading csv files
* [Cha-Learn Dataset](http://chalearnlap.cvc.uab.es/dataset/24/description/) - Dataset for this problem
* [Pretrained VGG-Face Model](http://www.vlfeat.org/matconvnet/models/vgg-face.mat) Pretrained Vgg-face model
* [Pillow 6.1.0](https://pypi.org/project/Pillow/) Python Imaging Library
* [OpenCV 3.4.1](https://breakthrough.github.io/Installing-OpenCV/)  library used for Image Processing
* [ffmpeg](https://ffmpeg.zeranoe.com/builds/ ) software suite of libraries and programs for handling video, audio, and other multimedia 
files and streams. Add this to your Path in environment variables.

### Installing

Clone the repository

```
git clone https://github.com/shrey912/IGT-Internship.git
```

[Downlad the dataset](https://chalearnlap.cvc.uab.cat/dataset/24/description/) and extract it into a new "data" directory with all 75 training zip files and 25 validation zip files. The extraction of those 75 training and 25 validation files has been coded in the Video_to_Image code file.

[Download](http://www.vlfeat.org/matconvnet/models/vgg-face.mat) Pretrained Vgg-face model and move it to the root directory

Run the Video_to_Image.py file. It extracts images from the videos involved and saves them in ImageData folder.

```
python Video_to_Image.py
```

Run the vid_to_wav.py file to extract audio(.wav) files from the videos and save it to a new VoiceData directory

```
python vid_to_wav.py
```
Run the following to serialize the records.

```
python Write_Into_TFRecords.py
```

Start the training by running the following command

```
python train.py
```


## Acknowledgments

* [paper](https://cs.nju.edu.cn/wujx/paper/eccvw16_APA.pdf) - Implemented paper
* [TfRecord Data Pipeline](http://machinelearninguru.com/deep_learning/data_preparation/tfrecord/tfrecord.html#read) - Used to make data pipeline
* [Seema Ma'am](https://www.linkedin.com/in/gangwarseema/) for the opportunity to intern at IGT solutions and for assigning me this project
