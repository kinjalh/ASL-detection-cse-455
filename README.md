# ASL-detection-cse-455
Final project for CSE 455 - sign language detection primarily using NN

# Video Submission

[Video Link](https://youtu.be/A1BSlHbufRE)

# The Problem

We were interested in solving a problem that could benefit the deaf and hard of hearing since they are often a group that is left out from the equation when it comes to technology. We also read that American Sign Language (ASL) is one of the top 5 minority languages in America right after Spanish, Italian, German and French. So it made sense to try to develop some sort of computer vision system to help understand such a complex gestural language. The problem we are solving is detecting ASL alphabet hand symbols in real time from video. This can have many uses such as translating to text for sign language transcription and potentially to translate to voice for people who don’t necessarily understand sign language.

# The Dataset

We used an MNIST Sign Language Dataset from Kaggle found here: [dataset](https://www.kaggle.com/datamunge/sign-language-mnist). The data is a CSV file of pixel data, each row pertaining to a 28x28 pixel image. The training set has 27,455 images while the test set has 7172 images. Each row also has an associated label that is between 0-25 to denote which letter is being signed. The dataset lacks any images for J or Z because they require movement of the hand which can't be captured from a single still image.

# Our Approach

We decided to compare two different neural net approaches, one that had some preprocessing done to the images and one that was a complete end to end CNN. This way we could see how preprocessing the images might affect the ability of the network to detect the sign. Our hypothesis was that the preprocessing might help make the network better at detecting the signs in real time since it could help with isolating the hand signs from the background.

# Results

With our end to end CNN we were able to get an accuracy of about 82% on the testing data, whereas with our preprocessing + NN model we got an accuracy of 74%. Both had high accuracy on training data (above 95%), so both overfit to the training set although to different extents. The CNN likely overfit less since trainable convolution layers combined with pooling layers help extract features even if there are some minor differences in images, while the non-convolutional ANN relies on exact pixels being certain values (or very close to certain values).

As for our real time predictions, they weren't as accurate as we had hoped for either model. We think this is probably because the images that we trained off of were taken in very specific settings, and the variation in skin tones and angles were minimal so it's likely that the model was not exposed to enough differences to be able to distinguish the sign accurately from our varying backgrounds and skin tones. 

# Next Steps

For our next steps, we would like to work on adding more data to the dataset that makes it more representative of different groups and with varying backgrounds so that the model can accurately distinguish the hand sign from the rest of the image. We would also like to try adding in other datasets, such as this [one](https://dxli94.github.io/WLASL/) that contains signs for words instead of just letters so that we can interpret a larger variety of signage.
