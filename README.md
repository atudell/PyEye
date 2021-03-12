# PyEye

The purpose of PyEye is to demonstrate the use of the Haar Cascades classifier with the concept of homography to project an image onto the eyes of an individual in a video.
The result is a SnapChat-style filter and fun application of augmented reality. 

Note that whatever image is used must include transparency. Setting transparency on images is available an almost every major photo/image editing software. For demonstration, I included an image of sunglasses with a transparent background.

The function may simply be called as:
projectOnEyes(image_src, window_name = "projectOnEyes")

where image_src denotes the source to the image to project onto the eyes and window_name denotes the name of the resulting window.


The below presents a demonstration of use:
![sunglasses demo 1](https://user-images.githubusercontent.com/50125339/110989236-94f20780-833f-11eb-97b5-508dd47a6911.gif)
