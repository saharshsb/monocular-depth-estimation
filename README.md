# Monocular Depth Estimation using YOLOv5 and MiDaS

The Monocular Depth Estimation model is a software based on deep learning that utilizes cutting-edge computer vision algorithms to estimate the depth of objects using a single RGB image and generates its corresponding depth image.

The model is designed to mainly assist in autonomous driving as well as ADAS, where it accurately estimates the depth of obstacles in a scene, without requiring the use of additional sensors or hardware such as RADAR or LiDAR systems. It works by taking a single RGB image or video frame as input and applying an MDE algorithm to generate the corresponding depth map, using which the depth of obstacles can be determined. It also features pre-processing and post-processing techniques to improve the accuracy and quality of the depth estimation.

The MDE model is highly scalable and can process input images or videos of various resolutions and sizes, without any constraints on the number of objects in the scene. We do, however, impose an artificial limit on the number of objects detected to simplify the model.

Overall, the Monocular Depth Estimation model is an essential tool not just for the automotive industry, but any industry that requires accurate depth estimation from a single image or video. The model can be integrated into various applications and systems, providing reliable depth information to aid in decision making and analysis.
