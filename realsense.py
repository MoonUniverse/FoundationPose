## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#####################################################
##              Align Depth to Color               ##
#####################################################

# First import the library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2

import os

# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

depth_stream = profile.get_stream(rs.stream.depth)
depth_intr = depth_stream.as_video_stream_profile().get_intrinsics()
print("depth_intr: ", depth_intr)

color_stream = profile.get_stream(rs.stream.color)
color_intr = color_stream.as_video_stream_profile().get_intrinsics()
print("color_intr: ", color_intr)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 1 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

save_images = False
image_count = 1
frame_counter = 0

def show_pixel_value(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:  # 当鼠标移动时
        # 获取鼠标位置的像素值
        pixel_value = depth_image[y, x]
        print(f"Pixel value at ({x}, {y}): {pixel_value}")
        
        
cv2.namedWindow('depth_image')
cv2.setMouseCallback('depth_image', show_pixel_value)  
# Streaming loop
try:
    while True:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        cv2.imshow('depth_image', depth_image)
        cv2.imshow('color_image', color_image)
        
        if save_images:
            frame_counter += 1
        
        key = cv2.waitKey(1)
        
        if key & 0xFF == ord('s'):
            save_images = True
            print("开始保存图像")
        elif key & 0xFF == ord('e'):
            save_images = False
            print("停止保存图像")
            
        
            
        # 保存图像
        if save_images and frame_counter % 3 == 0:
            # 格式化图像文件名
            depth_image_filename = os.path.join("C:/Users/49563\Desktop/lol/depth/",f'{image_count:07d}.png')
            color_image_filename = os.path.join("C:/Users/49563\Desktop/lol/rgb/",f'{image_count:07d}.png')

            # 保存深度图像和彩色图像
            cv2.imwrite(depth_image_filename, depth_image)
            cv2.imwrite(color_image_filename, color_image)

            # 更新图像计数器
            image_count += 1

        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()
