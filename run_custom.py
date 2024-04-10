# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from estimater import *
from datareader import *
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
import argparse
import pyrealsense2 as rs

class MouseCallback:
    def __init__(self, img):
        self.img = img
        self.rect_start = (0, 0)
        self.rect_end = (0, 0)
        self.dragging = False

    def mouse_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.rect_start = (x, y)
            self.dragging = True
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.rect_end = (x, y)
            self.dragging = False
            cv2.rectangle(self.img, self.rect_start, self.rect_end, (0, 255, 0), 2)
            cv2.imshow('color_image', self.img)
        
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            temp_img = self.img.copy()
            cv2.rectangle(temp_img, self.rect_start, (x, y), (0, 255, 0), 2)
            cv2.imshow('color_image', temp_img)

if __name__=='__main__':
  parser = argparse.ArgumentParser()
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/station/mesh/3dpea_scaled_model.obj')
  parser.add_argument('--est_refine_iter', type=int, default=10)
  parser.add_argument('--track_refine_iter', type=int, default=2)
  parser.add_argument('--debug', type=int, default=1)
  parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
  args = parser.parse_args()

  set_logging_format()
  set_seed(0)

  mesh = trimesh.load(args.mesh_file)

  debug = args.debug
  debug_dir = args.debug_dir
  os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

  to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

  scorer = ScorePredictor()
  refiner = PoseRefinePredictor()
  glctx = dr.RasterizeCudaContext()
  est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
  logging.info("estimator initialization done")
  
  # segment anything
  sam = sam_model_registry["vit_h"](checkpoint="weights/sam_vit_h_4b8939.pth").to(device="cuda")
  predictor = SamPredictor(sam) # predictor is a function that takes an image and returns a mask
  
  # realsense init 
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
      config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
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
  camera_K = np.array([[color_intr.fx, 0, color_intr.ppx], [0, color_intr.fy, color_intr.ppy], [0, 0, 1]])

  # We will be removing the background of objects more than
  #  clipping_distance_in_meters meters away
  clipping_distance_in_meters = 1 #1 meter
  clipping_distance = clipping_distance_in_meters / depth_scale

  # Create an align object
  # rs.align allows us to perform alignment of depth frames to others frames
  # The "align_to" is the stream type to which we plan to align depth frames.
  align_to = rs.stream.color
  align = rs.align(align_to)
  
  it_since_init = False
  
  try:
    while True:
      frames = pipeline.wait_for_frames()
      # Align the depth frame to color frame
      aligned_frames = align.process(frames)
      # Get aligned frames
      aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
      color_frame = aligned_frames.get_color_frame()
      # Validate that both frames are valid
      if not aligned_depth_frame or not color_frame:
          continue 
      depth_image = np.asanyarray(aligned_depth_frame.get_data())/4000 #realsense depth scale is 4000
      color_image = np.asanyarray(color_frame.get_data()) 
      color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB) 
      
      if not it_since_init:
        it_since_init = True
        color_mask = color_image.copy()
        callback_obj = MouseCallback(color_image)
        cv2.imshow('color_image', color_image)
        cv2.setMouseCallback('color_image', callback_obj.mouse_event)
        cv2.waitKey(0)
        # get rect_start and rect_end
        rect_start = callback_obj.rect_start
        rect_end = callback_obj.rect_end
        print(f"rect_start: {rect_start}, rect_end: {rect_end}")
        input_box = np.array([rect_start, rect_end])
        predictor.set_image(color_mask)
        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )
        mask = masks[0]*255
        os.system(f'rm -rf {debug_dir}/mask.png')
        cv2.imwrite(f'{debug_dir}/mask.png', mask)
        pose = est.register(K=camera_K, rgb=color_image, depth=depth_image, ob_mask=mask, iteration=args.est_refine_iter)
      else:
        pose = est.track_one(rgb=color_image, depth=depth_image, K=camera_K, iteration=args.track_refine_iter)

      if debug>=1:
        center_pose = pose@np.linalg.inv(to_origin)
        print("center_pose: ", center_pose)
        vis = draw_posed_3d_box(camera_K, img=color_image, ob_in_cam=center_pose, bbox=bbox)
        vis = draw_xyz_axis(color_image, ob_in_cam=center_pose, scale=0.1, K=camera_K, thickness=3, transparency=0, is_input_rgb=True)
        cv2.imshow('1', vis[...,::-1])
        cv2.waitKey(1)


      
  finally:
    pipeline.stop()



