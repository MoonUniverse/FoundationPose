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
  parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/battery/mesh/2.obj')
  parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/demo_data/battery')
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

  reader = YcbineoatReader(video_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf)

  for i in range(len(reader.color_files)):
    logging.info(f'i:{i}')
    color = reader.get_color(i)
    depth = reader.get_depth(i)
    if i==0:
      # 创建MouseCallback对象
      color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
      color_mask = color.copy()
      callback_obj = MouseCallback(color)
      cv2.imshow('color_image', color)
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
      pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)

      if debug>=3:
        m = mesh.copy()
        m.apply_transform(pose)
        m.export(f'{debug_dir}/model_tf.obj')
        xyz_map = depth2xyzmap(depth, reader.K)
        valid = depth>=0.1
        pcd = toOpen3dCloud(xyz_map[valid], color[valid])
        o3d.io.write_point_cloud(f'{debug_dir}/scene_complete.ply', pcd)
    else:
      pose = est.track_one(rgb=color, depth=depth, K=reader.K, iteration=args.track_refine_iter)

    os.makedirs(f'{reader.video_dir}/ob_in_cam', exist_ok=True)
    np.savetxt(f'{reader.video_dir}/ob_in_cam/{reader.id_strs[i]}.txt', pose.reshape(4,4))

    if debug>=1:
      center_pose = pose@np.linalg.inv(to_origin)
      print("center_pose: ", center_pose)
      vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
      vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
      cv2.imshow('1', vis[...,::-1])
      cv2.waitKey(1)


    if debug>=2:
      os.makedirs(f'{reader.video_dir}/track_vis', exist_ok=True)
      imageio.imwrite(f'{reader.video_dir}/track_vis/{reader.id_strs[i]}.png', vis)

