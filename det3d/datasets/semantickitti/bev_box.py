import numpy as np
import open3d as o3d
from yaml.events import NodeEvent
import cv2
import math
import copy
from .fitting import BBoxFitting


class BEVBoundingBox3D():
  def __init__(self, center, size, yaw, label=None) -> None:
    # super().__init__()

    self.center = center
    self.size = size
    self.extent = size
    self.yaw = yaw

    # left is x, front is y
    self.left = np.array([np.cos(self.yaw), -np.sin(self.yaw), 0])
    self.front = np.array([np.sin(self.yaw), np.cos(self.yaw), 0])
    self.up = np.array([0., 0., 1.])

    self.label_class = label
    self.arrow_length = 1.0
    self.velocity = np.zeros(3) #vx, vy, vz

  @staticmethod
  def create_from_points(points, method=None):
    # super().__init__()

    bbox_fit = BBoxFitting(points, method)
    if bbox_fit.center is None:
      return None
    return BEVBoundingBox3D(bbox_fit.center, bbox_fit.size, bbox_fit.yaw)
    # self.left, self.right, self.up = bbox_fit.left, bbox_fit.right, bbox_fit.up
  
  def to_xyzwhlr(self):
    """Returns box in the common 7-sized vector representation: (x, y, z, w,
    l, h, a), where (x, y, z) is the bottom center of the box, (w, l, h) is
    the width, length and height of the box a is the yaw angle.

    Returns:
      box: (7,)
    """
    bbox = np.zeros((7,))
    bbox[0:3] = self.center - [0, 0, self.size[2] / 2]
    bbox[3:6] = np.array(self.size)[[1, 0, 2]]
    bbox[6] = self.yaw #not transforming as custom convention
    return bbox

  def to_modal_xyzwhlr_vel(self):
    """Returns box in the common 10-sized vector representation: (x, y, z, w,
    l, h, a, vx, vy), where (x, y, z) is the bottom center of the box, (w, l, h) is
    the width, length and height of the box a is the yaw angle.

    Returns:
      box: (10,)
    """
    bbox = np.zeros((10,))
    bbox[0:3] = self.center
    bbox[3:6] = self.size
    bbox[6] = self.yaw #not transforming as custom convention
    bbox[7:10] = self.velocity
    return bbox

  def get_inside_points(self, points):
    indices = np.arange(len(points)).astype(np.int)

    self.left = np.array([np.cos(self.yaw), -np.sin(self.yaw), 0])
    self.front = np.array([np.sin(self.yaw), np.cos(self.yaw), 0])
    self.up = np.array([0., 0., 1.])

    axis = points[:, :3] - self.center
    dotx = np.abs(np.dot(axis, self.front)) <= 0.5*self.size[1]
    doty = np.abs(np.dot(axis, self.left)) <= 0.5*self.size[0]
    dotz = np.abs(np.dot(axis, self.up)) <= 0.5*self.size[2]

    dots = dotx & doty & dotz
    return indices[dots]

  @staticmethod
  def create_lines(boxes, method=None, lut=None):
    """Creates and returns an open3d.geometry.LineSet that can be used to
    render the boxes.

    Args:
      boxes: the list of bounding boxes
      lut: a ml3d.vis.LabelLUT that is used to look up the color based on
        the label_class argument of the BoundingBox3D constructor. If
        not provided, a color of 50% grey will be used. (optional)
    """
    if method == '2d':
      return BEVBoundingBox3D.create_lines_2d(boxes, lut=None)
    elif method == 'open3d':
      return BoundingBox3D.create_lines(boxes, lut=lut)

    nverts = 8#14
    nlines = 12#17
    points = np.zeros((nverts * len(boxes), 3), dtype="float32")
    indices = np.zeros((nlines * len(boxes), 2), dtype="int32")
    colors = np.zeros((nlines * len(boxes), 3), dtype="float32")

    for i in range(0, len(boxes)):
      box = boxes[i]
      pidx = nverts * i
      box_points = np.asarray(box.get_box_points())
      # x = 0.5 * box.size[0] * box.left
      # y = 0.5 * box.size[1] * box.up
      # z = 0.5 * box.size[2] * box.front
      # arrow_tip = box.center + z + box.arrow_length * box.front
      # arrow_mid = box.center + z + 0.60 * box.arrow_length * box.front
      # head_length = 0.3 * box.arrow_length
      # It seems to be substantially faster to assign directly for the
      # points, as opposed to points[pidx:pidx+nverts] = np.stack((...))
      points[pidx] = box_points[4]
      points[pidx + 1] = box_points[5]
      points[pidx + 2] = box_points[2]
      points[pidx + 3] = box_points[7]
      points[pidx + 4] = box_points[6]
      points[pidx + 5] = box_points[3]
      points[pidx + 6] = box_points[0]
      points[pidx + 7] = box_points[1]
      # points[pidx + 8] = box.center + z
      # points[pidx + 9] = arrow_tip
      # points[pidx + 10] = arrow_mid + head_length * box.up
      # points[pidx + 11] = arrow_mid - head_length * box.up
      # points[pidx + 12] = arrow_mid + head_length * box.left
      # points[pidx + 13] = arrow_mid - head_length * box.left

    # It is faster to break the indices and colors into their own loop.
    for i in range(0, len(boxes)):
      box = boxes[i]
      pidx = nverts * i
      idx = nlines * i
      indices[idx:idx +
          nlines] = ((pidx, pidx + 1), (pidx + 1, pidx + 2),
               (pidx + 2, pidx + 3), (pidx + 3, pidx),
               (pidx + 4, pidx + 5), (pidx + 5, pidx + 6),
               (pidx + 6, pidx + 7), (pidx + 7, pidx + 4),
               (pidx + 0, pidx + 4), (pidx + 1, pidx + 5),
               (pidx + 2, pidx + 6), (pidx + 3, pidx + 7))
              #  (pidx + 8, pidx + 9), (pidx + 9, pidx + 10),
              #  (pidx + 9, pidx + 11), (pidx + 9,
              #              pidx + 12), (pidx + 9,
              #                     pidx + 13))

      if lut is not None and box.label_class in lut.labels:
        label = lut.labels[box.label_class]
        c = (label.color[0], label.color[1], label.color[2])
      else:
        c = (0.5, 0.5, 0.5)

      colors[idx:idx +
         nlines] = c # copies c to each element in the range

    lines = o3d.geometry.LineSet()
    lines.points = o3d.utility.Vector3dVector(points)
    lines.lines = o3d.utility.Vector2iVector(indices)
    lines.colors = o3d.utility.Vector3dVector(colors)

    return lines

  def create_lines_2d(self, lut=None):
    nverts = 14 #8#14
    nlines = 17 #12#17

    points = np.zeros((nverts, 3), dtype="float32")
    indices = np.zeros((nlines, 2), dtype="int32")
    colors = np.zeros((nlines, 3), dtype="float32")

    pidx = 0
    x = 0.5 * self.size[0] * self.left
    y = 0.5 * self.size[1] * self.front
    z = 0.#0.5 * self.size[2] * self.front

    # self.center[2] = 0.
    self.arrow_length = 1.0
    head_length = 0.3 * self.arrow_length
    arrow_tip = self.center + z + self.arrow_length * self.front
    arrow_mid = self.center + z + 0.60 * self.arrow_length * self.front

    points[pidx] = self.center + x + y + z
    points[pidx + 1] = self.center - x + y + z
    points[pidx + 2] = self.center - x + y - z
    points[pidx + 3] = self.center + x + y - z
    points[pidx + 4] = self.center + x - y + z
    points[pidx + 5] = self.center - x - y + z
    points[pidx + 6] = self.center - x - y - z
    points[pidx + 7] = self.center + x - y - z
    points[pidx + 8] = self.center + z
    points[pidx + 9] = arrow_tip
    points[pidx + 10] = arrow_mid + head_length * self.up
    points[pidx + 11] = arrow_mid - head_length * self.up
    points[pidx + 12] = arrow_mid + head_length * self.left
    points[pidx + 13] = arrow_mid - head_length * self.left

    # It is faster to break the indices and colors into their own loop.
    pidx = 0
    idx = 0
    indices[idx:idx +
        nlines] = ((pidx, pidx + 1), (pidx + 1, pidx + 2),
              (pidx + 2, pidx + 3), (pidx + 3, pidx),
              (pidx + 4, pidx + 5), (pidx + 5, pidx + 6),
              (pidx + 6, pidx + 7), (pidx + 7, pidx + 4),
              (pidx + 0, pidx + 4), (pidx + 1, pidx + 5),
              (pidx + 2, pidx + 6), (pidx + 3, pidx + 7),
              (pidx + 8, pidx + 9), (pidx + 9, pidx + 10),
              (pidx + 9, pidx + 11), (pidx + 9,
                           pidx + 12), (pidx + 9,
                                  pidx + 13))

    if lut is not None and self.label_class in lut.labels:
      label = lut.labels[self.label_class]
      c = (label.color[0], label.color[1], label.color[2])
    else:
      c = (0.5, 0.5, 0.5)

    colors[idx:idx +
        nlines] = c # copies c to each element in the range

    lines = o3d.geometry.LineSet()
    lines.points = o3d.utility.Vector3dVector(points)
    lines.lines = o3d.utility.Vector2iVector(indices)
    lines.colors = o3d.utility.Vector3dVector(colors)

    return lines


class BoundingBox3D:
  """Class that defines an axially-oriented bounding box."""

  next_id = 1

  def __init__(self,
        center,
        front,
        up,
        left,
        size,
        label_class,
        confidence,
        meta=None,
        show_class=False,
        show_confidence=False,
        show_meta=None,
        identifier=None,
        arrow_length=1.0):
    """Creates a bounding box.

    Front, up, left define the axis of the box and must be normalized and
    mutually orthogonal.

    Args:
      center: (x, y, z) that defines the center of the box
      front: normalized (i, j, k) that defines the front direction of the box
      up: normalized (i, j, k) that defines the up direction of the box
      left: normalized (i, j, k) that defines the left direction of the box
      size: (width, height, depth) that defines the size of the box, as
        measured from edge to edge
      label_class: integer specifying the classification label. If an LUT is
        specified in create_lines() this will be used to determine the color
        of the box.
      confidence: confidence level of the box
      meta: a user-defined string (optional)
      show_class: displays the class label in text near the box (optional)
      show_confidence: displays the confidence value in text near the box
        (optional)
      show_meta: displays the meta string in text near the box (optional)
      identifier: a unique integer that defines the id for the box (optional,
        will be generated if not provided)
      arrow_length: the length of the arrow in the front_direct. Set to zero
        to disable the arrow (optional)
    """
    assert (len(center) == 3)
    assert (len(front) == 3)
    assert (len(up) == 3)
    assert (len(left) == 3)
    assert (len(size) == 3)

    self.center = np.array(center, dtype="float32")
    self.front = np.array(front, dtype="float32")
    self.up = np.array(up, dtype="float32")
    self.left = np.array(left, dtype="float32")
    self.size = size
    self.label_class = label_class
    self.confidence = confidence
    self.meta = meta
    self.show_class = show_class
    self.show_confidence = show_confidence
    self.show_meta = show_meta
    if identifier is not None:
      self.identifier = identifier
    else:
      self.identifier = "box:" + str(BoundingBox3D.next_id)
      BoundingBox3D.next_id += 1
    self.arrow_length = arrow_length

  
  def __repr__(self):
    s = str(self.identifier) + " (class=" + str(
      self.label_class) + ", conf=" + str(self.confidence)
    if self.meta is not None:
      s = s + ", meta=" + str(self.meta)
    s = s + ")"
    return s

  @staticmethod
  def create_lines(boxes, lut=None):
    """Creates and returns an open3d.geometry.LineSet that can be used to
    render the boxes.

    Args:
      boxes: the list of bounding boxes
      lut: a ml3d.vis.LabelLUT that is used to look up the color based on
        the label_class argument of the BoundingBox3D constructor. If
        not provided, a color of 50% grey will be used. (optional)
    """
    nverts = 14
    nlines = 17
    points = np.zeros((nverts * len(boxes), 3), dtype="float32")
    indices = np.zeros((nlines * len(boxes), 2), dtype="int32")
    colors = np.zeros((nlines * len(boxes), 3), dtype="float32")

    for i in range(0, len(boxes)):
      box = boxes[i]
      pidx = nverts * i
      x = 0.5 * box.size[0] * box.left
      y = 0.5 * box.size[1] * box.front
      z = 0.5 * box.size[2] * box.up
      arrow_tip = box.center + z + box.arrow_length * box.front
      arrow_mid = box.center + z + 0.60 * box.arrow_length * box.front
      head_length = 0.3 * box.arrow_length
      # It seems to be substantially faster to assign directly for the
      # points, as opposed to points[pidx:pidx+nverts] = np.stack((...))
      points[pidx] = box.center + x + y + z
      points[pidx + 1] = box.center - x + y + z
      points[pidx + 2] = box.center - x + y - z
      points[pidx + 3] = box.center + x + y - z
      points[pidx + 4] = box.center + x - y + z
      points[pidx + 5] = box.center - x - y + z
      points[pidx + 6] = box.center - x - y - z
      points[pidx + 7] = box.center + x - y - z
      points[pidx + 8] = box.center + z
      points[pidx + 9] = arrow_tip
      points[pidx + 10] = arrow_mid + head_length * box.up
      points[pidx + 11] = arrow_mid - head_length * box.up
      points[pidx + 12] = arrow_mid + head_length * box.left
      points[pidx + 13] = arrow_mid - head_length * box.left

    # It is faster to break the indices and colors into their own loop.
    for i in range(0, len(boxes)):
      box = boxes[i]
      pidx = nverts * i
      idx = nlines * i
      indices[idx:idx +
          nlines] = ((pidx, pidx + 1), (pidx + 1, pidx + 2),
               (pidx + 2, pidx + 3), (pidx + 3, pidx),
               (pidx + 4, pidx + 5), (pidx + 5, pidx + 6),
               (pidx + 6, pidx + 7), (pidx + 7, pidx + 4),
               (pidx + 0, pidx + 4), (pidx + 1, pidx + 5),
               (pidx + 2, pidx + 6), (pidx + 3, pidx + 7),
               (pidx + 8, pidx + 9), (pidx + 9, pidx + 10),
               (pidx + 9, pidx + 11), (pidx + 9,
                           pidx + 12), (pidx + 9,
                                  pidx + 13))

      if lut is not None and box.label_class in lut.labels:
        label = lut.labels[box.label_class]
        c = (label.color[0], label.color[1], label.color[2])
      else:
        c = (0.5, 0.5, 0.5)

      colors[idx:idx +
         nlines] = c # copies c to each element in the range

    lines = o3d.geometry.LineSet()
    lines.points = o3d.utility.Vector3dVector(points)
    lines.lines = o3d.utility.Vector2iVector(indices)
    lines.colors = o3d.utility.Vector3dVector(colors)

    return lines

def visualize_prediction(points, point_labels, dataset, result, mode=None, ignore_stuff=False, method=None):
  import numpy as np
  viewer = o3d.visualization.Visualizer()
  viewer.create_window()

  opt = viewer.get_render_option()
  opt.show_coordinate_frame = True

  pcd = o3d.geometry.PointCloud()

  if ignore_stuff:
    THINGS_MAPPED = [1, 2, 3, 4, 5, 6, 7, 8]
    things_mask = [p_i for p_i, p_v in enumerate(point_labels) if p_v in THINGS_MAPPED]
    points = points[things_mask]
    point_labels = point_labels[things_mask]

  pcd.points = o3d.utility.Vector3dVector(points)
  if mode != 'kitti':
    pcd.colors = o3d.utility.Vector3dVector(dataset.sem_color_lut[dataset.remap_lut[point_labels]])

  lines = [[0, 1], [1, 2], [2, 3], [3, 0],
        [0, 4], [1, 5], [2, 6], [3, 7],
        [4, 5], [5, 6], [6, 7], [7, 4], 
        [5, 2], [1, 6]]

  # Use the same color for all lines
  colors = [[1, 0, 0] for _ in range(len(lines))]

  viewer.add_geometry(result[0].create_lines(result, method=method))

  viewer.add_geometry(pcd)
  viewer.run()

def visualize_prediction_2d(points, point_labels, dataset, result, mode=None, ignore_stuff=False):
  import numpy as np
  viewer = o3d.visualization.Visualizer()
  viewer.create_window()

  opt = viewer.get_render_option()
  opt.show_coordinate_frame = True

  pcd = o3d.geometry.PointCloud()
  # points[:, 2] = 0.

  if ignore_stuff:
    THINGS_MAPPED = [1, 2, 3, 4, 5, 6, 7, 8]
    things_mask = [p_i for p_i, p_v in enumerate(point_labels) if p_v in THINGS_MAPPED]
    points = points[things_mask]
    point_labels = point_labels[things_mask]

  pcd.points = o3d.utility.Vector3dVector(points)
  if mode != 'kitti':
    pcd.colors = o3d.utility.Vector3dVector(dataset.sem_color_lut[dataset.remap_lut[point_labels]])

  [viewer.add_geometry(r.create_lines_2d()) for r in result]

  viewer.add_geometry(pcd)
  viewer.run()


class BEVBox3D(BoundingBox3D):
    """Class that defines a special bounding box for object detection, with only
    one rotation axis (yaw).

                            up z    x front (yaw=0.5*pi)
                                ^   ^
                                |  /
                                | /
        (yaw=pi) left y <------ 0

    The relative coordinate of bottom center in a BEV box is (0.5, 0.5, 0),
    and the yaw is around the z axis, thus the rotation axis=2.
    The yaw is 0 at the negative direction of y axis, and increases from
    the negative direction of y to the positive direction of x.
    """

    def __init__(self,
                 center,
                 size,
                 yaw,
                 label_class,
                 confidence,
                 world_cam=None,
                 cam_img=None,
                 **kwargs):
        """Creates a bounding box.

        Args:
            center: (x, y, z) that defines the center of the box
            size: (width, height, depth) that defines the size of the box, as
                measured from edge to edge
            yaw: yaw angle of box
            label_class: integer specifying the classification label. If an LUT is
                specified in create_lines() this will be used to determine the color
                of the box.
            confidence: confidence level of the box
            world_cam: world to camera transformation
            cam_img: camera to image transformation
        """
        self.yaw = yaw
        self.world_cam = world_cam
        self.cam_img = cam_img

        # x-axis
        left = [np.cos(self.yaw), -np.sin(self.yaw), 0]
        # y-axis
        front = [np.sin(self.yaw), np.cos(self.yaw), 0]
        # z-axis
        up = [0, 0, 1]

        super().__init__(center, front, up, left, size, label_class, confidence,
                         **kwargs)

        self.points_inside_box = np.array([])
        self.level = self.get_difficulty()
        self.dis_to_cam = np.linalg.norm(self.to_camera()[:3])

    def to_kitti_format(self, score=1.0):
        """This method transforms the class to KITTI format."""
        box2d = self.to_img()
        box2d[2:] += box2d[:2]  # Add w, h.
        truncation = -1
        occlusion = -1
        box = self.to_camera()
        center = box[:3]
        size = box[3:6]
        ry = box[6]

        x, z = center[0], center[2]
        beta = np.arctan2(z, x)
        alpha = -np.sign(beta) * np.pi / 2 + beta + ry

        kitti_str = '%s %.2f %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f' \
                    % (self.label_class, truncation, occlusion, alpha, box2d[0], box2d[1],
                       box2d[2], box2d[3], size[0], size[1], size[2], center[0], center[1], center[2],
                       ry, score)
        return kitti_str

    def generate_corners3d(self):
        """Generate corners3d representation for this object.

        Returns:
            corners_3d: (8, 3) corners of box3d in camera coordinates.
        """
        w, h, l = self.size
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

        R = np.array([[np.cos(self.yaw), 0,
                       np.sin(self.yaw)], [0, 1, 0],
                      [-np.sin(self.yaw), 0,
                       np.cos(self.yaw)]])
        corners3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)
        corners3d = np.dot(R, corners3d).T
        corners3d = corners3d + self.to_camera()[:3]
        return corners3d

    def to_xyzwhlr(self):
        """Returns box in the common 7-sized vector representation: (x, y, z, w,
        l, h, a), where (x, y, z) is the bottom center of the box, (w, l, h) is
        the width, length and height of the box a is the yaw angle.

        Returns:
            box: (7,)
        """
        bbox = np.zeros((7,))
        bbox[0:3] = self.center - [0, 0, self.size[1] / 2]
        bbox[3:6] = np.array(self.size)[[0, 2, 1]]
        bbox[6] = self.yaw
        return bbox

    def to_camera(self):
        """Transforms box into camera space.

                     up x    y front
                        ^   ^
                        |  /
                        | /
         left z <------ 0

        Returns box in the common 7-sized vector representation:
        (x, y, z, l, h, w, a), where
        (x, y, z) is the bottom center of the box,
        (l, h, w) is the length, height, width of the box
        a is the yaw angle

        Returns:
            transformed box: (7,)
        """
        if self.world_cam is None:
            return self.to_xyzwhlr()[[1, 2, 0, 4, 5, 3, 6]]

        bbox = np.zeros((7,))
        # In camera space, we define center as center of the bottom face of bounding box.
        bbox[0:3] = self.center - [0, 0, self.size[1] / 2]
        # Transform center to camera frame of reference.
        bbox[0:3] = (np.array([*bbox[0:3], 1.0]) @ self.world_cam)[:3]
        bbox[3:6] = [self.size[1], self.size[0], self.size[2]]  # h, w, l
        bbox[6] = self.yaw
        return bbox

    def to_img(self):
        """Transforms box into 2d box.

        Returns:
            transformed box: (4,)
        """
        if self.cam_img is None:
            return None

        corners = self.generate_corners3d()
        corners = np.concatenate(
            [corners, np.ones((corners.shape[0], 1))], axis=-1)

        bbox_img = np.matmul(corners, self.cam_img)
        bbox_img = bbox_img[:, :2] / bbox_img[:, 3:]

        minxy = np.min(bbox_img, axis=0)
        maxxy = np.max(bbox_img, axis=0)

        size = maxxy - minxy
        center = minxy + size / 2

        return np.concatenate([center, size])

    def get_difficulty(self):
        """General method to compute difficulty, can be overloaded.

        Returns:
            Difficulty depending on projected height of box.
        """
        if self.cam_img is None:
            return 0

        heights = [40, 25]
        height = self.to_img()[3] + 1
        diff = -1
        for j in range(len(heights)):
            if height >= heights[j]:
                diff = j
                break
        return diff

    def to_dict(self):
        """Convert data for evaluation:"""
        return {
            'bbox': self.to_camera(),
            'label': self.label_class,
            'score': self.confidence,
            'difficulty': self.level
        }

    @staticmethod
    def to_dicts(bboxes):
        """Convert data for evaluation:

        Args:
            bboxes: List of BEVBox3D bboxes.
        """
        box_dicts = {
            'bbox': np.empty((len(bboxes), 7)),
            'label': np.empty((len(bboxes),), dtype='<U20'),
            'score': np.empty((len(bboxes),)),
            'difficulty': np.empty((len(bboxes),))
        }

        for i in range(len(bboxes)):
            box_dict = bboxes[i].to_dict()
            for k in box_dict:
                box_dicts[k][i] = box_dict[k]

        return box_dicts

    def get_inside_points(self, points):
        indices = np.arange(len(points)).astype(np.int)

        axis = points[:, :3] - self.center
        dotx = np.abs(np.dot(axis, self.front)) <= 0.5*self.size[2]
        doty = np.abs(np.dot(axis, self.left)) <= 0.5*self.size[0]
        dotz = np.abs(np.dot(axis, self.up)) <= 0.5*self.size[1]

        dots = dotx & doty & dotz
        return indices[dots]
