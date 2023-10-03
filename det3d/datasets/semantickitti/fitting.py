from enum import Enum
from ml3d.tf.utils.objdet_helper import box3d_to_bev2d
import numpy as np
import math
import cv2
import open3d as o3d

def plot_rect(points, rect):
  factor = 100
  points_int = np.int0(points[:, 0:2]*factor)
  points_int[:, [0,1]] = points_int[:, [1,0]]
  
  points_min = np.min(points_int, axis=0) - 100
  points_int -= points_min

  im = np.zeros((800, 800, 3))
  for p_i, p_v in enumerate(points_int):
    im = cv2.circle(im, p_v, 5, (255, 0, 0), -1)
  try:
    box = cv2.boxPoints(rect) # cv2.boxPoints(rect) for OpenCV 3.x
  except:
    box = np.array(rect).T
    box[:, [0,1]] = box[:, [1,0]]
  box = np.int0(box*factor) - points_min
  # box = box // 100
  cv2.drawContours(im,[box],0,(0,0,255),2)
  cv2.imshow("Iage", im)
  cv2.waitKey(0)
  cv2.destroyAllWindows()


class BBoxFitting():
  def __init__(self, points, method=None):
    if method == 'minarea':
      self.minAreaCreate(points)
    elif method == 'minarea-hull':
      #compute points on hull, same result as above
      from scipy.spatial import ConvexHull
      qhull = ConvexHull(points[:, [0,1]])
      self.minAreaCreate(points[qhull.vertices])
    elif method == 'minarea-outlier':
      self.minAreaCreate(points)
    elif method == 'closeness':
      self.lfitting(points)

      #modify center_z, extent_z acc to pts before hull
      extent_z = np.max(points[:, 2]) - np.min(points[:, 2])
      center_z = 0.5*(np.max(points[:, 2]) + np.min(points[:, 2]))
      self.center[2] = center_z
      self.size[2] = extent_z
    
    elif method == 'hybrid-select':
      points_inline = points[:, :3]

      if len(points_inline) < 5:
        self.center = None
        return

      from scipy.spatial import ConvexHull
      qhull = ConvexHull(points_inline[:, [0,1]])
      points_hull = points[qhull.vertices]

      box3d_closeness = BBoxFitting(points_hull, method='closeness')
      box3d_area = BBoxFitting(points_inline, method='minarea-outlier')

      points = points_inline
      box3d_closeness_inside = len(box3d_closeness.get_inside_points(points))
      box3d_area_inside = len(box3d_area.get_inside_points(points))

      #modified close:
      select_yaw = box3d_closeness.yaw
      for i in range(4):
        box3d_closeness.yaw = (box3d_closeness.yaw+math.pi*0.5)%math.pi
        box3d_closeness_inside_alt = len(box3d_closeness.get_inside_points(points))
        if box3d_closeness_inside_alt > box3d_closeness_inside:
          select_yaw = box3d_closeness.yaw
          box3d_closeness_inside = box3d_closeness_inside_alt
      box3d_closeness.yaw = select_yaw

      if box3d_closeness_inside > box3d_area_inside:
        bbox_fit = box3d_closeness
      else:
        bbox_fit = box3d_area
      

      if bbox_fit.yaw < -0.5*math.pi:
        bbox_fit.yaw += math.pi
      elif bbox_fit.yaw > 0.5*math.pi:
        bbox_fit.yaw -= math.pi

      self.center, self.size, self.yaw = bbox_fit.center, bbox_fit.size, bbox_fit.yaw
    else:
      raise NotImplementedError
  
  def __repr__(self) -> str:
    return np.array2string(self.center) + ':' + \
        np.array2string(self.size) + ':' + \
        np.array2string(self.yaw)

  def lfitting(self, points):
    from .fitting import LShapeFitting
    lshapefitting = LShapeFitting()
    lshapefitting.criteria = lshapefitting.Criteria.CLOSENESS
    rect, costs, minp = lshapefitting.fitting(points[:, 0:2])
    rect.calc_rect_contour()

    corner_points = np.array((rect.rect_c_x[:4], rect.rect_c_y[:4]))
    extent_z = np.max(points[:, 2]) - np.min(points[:, 2])
    center_z = 0.5*(np.max(points[:, 2]) + np.min(points[:, 2]))

    self.yaw = minp[1]
    if self.yaw < 0:
      self.yaw = -math.pi*0.5 - self.yaw
    else:
      self.yaw = -self.yaw

    self.center = np.array([np.mean(rect.rect_c_x[:4]),
                np.mean(rect.rect_c_y[:4]),
                center_z]).astype(np.float64)
                
    #assuming rect are in order
    rect_x = np.array(rect.rect_c_x)
    rect_y = np.array(rect.rect_c_y)
    extent_x = np.max(np.abs(rect_x[:4] - rect_x[1:]))
    extent_y = np.max(np.abs(rect_y[:4] - rect_y[1:]))
    self.size = np.array([extent_x,
              extent_y,
              extent_z]).astype(np.float64)


    self.left = np.array([np.cos(self.yaw), -np.sin(self.yaw), 0])
    self.front = np.array([np.sin(self.yaw), np.cos(self.yaw), 0])
    self.up = np.array([0., 0., 1.])

  def minAreaCreate(self, points):
    # get only x-y points
    points_xy = (points[:, 0:2]).astype(np.float32)
    points_xy[:, [0, 1]] = points_xy[:, [1, 0]]

    rect = cv2.minAreaRect(points_xy)

    extent_z = np.max(points[:, 2]) - np.min(points[:, 2])
    center_z = 0.5*(np.max(points[:, 2]) + np.min(points[:, 2]))

    self.center = np.array([rect[0][1], rect[0][0], center_z]).astype(np.float64)
    self.size = np.array([rect[1][1], rect[1][0], extent_z]).astype(np.float64)

    self.yaw = rect[2]/180.*math.pi

    # in case rects x,y swiched. change axis
    orig_x_size = np.max(points[:, 0]) - np.min(points[:, 0])
    orig_y_size = np.max(points[:, 1]) - np.min(points[:, 1])
    if np.abs(self.size[0] - orig_y_size) < np.abs(self.size[0] - orig_x_size):
      self.size[[0, 1]] = self.size[[1,0]]
      self.yaw = math.pi*0.5 + self.yaw

    if self.yaw < 0:
      self.yaw = (self.yaw + math.pi) % math.pi
    else:
      self.yaw -= math.pi
      
    # x-axis
    self.left = np.array([np.cos(self.yaw), -np.sin(self.yaw), 0])
    # y-axis
    self.front = np.array([np.sin(self.yaw), np.cos(self.yaw), 0])
    self.up = np.array([0., 0., 1.])

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

class LShapeFitting():
  class Criteria(Enum):
    AREA = 1
    CLOSENESS = 2
    VARIANCE = 3
  def __init__(self):
    # Parameters
    self.criteria = self.Criteria.VARIANCE
    self.min_dist_of_closeness_crit = 0.1 # [m]
    self.dtheta_deg_for_search = 1.0 # [deg]
  def fitting(self, pc):
    rect, costs, minp = self._rectangle_search(pc)
    return rect, costs, minp
  def _rectangle_search(self, pc):
    X = np.array(pc)
    dtheta = np.deg2rad(self.dtheta_deg_for_search)
    minp = (-float('inf'), None)
    costs = []
    for theta in np.arange(-np.pi / 2.0, np.pi / 2.0 - dtheta, dtheta):
      e1 = np.array([np.cos(theta), np.sin(theta)])
      e2 = np.array([-np.sin(theta), np.cos(theta)])
      c1 = X @ e1.T
      c2 = X @ e2.T
      # Select criteria
      if self.criteria == self.Criteria.AREA:
        cost = self._calc_area_criterion(c1, c2)
      elif self.criteria == self.Criteria.CLOSENESS:
        cost = self._calc_closeness_criterion(c1, c2)
      elif self.criteria == self.Criteria.VARIANCE:
        cost = self._calc_variance_criterion(c1, c2)
      costs.append(cost)
      if minp[0] < cost:
        minp = (cost, theta)
    # calculate best rectangle
    sin_s = np.sin(minp[1])
    cos_s = np.cos(minp[1])
    c1_s = X @ np.array([cos_s, sin_s]).T
    c2_s = X @ np.array([-sin_s, cos_s]).T
    rect = RectangleData()
    rect.a[0] = cos_s
    rect.b[0] = sin_s
    rect.c[0] = min(c1_s)
    rect.a[1] = -sin_s
    rect.b[1] = cos_s
    rect.c[1] = min(c2_s)
    rect.a[2] = cos_s
    rect.b[2] = sin_s
    rect.c[2] = max(c1_s)
    rect.a[3] = -sin_s
    rect.b[3] = cos_s
    rect.c[3] = max(c2_s)
    return rect, costs, minp
  def _calc_area_criterion(self, c1, c2):
    c1_max = max(c1)
    c2_max = max(c2)
    c1_min = min(c1)
    c2_min = min(c2)
    alpha = -(c1_max - c1_min) * (c2_max - c2_min)
    return alpha
  def _calc_closeness_criterion(self, c1, c2):
    c1_max = max(c1)
    c2_max = max(c2)
    c1_min = min(c1)
    c2_min = min(c2)
    D1 = [min([np.linalg.norm(c1_max - ic1), np.linalg.norm(ic1 - c1_min)]) for ic1 in c1]
    D2 = [min([np.linalg.norm(c2_max - ic2), np.linalg.norm(ic2 - c2_min)]) for ic2 in c2]
    beta = 0
    for i, _ in enumerate(D1):
      d = max(min([D1[i], D2[i]]), self.min_dist_of_closeness_crit)
      beta += (1.0 / d)
    return beta
  def _calc_variance_criterion(self, c1, c2):
    c1_max = max(c1)
    c2_max = max(c2)
    c1_min = min(c1)
    c2_min = min(c2)
    D1 = [min([np.linalg.norm(c1_max - ic1),
         np.linalg.norm(ic1 - c1_min)]) for ic1 in c1]
    D2 = [min([np.linalg.norm(c2_max - ic2),
         np.linalg.norm(ic2 - c2_min)]) for ic2 in c2]
    E1, E2 = [], []
    for (d1, d2) in zip(D1, D2):
      if d1 < d2:
        E1.append(d1)
      else:
        E2.append(d2)
    V1 = 0.0
    if E1:
      V1 = - np.var(E1)
    V2 = 0.0
    if E2:
      V2 = - np.var(E2)
    gamma = V1 + V2
    return gamma

class RectangleData:

  def __init__(self):
    self.a = [None] * 4
    self.b = [None] * 4
    self.c = [None] * 4

    self.rect_c_x = [None] * 5
    self.rect_c_y = [None] * 5

  def plot(self):
    self.calc_rect_contour()
    plt.plot(self.rect_c_x, self.rect_c_y, "-r")

  def calc_rect_contour(self):

    self.rect_c_x[0], self.rect_c_y[0] = self.calc_cross_point(
      self.a[0:2], self.b[0:2], self.c[0:2])
    self.rect_c_x[1], self.rect_c_y[1] = self.calc_cross_point(
      self.a[1:3], self.b[1:3], self.c[1:3])
    self.rect_c_x[2], self.rect_c_y[2] = self.calc_cross_point(
      self.a[2:4], self.b[2:4], self.c[2:4])
    self.rect_c_x[3], self.rect_c_y[3] = self.calc_cross_point(
      [self.a[3], self.a[0]], [self.b[3], self.b[0]], [self.c[3], self.c[0]])
    self.rect_c_x[4], self.rect_c_y[4] = self.rect_c_x[0], self.rect_c_y[0]

  @staticmethod
  def calc_cross_point(a, b, c):
    x = (b[0] * -c[1] - b[1] * -c[0]) / (a[0] * b[1] - a[1] * b[0])
    y = (a[1] * -c[0] - a[0] * -c[1]) / (a[0] * b[1] - a[1] * b[0])
    return x, y