import concurrent
import concurrent.futures
import io
import math
import os.path
import cv2
import numpy as np
import scipy
import scipy.io as sio
from matplotlib import pyplot as plt
from scipy.interpolate import interp2d

from Files import Plot_mesh
from Files import Warp_fun
from Files import xy2mn
from Files import Combine_mesh
from Files import Plot_im


def rotMat(self):
    phi = math.radians(self.ph)
    thi = math.radians(self.th)
    R = [[math.cos(phi), -math.sin(phi) * math.sin(thi), math.sin(phi) * math.cos(thi)],
         [0, math.cos(thi), math.sin(thi)],
         [-math.sin(phi), -math.cos(phi) * math.sin(thi), math.cos(phi) * math.cos(thi)]]
    R = np.array(R)
    return R


def compute_VP_size(self):
    HFoV = self.HFoV
    HFoV_rad = (HFoV * math.pi) / 180
    d = self.d_back
    asp = self.asp
    VFoV = 2 * math.atan((d + 1) * (math.sin(HFoV_rad / 2)) / ((d + math.cos(HFoV_rad / 2)) * asp))  # compute
    # vertical FoV
    Vvs = 2 * math.tan(VFoV / 2)  # compute viewport vertical size
    Vhs = (2 * (d + 1) * (math.sin(HFoV_rad / 2)) / (d + math.cos(HFoV_rad / 2)))  # compute viewport horizontal size
    return Vhs, Vvs, VFoV


def VP_rendering_BG(self, erp, erp_mask, VP_rect, VP_rect_seg):

    VP_Vhs, VP_Vvs, VFoV = compute_VP_size(self)
    p1 = np.zeros((3, 1))
    R = rotMat(self)
    VP = np.zeros((self.H_vp, self.W_vp, 3), dtype=np.uint8)

    Wm = self.W_vp//self.mesh_ds_ratio
    Hm = self.H_vp//self.mesh_ds_ratio

    x_mesh_back = np.zeros((Hm, Wm), dtype=np.float32)
    y_mesh_back = np.zeros((Hm, Wm), dtype=np.float32)
    x_mesh_obj = np.zeros((Hm, Wm), dtype=np.float32)
    y_mesh_obj = np.zeros((Hm, Wm), dtype=np.float32)

    ## create regular grid mesh
    m_mesh_back = (np.arange(0, Wm, 1)).astype(np.float32)
    n_mesh_back = (np.arange(0, Hm, 1)).astype(np.float32)
    m_mesh_back = m_mesh_back * self.mesh_ds_ratio
    n_mesh_back = n_mesh_back * self.mesh_ds_ratio
    m_mesh_back, n_mesh_back = np.meshgrid(m_mesh_back, n_mesh_back)
    ## Plot_mesh.mesh_plt(m_mesh_back, n_mesh_back)  ## plot mesh

    VP = np.zeros((Hm, Wm, 3), dtype=np.uint8)

    for i in range(0, Hm, 1):
        for j in range(0, Wm, 1):

            m = m_mesh_back[i, j]
            n = n_mesh_back[i, j]
            u = (m + 0.5) * (VP_Vhs / self.W_vp)
            v = (n + 0.5) * (VP_Vvs / self.H_vp)
            x = u - (VP_Vhs / 2)
            y = -v + (VP_Vvs / 2)
            ## rectiliner mesh
            x_mesh_back[i, j] = x
            y_mesh_back[i, j] = y
            ## Backward projection
            k = x ** 2 / (self.d_obj + 1) ** 2
            cosPhi_ = ((-k * self.d_obj) + math.sqrt(
                (k ** 2 * self.d_obj ** 2) - ((k + 1) * (k * self.d_obj ** 2 - 1)))) / (k + 1)
            S = (self.d_obj + 1) / (self.d_obj + cosPhi_)
            phi = math.atan2(x, (S * cosPhi_))
            theta = math.atan2(y, ((1 - self.vc_obj) * S) + (self.vc_obj * (1 / cosPhi_)))


            ## Forward projection
            S = (self.d_back + 1) / (self.d_back + math.cos(phi))
            x_mesh_obj[i, j] = S * math.sin(phi)
            y_mesh_obj[i, j] = ((1 - self.vc_back) * S * math.tan(theta)) + (
                    self.vc_back * (math.tan(theta) / math.cos(phi)))


    x_mesh_back = np.round(x_mesh_back, 3)
    y_mesh_back = np.round(y_mesh_back, 3)

    ## Plot meshe -----------------------
    # mesh_name = 'd =0.5, vc = 0.5'
    # Plot_mesh.mesh_plt(x_mesh_back, y_mesh_back, mesh_name)
    # mesh_name = 'd =0.5, vc = 0'
    # Plot_mesh.mesh_plt(x_mesh_obj, y_mesh_obj, mesh_name)
    # # ## ------------------------ do the warping --------------------
    # out_name = 'Rectilinear.bmp'
    # Warp_fun.Warp_im(self, out_name, VP_rect, x_mesh_back, y_mesh_back, VP_Vhs, VP_Vvs)
    #out_name = 'Sterographic.bmp'
    #out = Warp_fun.Warp_im(self, out_name, VP_rect, x_mesh_obj, y_mesh_obj, VP_Vhs, VP_Vvs, self.W_vp, self.H_vp)
    #cv2.imwrite(out_name, out)
    # x_com, y_com, x_com_NGu, y_com_NGu = com_mesh(self, x_mesh_back, y_mesh_back,  x_mesh_obj, y_mesh_obj, VP_rect_seg)
    # out_name = 'Combined.bmp'
    # Warp_fun.Warp_im(self, out_name, VP_rect, x_com, y_com, VP_Vhs, VP_Vvs)
    #
    # out_name = 'Combined_nogussain.bmp'
    # Warp_fun.Warp_im(self, out_name, VP_rect, x_com_NGu, y_com_NGu, VP_Vhs, VP_Vvs)


    ## x_mesh_back =x_com
    ## y_mesh_back = y_com
    return VP_rect, x_mesh_back, y_mesh_back, x_mesh_obj, y_mesh_obj, VP_Vhs, VP_Vvs, self.HFoV, VFoV*180/math.pi