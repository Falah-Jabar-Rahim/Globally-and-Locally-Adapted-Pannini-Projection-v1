import concurrent
import concurrent.futures
import io
import math
import os.path
import cv2
import numpy as np
import scipy
from scipy.interpolate import interp2d
from Files import Back_Proj


def rotMat(self):
    phi = math.radians(self.ph)
    thi = math.radians(self.th)
    R = [[math.cos(phi), -math.sin(phi) * math.sin(thi), math.sin(phi) * math.cos(thi)],
         [0, math.cos(thi), math.sin(thi)],
         [-math.sin(phi), -math.cos(phi) * math.sin(thi), math.cos(phi) * math.cos(thi)]]
    R = np.array(R)
    return R


def VP_rendering(self, YUV_erp, erp_mask):  # VP rendering with Pannini
    Y_erp = YUV_erp['s'].Y
    row = range(0, Y_erp.shape[1])
    col = range(0, Y_erp.shape[0])
    self.f_Y = scipy.interpolate.RectBivariateSpline(col, row, Y_erp)  # for Y channel
    self.f_SegMap = scipy.interpolate.RectBivariateSpline(col, row, erp_mask)  # for segmenation map

    U_erp = YUV_erp['s'].U
    row = range(0, U_erp.shape[1])
    col = range(0, U_erp.shape[0])
    self.f_U = scipy.interpolate.RectBivariateSpline(col, row, U_erp)  # for U channel
    V_erp = YUV_erp['s'].V
    row = range(0, V_erp.shape[1])
    col = range(0, V_erp.shape[0])
    self.f_V = scipy.interpolate.RectBivariateSpline(col, row, V_erp)  # for V channel

    VP_Y, VP_UV = Back_Proj.Generate_file_name(self)

    if os.path.isfile(VP_Y + '.npy') and os.path.isfile(VP_UV + '.npy'):
        XYZ_Y = np.load(VP_Y + '.npy')
        self.X_Y = XYZ_Y[0]
        self.Y_Y = XYZ_Y[1]
        self.Z_Y = XYZ_Y[2]
        XYZ_UV = np.load(VP_UV + '.npy')
        self.X_UV = XYZ_UV[0]
        self.Y_UV = XYZ_UV[1]
        self.Z_UV = XYZ_UV[2]
    else:
        Back_Proj.VP_compute_XYZ(self)
        XYZ_Y = np.load(VP_Y + '.npy')
        self.X_Y = XYZ_Y[0]
        self.Y_Y = XYZ_Y[1]
        self.Z_Y = XYZ_Y[2]
        XYZ_UV = np.load(VP_UV + '.npy')
        self.X_UV = XYZ_UV[0]
        self.Y_UV = XYZ_UV[1]
        self.Z_UV = XYZ_UV[2]

    self.R = rotMat(self)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        f1 = executor.submit(render_Y, self, erp_mask)
        # print(f1.result())
        f2 = executor.submit(render_UV, self)
        # print(f2.result())
    # Open In-memory bytes streams
    out1 = f1.result()
    y = out1[0]
    VP_SegMap = out1[1]
    out2 = f2.result()
    u = out2[0]
    v = out2[1]
    f = io.BytesIO()
    # Write Y, U and V to the "streams".
    f.write(y.tobytes())
    f.write(u.tobytes())
    f.write(v.tobytes())
    f.seek(0)
    y = np.frombuffer(f.read(y.size), dtype=np.uint8).reshape(
        (y.shape[0], y.shape[1]))  # Read Y color channel and reshape to height x width numpy array
    u = np.frombuffer(f.read(y.size // 4), dtype=np.uint8).reshape(
        (y.shape[0] // 2, y.shape[1] // 2))  # Read U color channel and reshape to height x width numpy array
    v = np.frombuffer(f.read(y.size // 4), dtype=np.uint8).reshape(
        (y.shape[0] // 2, y.shape[1] // 2))  # Read V color channel and reshape to height x width numpy array

    # Resize u and v color channels to be the same size as y
    u = cv2.resize(u, (y.shape[1], y.shape[0]))
    v = cv2.resize(v, (y.shape[1], y.shape[0]))
    yvu = cv2.merge((y, v, u))  # Stack planes to 3D matrix (use Y,V,U ordering)
    bgr = cv2.cvtColor(yvu, cv2.COLOR_YCrCb2BGR)

    #cv2.imwrite('Input.bmp', bgr)
    #cv2.imwrite('Input_seg.bmp', VP_SegMap*255)
    return bgr, VP_SegMap


def render_Y(self, erp_mask):
    p1 = np.zeros((3, 1))
    VP_Y = np.zeros((self.H_vp, self.W_vp), dtype=np.uint8)
    VP_SegMap = np.zeros((self.H_vp, self.W_vp), dtype=np.uint8)
    for n in range(self.H_vp):
        for m in range(self.W_vp):
            p1[0] = self.X_Y[n][m]
            p1[1] = self.Y_Y[n][m]
            p1[2] = self.Z_Y[n][m]
            p = np.matmul(self.R, p1)
            phi = math.atan2(p[0], p[2])
            theta = math.atan2(p[1], (math.sqrt(p[0] ** 2 + p[2] ** 2)))
            m_erp = self.W_erp * (0.5 + phi / (2 * math.pi)) - 0.5
            n_erp = self.H_erp * (0.5 - (theta / math.pi)) - 0.5

            if m_erp < 0:
                m_erp = 0
            if n_erp < 0:
                n_erp = 0
            if m_erp >= self.W_erp:
                m_erp = self.W_erp-1
            if n_erp >= self.H_erp:
                n_erp = self.H_erp-1
            VP_Y[n][m] = self.f_Y(n_erp, m_erp)

            ## for Segmentation mask
            m_erp = math.floor(m_erp)
            n_erp = math.floor(n_erp)
            if m_erp < 0:
                m_erp = 0
            if n_erp < 0:
                n_erp = 0
            if m_erp >= self.W_erp:
                m_erp = self.W_erp-1
            if n_erp >= self.H_erp:
                n_erp = self.H_erp-1
            VP_SegMap[n][m] = erp_mask[n_erp, m_erp]

    VP_SegMap = np.where(VP_SegMap > 0, 1, 0).astype(np.uint8) ## binarize the mask
    return VP_Y, VP_SegMap


def render_UV(self):
    p1 = np.zeros((3, 1))
    VP_U = np.zeros((math.floor(self.H_vp / 2), math.floor(self.W_vp / 2)), dtype=np.uint8)
    VP_V = np.zeros((math.floor(self.H_vp / 2), math.floor(self.W_vp / 2)), dtype=np.uint8)
    cnt_list = []
    for n in range(self.H_vp):
        for m in range(self.W_vp):
            if m % 2 == 0 and n % 2 == 0:
                m_UV = m // 2
                n_UV = n // 2
                p1[0] = self.X_UV[n_UV][m_UV]
                p1[1] = self.Y_UV[n_UV][m_UV]
                p1[2] = self.Z_UV[n_UV][m_UV]
                p = np.matmul(self.R, p1)
                phi = math.atan2(p[0], p[2])
                theta = math.atan2(p[1], (math.sqrt(p[0] ** 2 + p[2] ** 2)))
                theta_check = math.atan2(p1[1], (math.sqrt(p1[0] ** 2 + p1[2] ** 2)))
                m_erp = (self.W_erp // 2) * (0.5 + phi / (2 * math.pi)) - 0.5
                n_erp = (self.H_erp // 2) * (0.5 - (theta / math.pi)) - 0.5

                if m_erp < 0:
                    m_erp = 0
                if n_erp < 0:
                    n_erp = 0
                if m_erp > self.W_erp:
                    m_erp = self.W_erp
                if n_erp > self.H_erp:
                    n_erp = self.H_erp

                VP_U[n_UV][m_UV] = self.f_U(n_erp, m_erp)
                VP_V[n_UV][m_UV] = self.f_V(n_erp, m_erp)

    return VP_U, VP_V
