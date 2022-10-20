import math
import cv2
import numpy
import numpy as np
import scipy
import scipy
import scipy.io as sio
from matplotlib.pyplot import subplots_adjust
from matplotlib import pyplot as plt
from scipy import interpolate

from Files import Plot_mesh
from Files import Warp_fun


def com_mesh(self, x_mesh_back, y_mesh_back, x_mesh_obj, y_mesh_obj, w):
    H = x_mesh_back.shape[0]
    W = x_mesh_back.shape[1]
    w = cv2.resize(w, (W, H))
    x_vj = (x_mesh_obj - x_mesh_back) * w
    y_vj = (y_mesh_obj - y_mesh_back) * w
    x_vi = np.zeros((H, W), dtype=np.float32)
    y_vi = np.zeros((H, W), dtype=np.float32)
    x_com_NGu = np.zeros((H, W), dtype=np.float32)
    y_com_NGu = np.zeros((H, W), dtype=np.float32)
    # fig, axs = plt.subplots(3)
    # subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
    h = 2.37

    ws = 4

    for n in range(0, H, 1):
        for m in range(0, W, 1):

            # Mix mesh without gussain
            if w[n, m] == 1:
                x_com_NGu[n, m] = x_mesh_obj[n, m]
                y_com_NGu[n, m] = y_mesh_obj[n, m]
            else:
                x_com_NGu[n, m] = x_mesh_back[n, m]
                y_com_NGu[n, m] = y_mesh_back[n, m]

            n_t = n - (ws // 2)
            n_b = n + (ws // 2)
            m_l = m - (ws // 2)
            m_r = m + (ws // 2)
            if n_t < 0:
                n_t = 0
            if n_b > H - 1:
                n_t = H - 1
            if m_l < 0:
                m_l = 0
            if m_l > W - 1:
                m_l = W - 1

            # for xcoordinates
            ne = x_mesh_back[n_t:n_b, m_l:m_r]
            p_x = x_mesh_back[n, m]
            k_h = np.exp(-(((ne - p_x) ** 2) / (2 * h ** 2)))
            term_1 = (w[n_t:n_b, m_l:m_r] * k_h * x_vj[n_t:n_b, m_l:m_r])
            term_2 = w[n_t:n_b, m_l:m_r] * k_h
            new_cor = np.sum(term_1) / np.sum(term_2)
            if math.isnan(new_cor):
                x_vi[n, m] = 0
            else:
                x_vi[n, m] = np.sum(term_1) / np.sum(term_2)
            # for y coordinates
            ne = y_mesh_back[n_t:n_b, m_l:m_r]
            p_y = y_mesh_back[n, m]
            k_h = np.exp(-(((ne - p_y) ** 2) / (2 * h ** 2)))
            term_1 = (w[n_t:n_b, m_l:m_r] * k_h * y_vj[n_t:n_b, m_l:m_r])
            term_2 = w[n_t:n_b, m_l:m_r] * k_h
            new_cor = np.sum(term_1) / np.sum(term_2)

            if math.isnan(new_cor):
                y_vi[n, m] = 0
            else:
                y_vi[n, m] = np.sum(term_1) / np.sum(term_2)

    x_com = x_mesh_back + x_vi
    y_com = y_mesh_back + y_vi

    mesh_name = 'd =0.5, vc = 0.6'
    Plot_mesh.mesh_plt(x_mesh_back, y_mesh_back, mesh_name)
    mesh_name = 'd =0.5, vc = 0'
    Plot_mesh.mesh_plt(x_mesh_obj, y_mesh_obj, mesh_name)
    mesh_name = 'Mixed mesh'
    Plot_mesh.mesh_plt(x_com, y_com, mesh_name)
    mesh_name = 'Mixed mesh without gussain'
    Plot_mesh.mesh_plt(x_com_NGu, y_com_NGu, mesh_name)

    # axs[0].plot(x_mesh_back, y_mesh_back, color='k',  # All points are set to black
    #                     marker ='.')  # The shape of the dot is a dot
    # axs[1].plot(x_mesh_obj, y_mesh_obj, color='k',  # All points are set to black
    #                             marker ='.')  # The shape of the dot is a dot
    # axs[2].plot(x_com, y_com, color='k',  # All points are set to black
    #         marker ='.')  # The shape of the dot is a dot
    #
    # axs[0].set(xlabel='x', ylabel='y')
    # axs[0].set_title('d=0.5, vc=0.6')
    # axs[0].set_xlim([-1.5, 1.5])
    # axs[1].set(xlabel='x', ylabel='y')
    # axs[1].set_title('d=0.5, vc=0')
    # axs[1].set_xlim([-1.5, 1.5])
    # axs[2].set(xlabel='x', ylabel='y')
    # axs[2].set_title('Mixed mesh')
    # axs[2].set_xlim([-1.5, 1.5])
    # plt.show()

    return x_com, y_com, x_com_NGu, y_com_NGu
