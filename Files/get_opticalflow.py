
import numpy as np
from Optimization.src.visualization import get_overlay_flow


def optical_flow(self, x_out, y_out, x_inp, y_inp, VP_rect, VP_Vhs, VP_Vvs):

    u_out = x_out + (VP_Vhs / 2)
    v_out = (VP_Vvs / 2) - y_out
    m_out = (((u_out * self.W_vp) / VP_Vhs) - 0.5)
    n_out = (((self.H_vp * v_out) / VP_Vvs) - 0.5)
    u_inp = x_inp + (VP_Vhs / 2)
    v_inp = (VP_Vvs / 2) - y_inp
    m_inp = (((u_inp * self.W_vp) / VP_Vhs) - 0.5)
    n_inp = (((self.H_vp * v_inp) / VP_Vvs) - 0.5)
    flow_x = m_inp - m_out
    flow_y = n_inp - n_out


    overlay_flow = get_overlay_flow(VP_rect[:, :, ::-1], flow_x, flow_y, ratio=0.7)
    overlay_flow = (255 * overlay_flow[:, :, ::-1]).astype(np.uint8)
    return overlay_flow
