import math
import numpy as np


def CutOut_border(self, hfov, vfov, VP, VP_rect, VP_seg, overlay_flow):

    HFoV = hfov-self.margin
    HFoV_rad = (HFoV * math.pi) / 180
    d = self.d_back
    asp = self.asp
    VFoV = 2 * math.atan((d + 1) * (math.sin(HFoV_rad / 2)) / ((d + math.cos(HFoV_rad / 2)) * asp))  # compute
    Vvs = 2 * math.tan(VFoV / 2)  # compute viewport vertical size
    Vhs = (2 * (d + 1) * (math.sin(HFoV_rad / 2)) / (d + math.cos(HFoV_rad / 2)))  # compute viewport horizontal size
    x_min = -Vhs/2
    x_max = Vhs/2
    y_min = -Vvs/2
    y_max = Vvs/2

    HFoV = hfov
    HFoV_rad = (HFoV * math.pi) / 180
    d = self.d_back
    asp = self.asp
    VFoV = 2 * math.atan((d + 1) * (math.sin(HFoV_rad / 2)) / ((d + math.cos(HFoV_rad / 2)) * asp))  # compute
    Vvs = 2 * math.tan(VFoV / 2)  # compute viewport vertical size
    Vhs = (2 * (d + 1) * (math.sin(HFoV_rad / 2)) / (d + math.cos(HFoV_rad / 2)))  # compute viewport horizontal size


    row = np.arange(start=0, stop=self.W_vp, step=1) ## horizontal line
    col = np.arange(start=0, stop=self.H_vp, step=1)  ## vertical line

    u = (row + 0.5) * (Vhs / self.W_vp)
    v = (col + 0.5) * (Vvs / self.H_vp)
    x = u - (Vhs / 2)
    y = -v + (Vvs / 2)
    Left = np.count_nonzero(x < x_min)
    Right = self.W_vp-np.count_nonzero(x > x_max)
    Top = np.count_nonzero(y < y_min)
    Bottom = self.H_vp-np.count_nonzero(y > y_max)
    VP_cut = VP[Top:Bottom, Left:Right]
    VP_seg_cut = VP_seg[Top:Bottom, Left:Right]
    VP_rect_cut = VP_rect[Top:Bottom, Left:Right]
    overlay_flow_cut = overlay_flow[Top:Bottom, Left:Right]

    return VP_cut, VP_seg_cut, VP_rect_cut, overlay_flow_cut
