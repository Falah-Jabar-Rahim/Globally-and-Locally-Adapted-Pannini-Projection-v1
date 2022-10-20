
import numpy as np


def Vp_border_mask(self, hfov, vfov):

    mask = np.zeros((self.H_vp, self.W_vp), dtype=np.uint8)
    px_deg = self.W_vp // hfov  ## pixel per degree in horizontal direction
    px_margin = int(self.margin * px_deg)
    Left = px_margin
    Right = self.W_vp - px_margin

    px_deg = self.H_vp // vfov  ## pixel per degree in vertical direction
    px_margin = int(self.margin * px_deg)
    Top = px_margin
    Bottom = self.H_vp - px_margin
    mask[Top:Bottom, Left:Right] = 1

    return mask
