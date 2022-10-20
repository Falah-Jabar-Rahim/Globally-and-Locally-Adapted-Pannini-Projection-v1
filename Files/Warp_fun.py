
import cv2
from cv2 import imshow, BORDER_TRANSPARENT, BORDER_CONSTANT

def Warp_im(self, name, VP_rect, x_mesh, y_mesh, VP_Vhs, VP_Vvs, w, h):
    x_mesh = cv2.resize(x_mesh, (w, h))
    y_mesh = cv2.resize(y_mesh, (w, h))
    u = x_mesh + (VP_Vhs / 2)
    v = (VP_Vvs / 2) - y_mesh
    m_vp = ((((u * w) / VP_Vhs) - 0.5))
    n_vp = ((((h * v) / VP_Vvs) - 0.5))
    out = cv2.remap(VP_rect, m_vp, n_vp, interpolation=cv2.INTER_LINEAR, borderValue=0, borderMode=BORDER_CONSTANT)
    return out
