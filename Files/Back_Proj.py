import math
import numpy
import numpy as np


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


def Generate_file_name(self):
    s1 = str(self.HFoV)
    s2 = str(self.W_vp)
    s3 = str(self.H_vp)
    s4 = str(self.d_back)
    s5 = str(self.W_erp)
    s6 = str(self.H_erp)
    s7 = str(self.vc_back)
    VP_Y = (
            'VP_Y_' + s1 + 'Fh_' + s2 + 'Wvp_' + s3 + 'Hvp_' + 'd' + s4 + '_vc' + s7 + '_' + s5 + 'Werp' + '_' + s6 + 'Herp')
    s2 = str(self.W_vp // 2)
    s3 = str(self.H_vp // 2)
    s5 = str(self.W_erp // 2)
    s6 = str(self.H_erp // 2)
    VP_UV = (
            'VP_UV_' + s1 + 'Fh_' + s2 + 'Wvp_' + s3 + 'Hvp_' + 'd' + s4 + '_vc' + s7 + '_' + s5 + 'Werp' + '_' + s6 + 'Herp')
    print(VP_Y)
    print(VP_UV)
    return VP_Y, VP_UV


def VP_compute_XYZ(self):
    VP_Vhs, VP_Vvs, _ = compute_VP_size(self)  # compute VP size for general paramters
    X_Y = np.zeros((self.H_vp, self.W_vp), dtype=numpy.float16)
    Y_Y = np.zeros((self.H_vp, self.W_vp), dtype=numpy.float16)
    Z_Y = np.zeros((self.H_vp, self.W_vp), dtype=numpy.float16)
    X_UV = np.zeros((self.H_vp // 2, self.W_vp // 2), dtype=numpy.float16)
    Y_UV = np.zeros((self.H_vp // 2, self.W_vp // 2), dtype=numpy.float16)
    Z_UV = np.zeros((self.H_vp // 2, self.W_vp // 2), dtype=numpy.float16)
    VP_Y, VP_UV = Generate_file_name(self)
    for n in range(self.H_vp):
        for m in range(self.W_vp - 1):
            u = (m + 0.5) * (VP_Vhs / self.W_vp)
            v = (n + 0.5) * (VP_Vvs / self.H_vp)
            x = u - (VP_Vhs / 2)
            y = -v + (VP_Vvs / 2)
            k = x ** 2 / (self.d_back + 1) ** 2
            cosPhi_ = ((-k * self.d_back) + math.sqrt(
                (k ** 2 * self.d_back ** 2) - ((k + 1) * (k * self.d_back ** 2 - 1)))) / (k + 1)
            S = (self.d_back + 1) / (self.d_back + cosPhi_)
            phi = math.atan2(x, (S * cosPhi_))
            theta = math.atan2(y, ((1 - self.vc_back) * S) + (self.vc_back * (1 / cosPhi_)))
            X_Y[n][m] = math.cos(theta) * math.sin(phi)
            Y_Y[n][m] = math.sin(theta)
            Z_Y[n][m] = math.cos(theta) * math.cos(phi)
            if m % 2 == 0 and n % 2 == 0:
                u = ((m // 2) + 0.5) * (VP_Vhs / (self.W_vp // 2))
                v = ((n // 2) + 0.5) * (VP_Vvs / (self.H_vp // 2))
                x = u - (VP_Vhs / 2)
                y = -v + (VP_Vvs / 2)
                k = x ** 2 / (self.d_back + 1) ** 2
                cosPhi_ = ((-k * self.d_back) + math.sqrt(
                    (k ** 2 * self.d_back ** 2) - ((k + 1) * (k * self.d_back ** 2 - 1)))) / (
                                  k + 1)
                S = (self.d_back + 1) / (self.d_back + cosPhi_)
                phi = math.atan2(x, (S * cosPhi_))
                theta = math.atan2(y, ((1 - self.vc_back) * S) + (self.vc_back * (1 / cosPhi_)))
                X_UV[n // 2][m // 2] = math.cos(theta) * math.sin(phi)
                Y_UV[n // 2][m // 2] = math.sin(theta)
                Z_UV[n // 2][m // 2] = math.cos(theta) * math.cos(phi)

    XYZ_Y = [X_Y, Y_Y, Z_Y]
    XYZ_UV = [X_UV, Y_UV, Z_UV]
    # save X, Y, Z for Y and UV channel separately
    numpy.save(VP_Y, XYZ_Y)
    numpy.save(VP_UV, XYZ_UV)
