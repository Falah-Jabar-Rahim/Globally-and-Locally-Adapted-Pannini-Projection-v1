import os
import scipy.io as sio
from matplotlib import pyplot as plt


def read_imputs(self):
    ERP_name = os.path.join('InputData', self.split(".")[0] + '.mat')  # read 360 image name
    ERP_seg_name = os.path.join('InputData', self.split(".")[0] + '_segmap' + '.mat')  # read semantic segmented 360 image
    print(ERP_name)
    print(ERP_seg_name)

    ERP_im = sio.loadmat(ERP_name, squeeze_me=True, struct_as_record=False)  ## 360 image only Y chanel from [Y,U,V]
    Y_ch = ERP_im['s'].Y
    W = Y_ch.shape[1]  ## get 360 image width
    H = Y_ch.shape[0]  ## get 360 image height
    ERP_im_seg = sio.loadmat(ERP_seg_name, squeeze_me=True, struct_as_record=False)
    Mask = ERP_im_seg['Mask']
    # plt.axis("off")
    # plt.imshow(Mask)
    # plt.show()
    # # plt.imshow(cv2.cvtColor(im_ERP, cv2.COLOR_BGR2RGB))
    # plt.show()
    # plt.close()

    return ERP_name, ERP_seg_name, ERP_im, Mask, H, W
