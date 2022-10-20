import cv2
import numpy as np
from cv2 import countNonZero



def get_image_by_file(self, classes=None):

    Border_mask = self.Border_mask
    x_meshsterog = self.x_meshsterog
    y_meshsterog = self.y_meshsterog
    x_mesh = self.x_mesh
    y_mesh = self.y_mesh
    H, W = x_meshsterog.shape

    Tp = H*W
    perc = (Tp*self.obj_size)/100

    VP_segmap_sterog = self.VP_segmap
    VP_segmap_sterog = cv2.resize(VP_segmap_sterog, (W, H))
    Border_mask = cv2.resize(Border_mask, (W, H))

    _, labels = cv2.connectedComponents(VP_segmap_sterog, connectivity=8)
    lab = np.unique(labels)
    lab = np.delete(lab, np.where(lab == 0)) # exclude background
    box_masks = np.zeros((lab.shape[0], H, W), dtype=np.uint8)
    for i in range(0, lab.shape[0]):
        rows, cols = np.where(labels == lab[i])
        Px_no = countNonZero(VP_segmap_sterog[rows, cols]*Border_mask[rows, cols])
        if Px_no > perc:
            box_masks[i, rows, cols] = 1
        else:
            box_masks[i, rows, cols] = 0
            VP_segmap_sterog[rows, cols] = 0



    seg_mask = VP_segmap_sterog.astype('float32')
    box_masks = box_masks.astype('float32')

    box_masks = np.stack(box_masks, axis=0)
    seg_mask_padded = np.pad(seg_mask, [[self.Q, self.Q], [self.Q, self.Q]], "constant")
    box_masks_padded = np.pad(box_masks, [[0, 0], [self.Q, self.Q], [self.Q, self.Q]], "constant")

    x_mesh = x_mesh.astype('float32')
    y_mesh = y_mesh.astype('float32')
    x_meshsterog = x_meshsterog.astype('float32')
    y_meshsterog = y_meshsterog.astype('float32')

    x_mesh = np.pad(x_mesh, [[self.Q, self.Q], [self.Q, self.Q]], 'reflect', reflect_type='odd')
    y_mesh = np.pad(y_mesh, [[self.Q, self.Q], [self.Q, self.Q]], 'reflect', reflect_type='odd')
    x_meshsterog = np.pad(x_meshsterog, [[self.Q, self.Q], [self.Q, self.Q]], 'reflect', reflect_type='odd')
    y_meshsterog = np.pad(y_meshsterog, [[self.Q, self.Q], [self.Q, self.Q]], 'reflect', reflect_type='odd')

    mesh_uniform_padded = np.stack([x_mesh, y_mesh], axis=0)
    mesh_stereo_padded = np.stack([x_meshsterog, y_meshsterog], axis=0)

    radial_distance_padded = np.linalg.norm(mesh_uniform_padded, axis=0)
    x_dist = abs(x_mesh.min()-x_mesh.max())
    y_dist = abs(y_mesh.min()-y_mesh.max())
    half_diagonal = np.linalg.norm([y_dist, x_dist]) / 2.
    ra = half_diagonal / 2.
    rb = half_diagonal / (2 * np.log(99))
    correction_strength = 1 / (1 + np.exp(-(radial_distance_padded - ra) / rb))


    return mesh_uniform_padded, mesh_stereo_padded, correction_strength, seg_mask_padded, box_masks_padded, cv2.resize(VP_segmap_sterog,(self.W_vp, self.H_vp))


def __getitem__(self, index):
    index = index % len(self.data_list)
    data_name = self.data_list[index]

    return self.get_image_by_file(data_name)


def __len__(self):
    return len(self.data_list)
