

def xy_to_mn(self, x_mesh, y_mesh):
    h = self.H_vp
    w = self.W_vp
    VP_Vhs = abs(x_mesh.min() - x_mesh.max())
    VP_Vvs = abs(y_mesh.min() - y_mesh.max())
    u = x_mesh + (VP_Vhs / 2)
    v = (VP_Vvs / 2) - y_mesh
    m = ((u * w) / VP_Vhs) - 0.5
    n = ((h * v) / VP_Vvs) - 0.5

    return m, n
