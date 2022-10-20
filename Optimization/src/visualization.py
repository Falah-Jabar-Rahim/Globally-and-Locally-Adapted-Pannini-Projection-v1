import cv2
import io
import numpy as np
import matplotlib.pyplot as plt


def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def get_overlay_flow(image, flow_x, flow_y, color="aquamarine", ratio=0.7):

    H, W, _ = image.shape
    Hm, Wm = flow_x.shape

    fig = plt.figure(figsize=(W, H), dpi=1, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.axis("equal")
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow((image * ratio).astype(np.uint8))
    X = np.arange(0.5, Wm, 1).astype(np.float32) * (H / Hm)
    Y = np.arange(0.5, Hm, 1).astype(np.float32) * (W / Wm)
    X, Y = np.meshgrid(X, Y)

    # Code to reduce the number of arrows
    X_new = X*0
    Y_new = Y*0
    flow_x_new = flow_x*0
    flow_y_new = flow_y*0
    step_sz = 4
    for n in range(0, Hm, step_sz):
        for m in range(0, Wm, step_sz):
            X_new[n, m] = X[n, m]
            Y_new[n, m] = Y[n, m]
            flow_x_new[n, m] = flow_x[n, m]
            flow_y_new[n, m] = flow_y[n, m]
    X = X_new
    Y = Y_new
    flow_x = flow_x_new
    flow_y = flow_y_new
    q = ax.quiver(X, Y, flow_x, -flow_y, color=color, scale=1.0, scale_units='xy')

    # plt.show()

    im = get_img_from_fig(fig, dpi=1)
    im = (im / 255.).astype(np.float32)

    plt.close(fig)

    return im