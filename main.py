
import cv2
import argparse
from Files import Data_input
from Files import VP_Rendering
from Files import VP_Rendering_BP_v2
from Files import Warp_fun
from Files import border_cut_v2
from Files import border_mask
from Files import get_opticalflow
from Optimization.src import data
from torch import optim
from Optimization.src.energy import Energy

##################### inputs and viewport rendering paramters  #####################################
parser = argparse.ArgumentParser(description='Locally Optimized Pannini Projection')
parser.add_argument('--ERP', type=str, default="Ind3", help="name of input 360 image")  ## name of input 360image
## Viewport resolution
parser.add_argument('--W_vp', type=int, default=1816, help="viewport width")
parser.add_argument('--H_vp', type=int, default=1020, help="viewport height")
## Pannini parameters for background
parser.add_argument('--d_back', type=float, default=0.5, help="d used for background")
parser.add_argument('--vc_back', type=float, default=0.6, help="vc used for background")
## Viewport viewing direction
parser.add_argument('--ph', type=float, default=-5, help="long")
parser.add_argument('--th', type=float, default=0, help="lat")
## Viewport FoV
parser.add_argument('--HFoV', type=float, default=150, help="HfoV")

parser.add_argument('--crop', type=int, default=0, help="to crop or not")
parser.add_argument('--obj_size', type=int, default=0.2, help="min object size")

##################### Optimization paramters and inputs #####################################
parser.add_argument('--num_iter', type=int, default=250, help="number of optimization steps")
parser.add_argument('--lr', type=float, default=0.005, help="learning rate")
parser.add_argument('--Q', type=int, default=8, help="number of padding vertices")
parser.add_argument('--mesh_ds_ratio', type=int, default=10, help="the pixel-to-vertex ratio")
parser.add_argument('--naive', type=int, default=0, help="if set True, perform naive orthographic correction")
parser.add_argument('--object_energy', type=float, default=0.25, help="weight of the face energy term")
parser.add_argument('--similarity', type=int, default=0, help="weight of similarity tranformation constraint")
parser.add_argument('--line_bending', type=float, default=2, help="weight of the line bending term")
parser.add_argument('--regularization', type=float, default=0.5, help="weight of the regularization term")
parser.add_argument('--boundary_constraint', type=float, default=4, help="weight of the mesh boundary constraint")
parser.add_argument('--margin', type=float, default=15, help="margin in degree to cut the borders after optimization")

if __name__ == '__main__':

    ## get arguments and inputs
    args = parser.parse_args()
    args.asp = args.W_vp / args.H_vp

    ## get 360 image and cooresponding semantic segmentation
    args.erp_name, args.erp_seg_name, args.ERP, args.ERP_Seg, args.H_erp, args.W_erp = Data_input.read_imputs(args.ERP)

    ## Expand FOV with some margin
    args.HFoV = args.HFoV+args.margin

    ## Pannini parameters for objects
    args.d_obj = args.d_back+0.1
    args.vc_obj = 0

    # Generate VP for background where lines are preserved using Pannini (d_back, vc_back)
    VP_rect, VP_seg = VP_Rendering.VP_rendering(args, args.ERP, args.ERP_Seg)  # VP rendering with Pannini

    #Generate two meshes
    VP_back, x_mesh_back, y_mesh_back, x_mesh_obj, y_mesh_obj, VP_Vhs, VP_Vvs, hfov, vfov = VP_Rendering_BP_v2.VP_rendering_BG(args, args.ERP, args.ERP_Seg, VP_rect, VP_seg)

    #Generate border mask
    args.Border_mask = border_mask.Vp_border_mask(args, hfov, vfov)

    #Asign meshes
    args.x_meshsterog = x_mesh_obj
    args.y_meshsterog = y_mesh_obj
    args.x_mesh = x_mesh_back
    args.y_mesh = y_mesh_back
    args.VP_segmap = VP_seg
    args.file = VP_back

    # Generate corrrection strength and padding meshes
    mesh_uniform_padded, mesh_stereo_padded, correction_strength, seg_mask_padded, box_masks_padded, _ = data.get_image_by_file(args)

    options = {
            "object_energy": args.object_energy,
            "similarity": args.similarity,
            "line_bending": args.line_bending,
            "regularization": args.regularization,
            "boundary_constraint": args.boundary_constraint}

    # Load the optimization model
    print("loading the optimization model")
    model = Energy(options, mesh_uniform_padded, mesh_stereo_padded, correction_strength, box_masks_padded,
                   seg_mask_padded, args.Q)
    optim = optim.Adam(model.parameters(), lr=args.lr)

    # Perform optimization
    print("optimizing")
    for i in range(args.num_iter):
        optim.zero_grad()
        loss = model.forward()
        loss.backward()
        optim.step()

    # Calculate optical flow from the optimized mesh
    print("calculating optical flow")
    mesh_uniform = mesh_uniform_padded[:, args.Q:-args.Q, args.Q:-args.Q].transpose([1, 2, 0])
    mesh_target = mesh_stereo_padded[:, args.Q:-args.Q, args.Q:-args.Q].transpose([1, 2, 0])
    mesh_optimal = model.mesh.detach().cpu().numpy()
    mesh_optimal = mesh_optimal[:, args.Q:-args.Q, args.Q:-args.Q].transpose([1, 2, 0])
    x_out, y_out = mesh_optimal[:, :, 0], mesh_optimal[:, :, 1]
    out = Warp_fun.Warp_im(args, 'Optimized.bmp', VP_rect, x_out, y_out,  VP_Vhs, VP_Vvs, args.W_vp, args.H_vp)
    #cv2.imwrite(out_name, out)
    #cv2.imwrite("Input.bmp", VP_rect)
    #cv2.imwrite("Input_seg.bmp", VP_seg*255)

    # Compute optical flow
    x_inp, y_inp = mesh_uniform[:, :, 0], mesh_uniform[:, :, 1]
    overlay_flow = get_opticalflow.optical_flow(args, x_out, y_out, x_inp, y_inp, VP_rect, VP_Vhs, VP_Vvs)

    # Generate outputs
    VP_cut, VP_seg_cut, VP_rect_cut, overlay_flow_cut = border_cut_v2.CutOut_border(args, hfov, vfov, out, VP_rect, VP_seg, overlay_flow)
    cv2.imwrite('output.bmp', VP_cut)
    #cv2.imwrite('output_seg.bmp', VP_seg_cut*255)
    cv2.imwrite('input.bmp', VP_rect_cut)
    cv2.imwrite('overlay_flow.bmp', overlay_flow_cut)


