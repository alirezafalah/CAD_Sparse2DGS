#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
# ============================ YOUR FIX GOES HERE ============================
import sys
# Add the two standard installation paths for Python 3.8 packages in Colab
sys.path.insert(0, '/content/drive/MyDrive/PhD_Projects/cad_detection/CAD_Sparse2DGS/submodules/simple-knn')
sys.path.insert(0, '/content/drive/MyDrive/PhD_Projects/cad_detection/CAD_Sparse2DGS/submodules/diff-surfel-rasterization')
# ============================================================================
from utils.loss_utils import l1_loss, ssim, get_fea_loss, get_disk_reg_loss, get_view_idx
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.vis_utils import visualize_depth, visualize_feature_map
import torchvision.transforms as transforms
from PIL import Image
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, render_net_image
from argparse import ArgumentParser, Namespace
from utils.general_utils import build_rotation
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.point_utils import compute_hom, compute_hom_v2
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1] * 11 if dataset.white_background else [0] * 11
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 500 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        render_pkg = render(viewpoint_cam,
                            gaussians,
                            pipe,
                            background,
                            fixed_color=gaussians.pre_all)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        # regularization
        lambda_normal = opt.lambda_normal if iteration > 1500 else 0.0
        lambda_dist = opt.lambda_dist if iteration > 300 else 0.0

        rend_dist = render_pkg["rend_dist"]
        rend_normal  = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()
        dist_loss = lambda_dist * (rend_dist).mean()
        fea_loss = get_fea_loss(render_pkg["feature_map"], viewpoint_cam.feature[0]) * opt.lambda_fea
        disl_regloss = get_disk_reg_loss(gaussians, 
                         viewpoint_cam, 
                         scene.getTrainCameras().copy()[1] if viewpoint_cam.image_name == scene.getTrainCameras().copy()[0].image_name else scene.getTrainCameras().copy()[0], 
                         surf_normal.detach(),
                         mask=gaussians.view_indexs[get_view_idx(scene.getTrainCameras().copy(), viewpoint_cam.image_name)].squeeze() > 0.5) * opt.lambda_dr

        # loss
        total_loss = loss + dist_loss + normal_loss + fea_loss + disl_regloss
        
        total_loss.backward(retain_graph=True)

        iter_end.record()

        with torch.no_grad():
            if iteration % 500 == 0:
                im_rend = transforms.ToPILImage()(image)
                im_gt = transforms.ToPILImage()(gt_image)
                r_n = transforms.ToPILImage()(rend_normal)
                s_n = transforms.ToPILImage()(surf_normal)
                depth = transforms.ToPILImage()(visualize_depth(render_pkg["surf_depth"].squeeze().cpu().numpy()))
                mvsdepth = transforms.ToPILImage()(visualize_depth(viewpoint_cam.depth.squeeze().cpu().numpy()))
                mvsfea = transforms.ToPILImage()(visualize_feature_map(viewpoint_cam.feature)) 
                rendfea = transforms.ToPILImage()(visualize_feature_map(render_pkg["feature_map"][None])) 
                _, h, w = gt_image.shape
                image = Image.new("RGB", (4*w, 2*h))
                image.paste(im_gt, (0, 0))
                image.paste(im_rend, (w, 0))
                image.paste(r_n, (2*w, 0))
                image.paste(s_n, (3*w, 0))
                image.paste(depth, (0, h))
                image.paste(mvsdepth, (w, h))
                image.paste(mvsfea, (2*w, h))
                image.paste(rendfea, (3*w, h))
                if not os.path.exists(os.path.join(dataset.model_path, 'vis')):
                    os.mkdir(os.path.join(dataset.model_path, 'vis'))
                save_path = os.path.join(dataset.model_path, 'vis/iteration{}.jpg'.format(iteration))
                image.save(save_path)
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log


            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "distort": f"{ema_dist_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)

                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)


            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    views = scene.getTrainCameras().copy()
                    depths = []
                    normals = []
                    nccs = []
                    all_points = []
                    gs_nccs = []
                    gs_rotation = build_rotation(gaussians.get_rotation) #n 3 3
                    gs_normal = torch.nn.functional.normalize(gs_rotation[..., 2], p=2.0, dim=-1, eps=1e-12, out=None)[None]#last axis #n 3
                    for i, view_ref in enumerate(views):
                        src_views = scene.getTrainCameras().copy()
                        src_views.pop(i)
                        render_pkg_d = render(view_ref, gaussians, pipe, background, fixed_color=gaussians.pre_all)
                        select_depth = render_pkg_d["surf_depth"]
                        select_world_normal = torch.nn.functional.normalize(render_pkg_d["rend_normal"], dim=0)#note render normal is unnormalized
                        select_cam_normal = torch.matmul(view_ref.w2c[:3, :3], select_world_normal.reshape(3, -1)).permute(1, 0)[None]#n 3
                        points = gaussians.get_point_from_depth(select_depth, view_ref)
                        ncc0, _, _ = compute_hom(select_depth, select_cam_normal, points, view_ref, view_src=src_views[0])
                        ncc1, _, _ = compute_hom(select_depth, select_cam_normal, points, view_ref, view_src=src_views[1])
                        ncc = torch.stack([ncc0, ncc1], dim=1)
                        ncc_min = torch.min(ncc, dim=1)[0]
                        depths.append(select_depth)
                        normals.append(select_world_normal)
                        nccs.append(ncc_min)
                        all_points.append(points)
                        view_index = gaussians.view_indexs[i].squeeze() > 0.5 #n 
                        gs_normals_camera = torch.matmul(gs_normal, view_ref.w2c[:3, :3].T)#1 3n 3
                        gs_normals_camera = gs_normals_camera.squeeze()[view_index][None]#1 n 3
                        gs_points = gaussians._xyz[view_index]#3n 3 -> n 3
                        gs_ncc0, _ = compute_hom_v2(view_ref.depth, gs_normals_camera, gs_points, view_ref, view_src=src_views[0], patch_size=3)
                        gs_ncc1, _ = compute_hom_v2(view_ref.depth, gs_normals_camera, gs_points, view_ref, view_src=src_views[1], patch_size=3)
                        gs_ncc = torch.stack([gs_ncc0, gs_ncc1], dim=1)
                        gs_ncc_min = torch.min(gs_ncc, dim=1)[0]
                        gs_nccs.append(gs_ncc_min)
                    gaussians.update_points(views, nccs, gs_nccs, all_points, normals)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        with torch.no_grad():        
            if network_gui.conn == None:
                network_gui.try_connect(dataset.render_items)
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                    if custom_cam != None:
                        render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer)   
                        net_image = render_net_image(render_pkg, dataset.render_items, render_mode, custom_cam)
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    metrics_dict = {
                        "#": gaussians.get_opacity.shape[0],
                        "loss": ema_loss_for_log
                        # Add more metrics as needed
                    }
                    # Send the data
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    # raise e
                    network_gui.conn = None

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, fixed_color=scene.gaussians.pre_all, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

                        try:
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)

                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                        except:
                            pass

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])

    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint)

    # All done
    print("\nTraining complete.")
