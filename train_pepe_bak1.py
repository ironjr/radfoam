import os
import uuid
import yaml
import gc
import numpy as np
from PIL import Image
import configargparse
import tqdm
import warnings

warnings.filterwarnings("ignore")

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from data_loader import DataHandler
from configs import *
# from radfoam_model.scene import RadFoamScene
from radfoam_model.scene_noop import RadFoamScene
from radfoam_model.utils import psnr
import radfoam

# New
from collections import deque


seed = 42
torch.random.manual_seed(seed)
np.random.seed(seed)


def train(args, pipeline_args, model_args, optimizer_args, dataset_args):
    device = torch.device(model_args.device)
    # Setting up output directory
    if not pipeline_args.debug:
        if len(pipeline_args.experiment_name) == 0:
            unique_str = str(uuid.uuid4())[:8]
            experiment_name = f"{dataset_args.scene}@{unique_str}"
        else:
            experiment_name = pipeline_args.experiment_name
        out_dir = f"output/{experiment_name}"
        writer = SummaryWriter(out_dir, purge_step=0)
        os.makedirs(f"{out_dir}/test", exist_ok=True)
        print(f"Output directory: {out_dir}")

        def represent_list_inline(dumper, data):
            return dumper.represent_sequence(
                "tag:yaml.org,2002:seq", data, flow_style=True
            )

        yaml.add_representer(list, represent_list_inline)

        # Save the arguments to a YAML file
        with open(f"{out_dir}/config.yaml", "w") as yaml_file:
            yaml.dump(vars(args), yaml_file, default_flow_style=False)

    # Setting up dataset
    iter2downsample = dict(
        zip(
            dataset_args.downsample_iterations,
            dataset_args.downsample,
        )
    )
    train_data_handler = DataHandler(
        dataset_args, rays_per_batch=1_000_000, device=device
    )
    downsample = iter2downsample[0]
    train_data_handler.reload(split="train", downsample=downsample)

    test_data_handler = DataHandler(
        dataset_args, rays_per_batch=0, device=device
    )
    test_data_handler.reload(
        split="test", downsample=min(dataset_args.downsample)
    )
    test_ray_batch_fetcher = radfoam.BatchFetcher(
        test_data_handler.rays, batch_size=1, shuffle=False
    )
    test_rgb_batch_fetcher = radfoam.BatchFetcher(
        test_data_handler.rgbs, batch_size=1, shuffle=False
    )

    # Define viewer settings
    viewer_options = {
        "camera_pos": train_data_handler.viewer_pos,
        "camera_up": train_data_handler.viewer_up,
        "camera_forward": train_data_handler.viewer_forward,
    }

    # Setting up pipeline
    rgb_loss = nn.SmoothL1Loss(reduction="none")

    # Setting up model
    model = RadFoamScene(
        args=model_args,
        device=device,
        points=train_data_handler.points3D,
        points_colors=train_data_handler.points3D_colors,
    )

    # Setting up optimizer
    model.declare_optimizer(
        args=optimizer_args,
        warmup=pipeline_args.densify_from,
        max_iterations=pipeline_args.iterations,
    )

    # Add after model initialization
    damping_factor = args.damping_factor  # Adjustable Tikhonov regularization
    min_variance_threshold = args.min_variance_threshold  # Prevent division by zero

    def test_render(
        test_data_handler, ray_batch_fetcher, rgb_batch_fetcher, debug=False
    ):
        rays = test_data_handler.rays
        points, _, _, _ = model.get_trace_data()
        start_points = model.get_starting_point(
            rays[:, 0, 0].cuda(), points, model.aabb_tree
        )

        psnr_list = []
        with torch.no_grad():
            for i in range(rays.shape[0]):
                ray_batch = ray_batch_fetcher.next()[0]
                rgb_batch = rgb_batch_fetcher.next()[0]
                output, _, _, _, _ = model(ray_batch, start_points[i])

                # White background
                opacity = output[..., -1:]
                rgb_output = output[..., :3] + (1 - opacity)
                rgb_output = rgb_output.reshape(*rgb_batch.shape).clip(0, 1)

                img_psnr = psnr(rgb_output, rgb_batch).mean()
                psnr_list.append(img_psnr)
                torch.cuda.synchronize()

                if not debug:
                    error = np.uint8((rgb_output - rgb_batch).cpu().abs() * 255)
                    rgb_output = np.uint8(rgb_output.cpu() * 255)
                    rgb_batch = np.uint8(rgb_batch.cpu() * 255)

                    im = Image.fromarray(
                        np.concatenate([rgb_output, rgb_batch, error], axis=1)
                    )
                    im.save(
                        f"{out_dir}/test/rgb_{i:03d}_psnr_{img_psnr:.3f}.png"
                    )

        average_psnr = sum(psnr_list) / len(psnr_list)
        if not debug:
            f = open(f"{out_dir}/metrics.txt", "w")
            f.write(f"Average PSNR: {average_psnr}")
            f.close()

        return average_psnr

    def train_loop(viewer):
        print("Training")

        torch.cuda.synchronize()

        data_iterator = train_data_handler.get_iter()
        ray_batch, rgb_batch = next(data_iterator)

        triangulation_update_period = 1
        iters_since_update = 1
        iters_since_densification = 0
        next_densification_after = 1

        # Gradient queues temporarily placed here.
        # Parameters:
        # - model.primal_points: (num_points, 3)
        # - model.density: (num_points, 1)
        # - model.att_dc: (num_points, 3)
        # - model.att_sh: (num_points, 3 * ((sh_degree + 1) ** 2 - 1))
        num_cameras = train_data_handler.cameras.shape[0]
        stats_initialized = False
        param_mu = {n: None for n, _ in model.named_parameters()}
        param_sigma = {n: None for n, _ in model.named_parameters()}
        param_count = {n: None for n, _ in model.named_parameters()}

        with tqdm.trange(pipeline_args.iterations) as train:
            for i in train:
                if viewer is not None:
                    model.update_viewer(viewer)
                    viewer.step(i)

                if i in iter2downsample and i:
                    downsample = iter2downsample[i]
                    train_data_handler.reload(
                        split="train", downsample=downsample
                    )
                    data_iterator = train_data_handler.get_iter()
                    ray_batch, rgb_batch = next(data_iterator)

                depth_quantiles = (
                    torch.rand(*ray_batch.shape[:-1], 2, device=device)
                    .sort(dim=-1, descending=True)
                    .values
                )

                rgba_output, depth, _, _, _ = model(
                    ray_batch,
                    depth_quantiles=depth_quantiles,
                )

                # White background
                opacity = rgba_output[..., -1:]
                if pipeline_args.white_background:
                    rgb_output = rgba_output[..., :3] + (1 - opacity)
                else:
                    rgb_output = rgba_output[..., :3]


                # Compute losses
                color_loss = rgb_loss(rgb_batch, rgb_output)
                opacity_loss = ((1 - opacity) ** 2).mean()

                valid_depth_mask = (depth > 0).all(dim=-1)
                quant_loss = (depth[..., 0] - depth[..., 1]).abs()
                quant_loss = (quant_loss * valid_depth_mask).mean()
                w_depth = pipeline_args.quantile_weight * min(
                    2 * i / pipeline_args.iterations, 1
                )

                loss = color_loss.mean() + opacity_loss + w_depth * quant_loss


                # Optimize
                model.optimizer.zero_grad(set_to_none=True)

                # Hide latency of data loading behind the backward pass
                event = torch.cuda.Event()
                event.record()
                loss.backward()
                event.synchronize()
                ray_batch, rgb_batch = next(data_iterator)

                # Update sequential statistics
                for n, p in model.named_parameters():
                    g = p.grad.data.clone() * num_cameras
                    
                    if not stats_initialized:
                        param_mu[n] = g
                        param_sigma[n] = g.pow(2)  # Use squared gradients for GN approximation
                        param_count[n] = 1
                    else:
                        count = param_count[n]
                        # Update mean
                        param_mu[n] = param_mu[n] * (count / (count + 1)) + g / (count + 1)
                        # Update squared gradients
                        param_sigma[n] = param_sigma[n] * (count / (count + 1)) + g.pow(2) / (count + 1)
                        param_count[n] += 1
                if not stats_initialized:
                    stats_initialized = True

                # Update parameters using Gauss-Newton
                # if (i + 1) % num_cameras == 0:
                if True:
                    for n, p in model.named_parameters():
                        count = param_count[n]
                        if count > 1:
                            # p.grad.data = param_mu[n]
                            mu = param_mu[n]
                            variance = param_sigma[n] - param_mu[n].pow(2)  # Compute variance
                            # import pdb; pdb.set_trace()
                            variance = torch.clamp(variance, min=min_variance_threshold)
                            
                            # Compute GN update: H⁻¹g where H ≈ J^T J + λI
                            # Using variance as diagonal approximation of J^T J
                            # current_lr = model.xyz_scheduler_args(i)  # Get current learning rate
                            gn_update = mu / (variance + damping_factor)
                            # p.grad.data = (current_lr * gn_update).to(p.device)
                            p.grad.data = gn_update

                            learning_rate = 0.1
                            if n == "primal_points":
                                learning_rate *= 0.01
                            # elif n == "density":
                            #     learning_rate *= model.den_scheduler_args(i)
                            elif n == "att_dc":
                                learning_rate *= 0.1
                            elif n == "att_sh":
                                learning_rate *= 0.1 * args.sh_factor

                            # if n == "primal_points":
                            #     learning_rate *= model.xyz_scheduler_args(i)
                            # elif n == "density":
                            #     learning_rate *= model.den_scheduler_args(i)
                            # elif n == "att_dc":
                            #     learning_rate *= model.attr_dc_scheduler_args(i)
                            # elif n == "att_sh":
                            #     learning_rate *= model.attr_rest_scheduler_args(i)
                            print(f"Param: {n}, Learning rate: {learning_rate}, Variance: {variance.min()}, GN update: {gn_update.abs().max()}"
                                  f"Grad Size: {p.grad.data.abs().max()}")
                            p.data.sub_(p.grad.data * learning_rate)
                    
                    # model.optimizer.step()
                    
                    # Reset statistics after update
                    # if args.reset_stats_after_update:
                    if (i + 1) % num_cameras == 0:
                        stats_initialized = False
                        for k in param_mu.keys():
                            param_mu[k] = None
                            param_sigma[k] = None
                            param_count[k] = None

                model.update_learning_rate(i)


                # Log metrics
                train.set_postfix(color_loss=f"{color_loss.mean().item():.5f}")

                if i % 100 == 99 and not pipeline_args.debug:
                    writer.add_scalar("train/rgb_loss", color_loss.mean(), i)
                    num_points = model.primal_points.shape[0]
                    writer.add_scalar("test/num_points", num_points, i)

                    test_psnr = test_render(
                        test_data_handler,
                        test_ray_batch_fetcher,
                        test_rgb_batch_fetcher,
                        True,
                    )
                    writer.add_scalar("test/psnr", test_psnr, i)

                    writer.add_scalar(
                        "lr/points_lr", model.xyz_scheduler_args(i), i
                    )
                    writer.add_scalar(
                        "lr/density_lr", model.den_scheduler_args(i), i
                    )
                    writer.add_scalar(
                        "lr/attr_lr", model.attr_dc_scheduler_args(i), i
                    )

                # Update triangulation
                if iters_since_update >= triangulation_update_period:
                    model.update_triangulation(incremental=True)
                    iters_since_update = 0

                    if triangulation_update_period < 100:
                        triangulation_update_period += 2

                # Densify
                # iters_since_update += 1
                # if i + 1 >= pipeline_args.densify_from:
                #     iters_since_densification += 1

                # if (
                #     iters_since_densification == next_densification_after
                #     and model.primal_points.shape[0]
                #     < 0.9 * model.num_final_points
                # ):
                #     point_error, point_contribution = model.collect_error_map(
                #         train_data_handler, pipeline_args.white_background
                #     )
                #     model.prune_and_densify(
                #         point_error,
                #         point_contribution,
                #         pipeline_args.densify_factor,
                #     )

                #     model.update_triangulation(incremental=False)
                #     triangulation_update_period = 1
                #     gc.collect()

                #     # Linear growth
                #     iters_since_densification = 0
                #     next_densification_after = int(
                #         (
                #             (pipeline_args.densify_factor - 1)
                #             * model.primal_points.shape[0]
                #             * (
                #                 pipeline_args.densify_until
                #                 - pipeline_args.densify_from
                #             )
                #         )
                #         / (model.num_final_points - model.num_init_points)
                #     )
                #     next_densification_after = max(
                #         next_densification_after, 100
                #     )

                #     # TODO
                #     if True:
                #         for k in param_mu.keys():
                #             param_mu[k] = None
                #             param_sigma[k] = None
                #             param_count[k] = None
                #         stats_initialized = False

                # Freeze points
                if i == optimizer_args.freeze_points:
                    model.update_triangulation(incremental=False)

                # Break if viewer is closed
                if viewer is not None and viewer.is_closed():
                    break

        # Save scene and model
        model.save_ply(f"{out_dir}/scene.ply")
        model.save_pt(f"{out_dir}/model.pt")
        del data_iterator

    if pipeline_args.viewer:
        model.show(
            train_loop, iterations=pipeline_args.iterations, **viewer_options
        )
    else:
        train_loop(viewer=None)
    if not pipeline_args.debug:
        writer.close()

    test_render(
        test_data_handler,
        test_ray_batch_fetcher,
        test_rgb_batch_fetcher,
        pipeline_args.debug,
    )


def main():
    parser = configargparse.ArgParser(
        default_config_files=["arguments/mipnerf360_outdoor_config.yaml"]
    )

    model_params = ModelParams(parser)
    pipeline_params = PipelineParams(parser)
    optimization_params = OptimizationParams(parser)
    dataset_params = DatasetParams(parser)

    # Add argument to specify a custom config file
    parser.add_argument(
        "-c", "--config", is_config_file=True, help="Path to config file"
    )

    # Add new arguments
    parser.add_argument('--damping_factor', type=float, default=0.1)
    parser.add_argument('--min_variance_threshold', type=float, default=1e-12)
    parser.add_argument('--reset_stats_after_update', type=bool, default=False)

    # Parse arguments
    args = parser.parse_args()

    train(
        args,
        pipeline_params.extract(args),
        model_params.extract(args),
        optimization_params.extract(args),
        dataset_params.extract(args),
    )


if __name__ == "__main__":
    main()
