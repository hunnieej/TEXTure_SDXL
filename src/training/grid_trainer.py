from pathlib import Path
from typing import Any, Dict, Union, List

import cv2
import einops
import imageio
import numpy as np
import pyrallis
import torch
import torch.nn.functional as F
from PIL import Image
from loguru import logger
from matplotlib import cm
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional

from src import utils
from src.configs.train_config import TrainConfig
from src.models.textured_mesh import TexturedMeshModel
from src.models.grid_textured_mesh import GridTexturedMeshModel
from src.stable_diffusion_depth import StableDiffusion #Base TEXTure
from src.stable_diffusion_depth_sdxl_inpaint import StableDiffusion_inpaintXL #SDv2_Depth + SDXL inpaint 1.0
from src.sdxl_depth import SDXL #SDXL base 1.0 + SDXL inpaint 1.0
from src.training.views_dataset_grid import ViewsDataset, MultiviewDataset
from src.utils import make_path, tensor2numpy

import time # texture generation time calculation
import os
from pathlib import Path

import torchvision.utils as vutils

class TEXTureGrid:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.paint_step = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ncount = 0
        self.texturecount = 0
        self.image_count = 0  # Counter for valid images

        utils.seed_everything(self.cfg.optim.seed)

        # Make view_dirs
        self.exp_path = make_path(self.cfg.log.exp_dir)
        self.ckpt_path = make_path(self.exp_path / 'checkpoints')
        self.train_renders_path = make_path(self.exp_path / 'vis' / 'train')
        self.eval_renders_path = make_path(self.exp_path / 'vis' / 'eval')
        self.final_renders_path = make_path(self.exp_path / 'results')

        self.init_logger()
        pyrallis.dump(self.cfg, (self.exp_path / 'config.yaml').open('w'))

        self.view_dirs = ['front', 'left', 'back', 'right', 'overhead', 'bottom']
        # Mesh 불러오는 과정
        self.mesh_model = self.init_mesh_model()
        # Diffusion Initialization
        self.diffusion = self.init_diffusion()
        # Text_embeddings initialization
        self.text_z, self.text_string ,self.text_z_origin, self.text_string_origin= self.calc_text_embeddings()
        self.dataloaders = self.init_dataloaders()
        self.back_im = torch.Tensor(np.array(Image.open(self.cfg.guide.background_img).convert('RGB'))).to(
            self.device).permute(2, 0,
                                 1) / 255.0

        logger.info(f'Successfully initialized {self.cfg.log.exp_name}')

    def init_mesh_model(self) -> nn.Module:
        cache_path = Path('cache') / Path(self.cfg.guide.shape_path).stem
        cache_path.mkdir(parents=True, exist_ok=True)
        model = GridTexturedMeshModel(self.cfg.guide, device=self.device,
                                  render_grid_size=self.cfg.render.train_grid_size,
                                  cache_path=cache_path,
                                  texture_resolution=self.cfg.guide.texture_resolution,
                                  augmentations=False)

        model = model.to(self.device)
        logger.info(
            f'Loaded Mesh, #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')
        logger.info(model)
        return model

    def init_diffusion(self) -> Any:
        # diffusion model로 stable_diffusion_depth.py의 StableDiffusion 클래스 사용
        # Original Code : Stable Diffusion v2
        # diffusion_model = StableDiffusion(self.device, model_name=self.cfg.guide.diffusion_name,
        #                                   concept_name=self.cfg.guide.concept_name,
        #                                   concept_path=self.cfg.guide.concept_path,
        #                                   latent_mode=False,
        #                                   min_timestep=self.cfg.optim.min_timestep,
        #                                   max_timestep=self.cfg.optim.max_timestep,
        #                                   no_noise=self.cfg.optim.no_noise,
        #                                   use_inpaint=True)
        
        # New model : SDXL base 1.0
        diffusion_model = SDXL(self.device, model_name=self.cfg.guide.diffusion_name,
                                          concept_name=self.cfg.guide.concept_name,
                                          concept_path=self.cfg.guide.concept_path,
                                          latent_mode=False,
                                          min_timestep=self.cfg.optim.min_timestep,
                                          max_timestep=self.cfg.optim.max_timestep,
                                          no_noise=self.cfg.optim.no_noise,
                                          use_inpaint=True, use_autodepth=False)

        for p in diffusion_model.parameters():
            p.requires_grad = False
        return diffusion_model

    def calc_text_embeddings(self) -> Union[torch.Tensor, List[torch.Tensor]]:
        ref_text = self.cfg.guide.text
        
        if not self.cfg.guide.append_direction:
            text_z = self.diffusion.get_text_embeds([ref_text])
            text_string = ref_text
        else:
            text_z = []
            text_string = []
            text_string_origin = ref_text.split('{', 1)[0].strip()
            text_z_origin = self.diffusion.get_text_embeds([ref_text.split('{', 1)[0].strip()])
            for d in self.view_dirs:
                text = ref_text.format(d)
                text_string.append(text)
                logger.info(text)
                negative_prompt = None
                logger.info(negative_prompt)
                text_z.append(self.diffusion.get_text_embeds([text], negative_prompt=negative_prompt))
        return text_z, text_string, text_z_origin, text_string_origin

    def init_dataloaders(self) -> Dict[str, DataLoader]:
        init_train_dataloader = MultiviewDataset(self.cfg.render, device=self.device).dataloader()

        val_loader = ViewsDataset(self.cfg.render, device=self.device,
                                  size=self.cfg.log.eval_size).dataloader()
        # Will be used for creating the final video
        val_large_loader = ViewsDataset(self.cfg.render, device=self.device,
                                        size=self.cfg.log.full_eval_size).dataloader()
        dataloaders = {'train': init_train_dataloader, 'val': val_loader,
                       'val_large': val_large_loader}
        return dataloaders

    def init_logger(self):
        logger.remove()  # Remove default logger
        log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> <level>{message}</level>"
        logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, format=log_format)
        logger.add(self.exp_path / 'log.txt', colorize=False, format=log_format)

    def paint(self):
        logger.info('Starting training ^_^')
        # Evaluate the initialization
        self.evaluate(self.dataloaders['val'], self.eval_renders_path)
        # TexturedMeshModle to training Mode <-> self.mesh_model.eval()
        self.mesh_model.train()

        pbar = tqdm(total=len(self.dataloaders['train']), initial=self.paint_step,
                    bar_format='{desc}: {percentage:3.0f}% painting step {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        #self.dataloaders : train, val, val_large dict
        #train 에는 Mesh, Phi, theta등 카메라 뷰 parameter 포함됨
        for data in self.dataloaders['train']:
            if self.paint_step == 0:
                self.paint_step += 1
                pbar.update(1)
                self.paint_viewpoint_initial(data)
                self.evaluate(self.dataloaders['val'], self.eval_renders_path)
                self.mesh_model.train()

            else :
                self.paint_step += 1
                pbar.update(1)
                self.paint_viewpoint(data)
                self.evaluate(self.dataloaders['val'], self.eval_renders_path)
                self.mesh_model.train()

        self.mesh_model.change_default_to_median()
        logger.info('Finished Painting ^_^')
        logger.info('Saving the last result...')
        self.full_eval()
        logger.info('\tDone!')

    # Dataloader을 이용해 mesh_model의 현재상태 평가,save picture, video 생성
    def evaluate(self, dataloader: DataLoader, save_path: Path, save_as_video: bool = False):
        logger.info(f'Evaluating and saving model, painting iteration #{self.paint_step}...')
        self.mesh_model.eval()
        save_path.mkdir(exist_ok=True)

        if save_as_video:
            all_preds = []
        for i, data in enumerate(dataloader):
            preds, textures, depths, normals = self.eval_render(data)

            pred = tensor2numpy(preds[0])

            if save_as_video:
                all_preds.append(pred)
            else:
                # Image.fromarray(pred).save(save_path / f"step_{self.paint_step:05d}_{i:04d}_rgb.jpg")
                # Image.fromarray((cm.seismic(normals[0, 0].cpu().numpy())[:, :, :3] * 255).astype(np.uint8)).save(
                #   save_path / f'{self.paint_step:04d}_{i:04d}_normals_cache.jpg')
                if self.paint_step == 0:
                    # Also save depths for debugging
                    torch.save(depths[0], save_path / f"{i:04d}_depth.pt")

        # Texture map is the same, so just take the last result
        texture = tensor2numpy(textures[0])
        Image.fromarray(texture).save(save_path / f"step_{self.paint_step:05d}_texture.png")

        if save_as_video:
            all_preds = np.stack(all_preds, axis=0)

            dump_vid = lambda video, name: imageio.mimsave(save_path / f"step_{self.paint_step:05d}_{name}.mp4", video,
                                                           fps=25,
                                                           quality=8, macro_block_size=1)

            dump_vid(all_preds, 'rgb')
        logger.info('Done!')

    def full_eval(self, output_dir: Path = None):
        if output_dir is None:
            output_dir = self.final_renders_path
        self.evaluate(self.dataloaders['val_large'], output_dir, save_as_video=True)
        # except:
        #     logger.error('failed to save result video')

        if self.cfg.log.save_mesh:
            save_path = make_path(self.exp_path / 'mesh')
            logger.info(f"Saving mesh to {save_path}")

            self.mesh_model.export_mesh(save_path)

            logger.info(f"\tDone!")

    def paint_viewpoint(self, data: Dict[str, Any]):
        logger.info(f'--- Painting step #{self.paint_step} ---')
        theta, phi, radius = data['theta'], data['phi'], data['radius']
        # If offset of phi was set from code

        phi = phi - np.deg2rad(self.cfg.render.front_offset)
        phi = float(phi + 2 * np.pi if phi < 0 else phi)
        logger.info(f'Painting from theta: {theta}, phi: {phi}, radius: {radius}')

        # Set background image
        if self.cfg.guide.use_background_color:
            background = torch.Tensor([0, 0.8, 0]).to(self.device)
        else:
            background = F.interpolate(self.back_im.unsqueeze(0),
                                       (self.cfg.render.train_grid_size, self.cfg.render.train_grid_size),
                                       mode='bilinear', align_corners=False)

        # Render from viewpoint
        # 여기서 정면 시작, Rgb 이미지, depth를 뽑는다.
        # outputs[Tensor] : image, mask, background, foreground, depth, normals, render_cache(uv_features, face_normals, face_idx, depth_map), texture_map
        outputs = self.mesh_model.render(theta=theta, phi=phi, radius=radius, background=background)
        render_cache = outputs['render_cache']
        rgb_render_raw = outputs['image']  # Render where missing values have special color
        depth_render = outputs['depth']
        # Render again with the median value to use as rgb, we shouldn't have color leakage, but just in case
        outputs = self.mesh_model.render(background=background,
                                         render_cache=render_cache, use_median=self.paint_step > 1)
        rgb_render = outputs['image']
        # Render meta texture map
        meta_output = self.mesh_model.render(background=torch.Tensor([0, 0, 0]).to(self.device),
                                             use_meta_texture=True, render_cache=render_cache)
        
        texture_map = meta_output['texture_map']
        self.log_train_image(texture_map, 'texture_map')

        z_normals = outputs['normals'][:, -1:, :, :].clamp(0, 1)
        # z_normals_cache = meta_output['image'].clamp(0, 1)
        z_normals_cache = meta_output['normals'][:,-1:,:,:].clamp(0,1)
        edited_mask = meta_output['image'].clamp(0, 1)[:, 1:2]

        self.log_train_image(rgb_render, 'rendered_input')
        self.log_train_image(depth_render[0, 0], 'depth', colormap=True)
        self.log_train_image(z_normals[0, 0], 'z_normals', colormap=True)
        # self.log_train_image(z_normals_test[0, 0], 't_normals', colormap=True)
        self.log_train_image(z_normals_cache[0, 0], 'z_normals_cache', colormap=True)

        # text embeddings
        if self.cfg.guide.append_direction:
            dirs = data['dir']  # [B,]
            text_z = self.text_z[dirs]
            text_string = self.text_string[dirs]
        else:
            text_z = self.text_z
            text_string = self.text_string
        logger.info(f'text: {text_string}')
        
        update_mask, generate_mask, refine_mask = self.calculate_trimap(rgb_render_raw=rgb_render_raw,
                                                                        depth_render=depth_render,
                                                                        z_normals=z_normals,
                                                                        z_normals_cache=z_normals_cache,
                                                                        edited_mask=edited_mask,
                                                                        mask=outputs['mask'])

        update_ratio = float(update_mask.sum() / (update_mask.shape[2] * update_mask.shape[3]))
        if self.cfg.guide.reference_texture is not None and update_ratio < 0.01:
            logger.info(f'Update ratio {update_ratio:.5f} is small for an editing step, skipping')
            return

        self.log_train_image(rgb_render * (1 - update_mask), name='masked_input')
        self.log_train_image(rgb_render * refine_mask, name='refine_regions')

        # Crop to inner region based on object mask
        min_h, min_w, max_h, max_w = utils.get_nonzero_region(outputs['mask'][0, 0])
        crop = lambda x: x[:, :, min_h:max_h, min_w:max_w]
        cropped_rgb_render = crop(rgb_render)
        cropped_depth_render = crop(depth_render)
        cropped_update_mask = crop(update_mask)
        self.log_train_image(cropped_rgb_render, name='cropped_input')

        # checker_mask = None
        no_checker_mask = None
        # if self.paint_step > 1 or self.cfg.guide.initial_texture is not None:
        #     checker_mask = self.generate_checkerboard(crop(update_mask), crop(refine_mask),
        #                                               crop(generate_mask))
        #     self.log_train_image(F.interpolate(cropped_rgb_render, (1024, 1024)) * (1 - checker_mask),
        #                          'checkerboard_input')
        if self.paint_step > 1 or self.cfg.guide.initial_texture is not None:
            no_checker_mask = self.generate_mask(crop(update_mask), crop(refine_mask),
                                                      crop(generate_mask))
            self.log_train_image(F.interpolate(cropped_rgb_render, (1024, 1024)) * (1 - no_checker_mask),
                                 'refine_mask_input')
        self.diffusion.use_inpaint = self.cfg.guide.use_inpainting and self.paint_step > 1

        cropped_rgb_output, steps_vis = self.diffusion.img2img_step(
                text_z, 
                cropped_rgb_render.detach(),
                cropped_depth_render.detach(),
                guidance_scale=self.cfg.guide.guidance_scale,
                strength=1.0, update_mask=cropped_update_mask,
                fixed_seed=self.cfg.optim.seed,
                check_mask=no_checker_mask,
                intermediate_vis=self.cfg.log.vis_diffusion_steps)
        self.log_train_image(cropped_rgb_output, name='direct_output')
        self.log_diffusion_steps(steps_vis)

        cropped_rgb_output = F.interpolate(cropped_rgb_output,
                                           (cropped_rgb_render.shape[2], cropped_rgb_render.shape[3]),
                                           mode='bilinear', align_corners=False)

        # Extend rgb_output to full image size
        rgb_output = rgb_render.clone()
        rgb_output[:, :, min_h:max_h, min_w:max_w] = cropped_rgb_output
        self.log_train_image(rgb_output, name='full_output')

        # Project back
        object_mask = outputs['mask']
        fitted_pred_rgb, _ = self.project_back(render_cache=render_cache, background=background, rgb_output=rgb_output,
                                               object_mask=object_mask, update_mask=update_mask, z_normals=z_normals,
                                               z_normals_cache=z_normals_cache)
        self.log_train_image(fitted_pred_rgb, name='fitted')

        return

    def paint_viewpoint_initial(self, data: Dict[str, Any]):
        logger.info(f'--- Painting step #{self.paint_step} ---')
        theta, phi, radius = data['theta'], data['phi'], data['radius']
        # phi_angles = [np.pi, np.pi/2, 3*np.pi/2, 0]
        phi_angles = [0, np.pi/2, 3*np.pi/2, np.pi]
        cropped_renders = []
        cropped_depths = []
        cropped_masks = []
        render_caches = []
        object_masks = []
        update_masks = []
        generate_masks = []
        z_normals_list = []
        z_normals_caches = []
        rgb_renders = []
        min_hs = []
        min_ws = []
        max_hs = []
        max_ws = []
        depth_renders = []
        edited_masks = []
        refine_masks = []


        for phi in phi_angles:
            phi = phi - np.deg2rad(self.cfg.render.front_offset)
            phi = float(phi + 2 * np.pi if phi < 0 else phi)
            logger.info(f'Painting from theta: {theta}, phi: {phi}, radius: {radius}')

            # Set background image
            if self.cfg.guide.use_background_color:
                background = torch.Tensor([0, 0.8, 0]).to(self.device)
            else:
                #background image를 grid size로 맞춤
                #self.back_im.unsqueeze(0) : [1, 3, 1024, 1024]
                background = F.interpolate(self.back_im.unsqueeze(0),
                                        (self.cfg.render.train_grid_size, self.cfg.render.train_grid_size),
                                        mode='bilinear', align_corners=False)

            # Rendering Process : Kaolin을 이용해 depthMap, Rendered image를 얻음
            # outputs[Tensor] : image, mask, background, foreground, depth, normals, render_cache(uv_features, face_normals, face_idx, depth_map), texture_map
            outputs = self.mesh_model.render(theta=theta, phi=phi, radius=radius, background=background)
            render_cache = outputs['render_cache']
            rgb_render_raw = outputs['image']  # Render where missing values have special color
            depth_render = outputs['depth']
            # Render again with the median value to use as rgb, we shouldn't have color leakage, but just in case
            outputs = self.mesh_model.render(background=background,
                                            render_cache=render_cache, use_median=self.paint_step > 1)
            rgb_render = outputs['image']
            # Render meta texture map - Kaolin 사용
            meta_output = self.mesh_model.render(background=torch.Tensor([0, 0, 0]).to(self.device),
                                                use_meta_texture=True, render_cache=render_cache)

            z_normals = outputs['normals'][:, -1:, :, :].clamp(0, 1)
            # z_normals_cache = meta_output['image'].clamp(0, 1)
            z_normals_cache = meta_output['normals'][:,-1:,:,:].clamp(0, 1)
            edited_mask = meta_output['image'].clamp(0, 1)[:, 1:2]

            # self.log_train_image(rgb_render, 'rendered_input_initial')
            # self.log_train_image(depth_render[0, 0], 'depth_initial', colormap=True)
            # self.log_train_image(z_normals[0, 0], 'z_normals_initial', colormap=True)
            self.log_train_image(z_normals_cache[0, 0], 'z_normals_cache_initial', colormap=True)

            # text embeddings
            if self.cfg.guide.append_direction:
                dirs = data['dir']  # [B,]
                text_z = self.text_z
                text_z_origin = self.text_z_origin
                text_string = self.text_string
                text_string_origin = self.text_string_origin
            else:
                text_z = self.text_z
                text_z_origin = self.text_z_origin
                text_string = self.text_string
                text_string_origin = self.text_string_origin
            logger.info(f'text: {text_string_origin}')
            
            #Making Trimap_original
            update_mask, generate_mask, refine_mask = self.calculate_trimap(rgb_render_raw=rgb_render_raw,
                                                                            depth_render=depth_render,
                                                                            z_normals=z_normals,
                                                                            z_normals_cache=z_normals_cache,
                                                                            edited_mask=edited_mask,
                                                                            mask=outputs['mask'])

            update_ratio = float(update_mask.sum() / (update_mask.shape[2] * update_mask.shape[3]))
            if self.cfg.guide.reference_texture is not None and update_ratio < 0.01:
                logger.info(f'Update ratio {update_ratio:.5f} is small for an editing step, skipping')
                return

            self.log_train_image(rgb_render * (1 - update_mask), name='masked_input')
            self.log_train_image(rgb_render * refine_mask, name='refine_regions')
            
            # Crop to inner region based on object mask
            min_h, min_w, max_h, max_w = utils.get_nonzero_region(outputs['mask'][0, 0])
            crop = lambda x: x[:, :, min_h:max_h, min_w:max_w]
            cropped_rgb_render = crop(rgb_render)
            cropped_depth_render = crop(depth_render)
            cropped_update_mask = crop(update_mask)

            #Add to list for concatenate
            cropped_renders.append(cropped_rgb_render)
            cropped_depths.append(cropped_depth_render)
            cropped_masks.append(cropped_update_mask)
            
            # Save the required tensors for each view
            rgb_renders.append(rgb_render)
            render_caches.append(render_cache)
            object_masks.append(outputs['mask'])
            update_masks.append(update_mask)
            z_normals_list.append(z_normals)
            z_normals_caches.append(z_normals_cache)
            depth_renders.append(depth_render)
            edited_masks.append(edited_mask)
            refine_masks.append(refine_mask)
            generate_masks.append(generate_mask)

            min_hs.append(min_h)
            min_ws.append(min_w)
            max_hs.append(max_h)
            max_ws.append(max_w)

            self.log_train_image(cropped_rgb_render, name='cropped_input')

        # Find the minimum height and width among the cropped images
        min_height = min([img.shape[2] for img in cropped_renders])
        min_width = min([img.shape[3] for img in cropped_renders])

        # Resize all cropped images to the minimum height and width
        cropped_renders_r = [F.interpolate(img, size=(min_height, min_width), mode='bilinear', align_corners=False) for img in cropped_renders]
        cropped_depths_r = [F.interpolate(img, size=(min_height, min_width), mode='bilinear', align_corners=False) for img in cropped_depths]
        cropped_masks_r = [F.interpolate(img, size=(min_height, min_width), mode='bilinear', align_corners=False) for img in cropped_masks]

        # Concatenate the cropped images into a 2x2 grid
        cropped_rgb_render_2x2 = torch.cat([
            torch.cat([cropped_renders_r[0], cropped_renders_r[1]], dim=3),
            torch.cat([cropped_renders_r[2], cropped_renders_r[3]], dim=3)
        ], dim=2)
        cropped_depth_render_2x2 = torch.cat([
            torch.cat([cropped_depths_r[0], cropped_depths_r[1]], dim=3),
            torch.cat([cropped_depths_r[2], cropped_depths_r[3]], dim=3)
        ], dim=2)
        cropped_update_mask_2x2 = torch.cat([
            torch.cat([cropped_masks_r[0], cropped_masks_r[1]], dim=3),
            torch.cat([cropped_masks_r[2], cropped_masks_r[3]], dim=3)
        ], dim=2)

        # Resize the concatenated image to the required size for the diffusion process
        cropped_rgb_render_2x2 = F.interpolate(cropped_rgb_render_2x2, (1024, 1024), mode='bilinear', align_corners=False)
        cropped_depth_render_2x2 = F.interpolate(cropped_depth_render_2x2, (1024, 1024), mode='bilinear', align_corners=False)
        cropped_update_mask_2x2 = F.interpolate(cropped_update_mask_2x2, (1024, 1024), mode='bilinear', align_corners=False)

        # checker_mask = None
        no_checker_mask = None
        # if self.paint_step > 1 or self.cfg.guide.initial_texture is not None:
        #     checker_mask = self.generate_checkerboard(cropped_update_mask_2x2, cropped_update_mask_2x2,
        #                                             cropped_update_mask_2x2)
        #     self.log_train_image(F.interpolate(cropped_rgb_render_2x2, (1024, 1024)) * (1 - checker_mask),
        #                         'checkerboard_input')
        if self.paint_step > 1 or self.cfg.guide.initial_texture is not None:
            no_checker_mask = self.generate_mask(crop(update_mask), crop(refine_mask),
                                                      crop(generate_mask))
            self.log_train_image(F.interpolate(cropped_rgb_render, (1024, 1024)) * (1 - no_checker_mask),
                                 'refine_mask_input')
        self.diffusion.use_inpaint = self.cfg.guide.use_inpainting and self.paint_step > 1

        # Diffusion Process with 2x2 grid
        cropped_rgb_output, steps_vis = self.diffusion.img2img_step(
            text_z_origin, 
            cropped_rgb_render_2x2.detach(),
            cropped_depth_render_2x2.detach(),
            guidance_scale=self.cfg.guide.guidance_scale,
            strength=1.0, update_mask=cropped_update_mask_2x2,
            fixed_seed=self.cfg.optim.seed,
            check_mask=no_checker_mask,
            intermediate_vis=self.cfg.log.vis_diffusion_steps)
        
        self.log_train_image(cropped_rgb_output, name='direct_output_initial')
        self.log_diffusion_steps(steps_vis)

        # Split the 2x2 grid into four separate images
        split_images = torch.split(cropped_rgb_output, 512, dim=2)
        top_left = torch.split(split_images[0], 512, dim=3)[0]
        top_right = torch.split(split_images[0], 512, dim=3)[1]
        bottom_left = torch.split(split_images[1], 512, dim=3)[0]
        bottom_right = torch.split(split_images[1], 512, dim=3)[1]


        # Resize each image to match the size of the corresponding cropped render
        resized_top_left = F.interpolate(top_left, size=(cropped_renders[0].shape[2], cropped_renders[0].shape[3]), mode='bilinear', align_corners=False)
        resized_top_right = F.interpolate(top_right, size=(cropped_renders[1].shape[2], cropped_renders[1].shape[3]), mode='bilinear', align_corners=False)
        resized_bottom_left = F.interpolate(bottom_left, size=(cropped_renders[2].shape[2], cropped_renders[2].shape[3]), mode='bilinear', align_corners=False)
        resized_bottom_right = F.interpolate(bottom_right, size=(cropped_renders[3].shape[2], cropped_renders[3].shape[3]), mode='bilinear', align_corners=False)

        # Project back
        # for i, cropped_rgb_out in enumerate([resized_top_left, resized_top_right, resized_bottom_left, resized_bottom_right]):
        #     rgb_output = rgb_renders[i].clone()
        #     rgb_output[:, :, min_hs[i]:max_hs[i], min_ws[i]:max_ws[i]] = cropped_rgb_out

        #     fitted_pred_rgb, _ = self.project_back(
        #         render_cache=render_caches[i], 
        #         background=background, 
        #         rgb_output=rgb_output,
        #         object_mask=object_masks[i], 
        #         update_mask=update_masks[i], 
        #         z_normals=z_normals_list[i],
        #         z_normals_cache=z_normals_caches[i])
            
        #     self.save_vu_image(fitted_pred_rgb, f'fitted_{i}_rgb')
        #     self.save_uv_map(self.dataloaders['val'], self.eval_renders_path, 'collapsed')

        # prev_texture_mask = None
        # prev_z_normals_cache = None

        # NOTE: 입력 순서 0도, 90도, 270도, 180도
        # cropped_rgb_output, steps_vis = self.diffusion.img2img_step(
                # text_z, 
                # cropped_rgb_render.detach(),
                # cropped_depth_render.detach(),
                # guidance_scale=self.cfg.guide.guidance_scale,
                # strength=1.0, update_mask=cropped_update_mask,
                # fixed_seed=self.cfg.optim.seed, 
                # check_mask=no_checker_mask,
                # intermediate_vis=self.cfg.log.vis_diffusion_steps)
        prev_texture_mask = None

        for i, cropped_rgb_out in enumerate([
            resized_top_left, resized_top_right,
            resized_bottom_left, resized_bottom_right
        ]):
            # RGB output 구성
            rgb_output = rgb_renders[i].clone()
            rgb_output[:, :, min_hs[i]:max_hs[i], min_ws[i]:max_ws[i]] = cropped_rgb_out

            # 이전까지의 텍스처로 현재 view를 렌더링 → trimap 계산용
            # rendered_after_texture = self.mesh_model.render(
            #     theta=theta, phi=phi_angles[i], radius=radius,
            #     background=background, use_meta_texture=False
            # )
            # rgb_render_raw = rendered_after_texture['image']
            # rgb_render_raws = []
            # rgb_render_raws.append(rgb_render_raw)
            # self.log_train_image(rgb_render_raw, name=f'rgb_render_raw_{i}')
            fitted_pred_rgb = self.mesh_model.render(
                theta=theta, phi=phi_angles[i], radius=radius,
                background=background, use_meta_texture=False)
            rgb_render_raw = fitted_pred_rgb['image']
            self.log_train_image(rgb_render_raw, name=f'rgb_render_raw_{i}')

            # 각 view별 trimap 계산
            if i == 0:
                # 첫 번째 뷰는 초기 update_mask 그대로 사용
                update_mask = update_masks[i]
                refine_mask = refine_masks[i]
                generate_mask = generate_masks[i]

            elif i == 1:
                update_mask, generate_mask, refine_mask= self.calculate_trimap_ref(
                    rgb_render_raw=rgb_render_raw,
                    depth_render=depth_renders[i],
                    z_normals=z_normals_list[i],
                    z_normals_cache=z_normals_caches[i],  # 정면 기준
                    edited_mask=edited_masks[i],
                    mask=object_masks[i]
                )

            elif i == 2:
                update_mask, generate_mask, refine_mask = self.calculate_trimap_ref(
                    rgb_render_raw=rgb_render_raw,
                    depth_render=depth_renders[i],
                    z_normals=z_normals_list[i],
                    z_normals_cache=z_normals_caches[i],  # 정면 기준
                    edited_mask=edited_masks[i],
                    mask=object_masks[i]
                )

            elif i == 3:
                update_mask, generate_mask, refine_mask = self.calculate_trimap_ref(
                    rgb_render_raw=rgb_render_raw,
                    depth_render=depth_renders[i],
                    z_normals=z_normals_list[i],
                    z_normals_cache=z_normals_caches[i],  # 90도 기준
                    edited_mask=edited_masks[i],
                    mask=object_masks[i]
                )

            update_masks[i] = update_mask
            refine_masks[i] = refine_mask
            generate_masks[i] = generate_mask
            self.log_train_image(rgb_render_raw * (1 - update_mask), name=f'masked_input_{i}')
            self.log_train_image(rgb_render_raw * update_mask, name=f'update_regions_{i}')
            self.log_train_image(rgb_render_raw * refine_mask, name=f'refine_regions_{i}')
            self.log_train_image(rgb_render_raw * generate_mask, name=f'generate_regions_{i}')

            fitted_pred_rgb, _ = self.project_back_ref(
                render_cache=render_caches[i],
                background=background,
                rgb_output=rgb_output,
                object_mask=object_masks[i],
                update_mask=update_mask,
                z_normals=z_normals_list[i],
                z_normals_cache=z_normals_caches[i],
                refine_mask=refine_masks[i]
            )

            # 이후 view를 위한 texture mask 업데이트
            prev_texture_mask = self.get_texture_mask_from_render(fitted_pred_rgb.detach())

            # 시각화 및 저장
            self.save_vu_image(fitted_pred_rgb, f"fitted_{i}_rgb")
            self.save_uv_map(self.dataloaders['val'], self.eval_renders_path, 'collapsed')

        return

    def eval_render(self, data):
        theta = data['theta']
        phi = data['phi']
        radius = data['radius']
        phi = phi - np.deg2rad(self.cfg.render.front_offset)
        phi = float(phi + 2 * np.pi if phi < 0 else phi)
        dim = self.cfg.render.eval_grid_size
        outputs = self.mesh_model.render(theta=theta, phi=phi, radius=radius,
                                         dims=(dim, dim), background='white')
        z_normals = outputs['normals'][:, -1:, :, :].clamp(0, 1)
        rgb_render = outputs['image']  # .permute(0, 2, 3, 1).contiguous().clamp(0, 1)
        diff = (rgb_render.detach() - torch.tensor(self.mesh_model.default_color).view(1, 3, 1, 1).to(
            self.device)).abs().sum(axis=1)
        uncolored_mask = (diff < 0.1).float().unsqueeze(0)
        rgb_render = rgb_render * (1 - uncolored_mask) + utils.color_with_shade([0.85, 0.85, 0.85], z_normals=z_normals,
                                                                                light_coef=0.3) * uncolored_mask
        outputs_with_median = self.mesh_model.render(theta=theta, phi=phi, radius=radius,
                                                     dims=(dim, dim), use_median=True,
                                                     render_cache=outputs['render_cache'])
        
        meta_output = self.mesh_model.render(theta=theta, phi=phi, radius=radius,
                                             background=torch.Tensor([0, 0, 0]).to(self.device),
                                             use_meta_texture=True, render_cache=outputs['render_cache'])
        pred_z_normals = meta_output['image'][:, :1].detach()
        rgb_render = rgb_render.permute(0, 2, 3, 1).contiguous().clamp(0, 1).detach()
        texture_rgb = outputs_with_median['texture_map'].permute(0, 2, 3, 1).contiguous().clamp(0, 1).detach()

        depth_render = outputs['depth'].permute(0, 2, 3, 1).contiguous().detach()

        return rgb_render, texture_rgb, depth_render, pred_z_normals
    
    def calculate_trimap(self, rgb_render_raw: torch.Tensor,
                         depth_render: torch.Tensor,
                         z_normals: torch.Tensor, z_normals_cache: torch.Tensor, edited_mask: torch.Tensor,
                         mask: torch.Tensor):
        diff = (rgb_render_raw.detach() - torch.tensor(self.mesh_model.default_color).view(1, 3, 1, 1).to(
            self.device)).abs().sum(axis=1)
        exact_generate_mask = (diff < 0.1).float().unsqueeze(0)

        # Extend mask
        generate_mask = torch.from_numpy(
            cv2.dilate(exact_generate_mask[0, 0].detach().cpu().numpy(), np.ones((31, 31), np.uint8))).to(
            exact_generate_mask.device).unsqueeze(0).unsqueeze(0)

        update_mask = generate_mask.clone()

        object_mask = torch.ones_like(update_mask)
        object_mask[depth_render == 0] = 0
        object_mask = torch.from_numpy(
            cv2.erode(object_mask[0, 0].detach().cpu().numpy(), np.ones((31, 31), np.uint8))).to(
            object_mask.device).unsqueeze(0).unsqueeze(0)

        #update mask 기반 refine_mask shape 일치하게
        refine_mask = torch.zeros_like(update_mask)

        # z_normal 부분이 cache + thr 보다 큰 부분만 refine_mask에 1로 채움)(차이가 큰부분)
        refine_mask[torch.abs(z_normals - z_normals_cache[:, :1, :, :]) > self.cfg.guide.z_update_thr] = 1
        # refine_mask[z_normals > z_normals_cache[:, :1, :, :] + self.cfg.guide.z_update_thr] = 1
        # initial texture이 없는 부분은 refine하지 않는다.
        if self.cfg.guide.initial_texture is None:
            refine_mask[z_normals_cache[:, :1, :, :] == 0] = 0
        elif self.cfg.guide.reference_texture is not None:
            # edited_mask 부분만 refinement 되도록
            refine_mask[edited_mask == 0] = 0
            refine_mask = torch.from_numpy(
                cv2.dilate(refine_mask[0, 0].detach().cpu().numpy(), np.ones((31, 31), np.uint8))).to(
                mask.device).unsqueeze(0).unsqueeze(0)
            refine_mask[mask == 0] = 0
            # z_normal이 작은 부분(bad angle)은 refinement하지 않는다
            refine_mask[z_normals < 0.4] = 0
        else:
            # Update all regions inside the object
            refine_mask[mask == 0] = 0

        refine_mask = torch.from_numpy(
            cv2.erode(refine_mask[0, 0].detach().cpu().numpy(), np.ones((5, 5), np.uint8))).to(
            mask.device).unsqueeze(0).unsqueeze(0)
        refine_mask = torch.from_numpy(
            cv2.dilate(refine_mask[0, 0].detach().cpu().numpy(), np.ones((5, 5), np.uint8))).to(
            mask.device).unsqueeze(0).unsqueeze(0)
        update_mask[refine_mask == 1] = 1

        update_mask[torch.bitwise_and(object_mask == 0, generate_mask == 0)] = 0

        # Visualize trimap
        if self.cfg.log.log_images:
            trimap_vis = utils.color_with_shade(color=[112 / 255.0, 173 / 255.0, 71 / 255.0], z_normals=z_normals)
            trimap_vis[mask.repeat(1, 3, 1, 1) == 0] = 1
            trimap_vis = trimap_vis * (1 - exact_generate_mask) + utils.color_with_shade(
                [255 / 255.0, 22 / 255.0, 67 / 255.0],
                z_normals=z_normals,
                light_coef=0.7) * exact_generate_mask

            shaded_rgb_vis = rgb_render_raw.detach()
            shaded_rgb_vis = shaded_rgb_vis * (1 - exact_generate_mask) + utils.color_with_shade([0.85, 0.85, 0.85],
                                                                                                 z_normals=z_normals,
                                                                                                 light_coef=0.7) * exact_generate_mask

            if self.paint_step > -1 or self.cfg.guide.initial_texture is not None:
                refinement_color_shaded = utils.color_with_shade(color=[91 / 255.0, 155 / 255.0, 213 / 255.0],
                                                                 z_normals=z_normals)
                only_old_mask_for_vis = torch.bitwise_and(refine_mask == 1, exact_generate_mask == 0).float().detach()
                trimap_vis = trimap_vis * 0 + 1.0 * (trimap_vis * (
                        1 - only_old_mask_for_vis) + refinement_color_shaded * only_old_mask_for_vis)
            self.log_train_image(shaded_rgb_vis, 'shaded_input')
            self.log_train_image(trimap_vis, 'trimap')

        return update_mask, generate_mask, refine_mask
    
    def calculate_trimap_ref(self, rgb_render_raw: torch.Tensor,
                            depth_render: torch.Tensor,
                            z_normals: torch.Tensor, z_normals_cache: torch.Tensor, edited_mask: torch.Tensor,
                            mask: torch.Tensor):
        # 기본 generate mask 생성 (default color 기준)
        diff = (rgb_render_raw.detach() - torch.tensor(self.mesh_model.default_color).view(1, 3, 1, 1).to(self.device)).abs().sum(axis=1)
        exact_generate_mask = (diff < 0.1).float().unsqueeze(0)

        generate_mask_z_norm = (torch.abs(z_normals - z_normals_cache[:, :1, :, :]) > self.cfg.guide.z_update_thr).float()
        combined_generate_mask = ((exact_generate_mask == 1) | (generate_mask_z_norm == 1)).float()

        # Extend mask
        generate_mask = torch.from_numpy(
            cv2.dilate(combined_generate_mask[0, 0].detach().cpu().numpy(), np.ones((15, 15), np.uint8))).to(
            combined_generate_mask.device).unsqueeze(0).unsqueeze(0)

        # object mask: depth = 0 인 부분 제외
        object_mask = torch.ones_like(generate_mask)
        object_mask[depth_render == 0] = 0
        object_mask = torch.from_numpy(
            cv2.erode(object_mask[0, 0].cpu().numpy(), np.ones((15, 15), np.uint8))
        ).to(object_mask.device).unsqueeze(0).unsqueeze(0)

        # 1. generate & keep mask 추출
        keep_mask = ((object_mask == 1) & (generate_mask == 0)).float()

        # 2. numpy 변환
        generate_mask_np = generate_mask[0, 0].cpu().numpy().astype(np.uint8)
        keep_mask_np = keep_mask[0, 0].cpu().numpy().astype(np.uint8)

        # 3. edge 추출
        gen_edge = cv2.Canny(generate_mask_np * 255, 10, 150)
        keep_edge = cv2.Canny(keep_mask_np * 255, 10, 150)

        # 4. 두 edge가 인접한 영역을 더해서 포함
        combined_edge = cv2.add(gen_edge, keep_edge)
        refine_zone = cv2.dilate(gen_edge, np.ones((31, 31), np.uint8))
        # refine_zone = cv2.dilate(combined_edge, np.ones((71, 71), np.uint8))

        # 5. torch mask로 변환
        refine_mask = torch.from_numpy(refine_zone > 0).to(generate_mask.device).unsqueeze(0).unsqueeze(0).float()
        refine_mask = refine_mask * object_mask

        # 최종 update_mask
        update_mask = ((generate_mask == 1) | (refine_mask == 1)).float()

        if self.cfg.log.log_images:
            trimap_vis = utils.color_with_shade(color=[112 / 255.0, 173 / 255.0, 71 / 255.0], z_normals=z_normals)
            trimap_vis[mask.repeat(1, 3, 1, 1) == 0] = 1
            trimap_vis = trimap_vis * (1 - combined_generate_mask) + utils.color_with_shade(
                [255 / 255.0, 22 / 255.0, 67 / 255.0],
                z_normals=z_normals,
                light_coef=0.7) * combined_generate_mask

            shaded_rgb_vis = rgb_render_raw.detach()
            shaded_rgb_vis = shaded_rgb_vis * (1 - combined_generate_mask) + utils.color_with_shade([0.85, 0.85, 0.85],
                                                                                                 z_normals=z_normals,
                                                                                                 light_coef=0.7) * combined_generate_mask

            if self.paint_step > -1 or self.cfg.guide.initial_texture is not None:
                refinement_color_shaded = utils.color_with_shade(color=[91 / 255.0, 155 / 255.0, 213 / 255.0],
                                                                 z_normals=z_normals)
                only_old_mask_for_vis = torch.bitwise_and(refine_mask == 1, combined_generate_mask == 0).float().detach()
                trimap_vis = trimap_vis * 0 + 1.0 * (trimap_vis * (
                        1 - only_old_mask_for_vis) + refinement_color_shaded * only_old_mask_for_vis)
            self.log_train_image(shaded_rgb_vis, 'shaded_input_init')
            self.log_train_image(trimap_vis, 'trimap_init')

        return update_mask, generate_mask, refine_mask
    
    def generate_checkerboard(self, update_mask_inner, improve_z_mask_inner, update_mask_base_inner):
        checkerboard = torch.ones((1, 1, 64 // 2, 64 // 2)).to(self.device)
        # Create a checkerboard grid
        checkerboard[:, :, ::2, ::2] = 0
        checkerboard[:, :, 1::2, 1::2] = 0
        checkerboard = F.interpolate(checkerboard,
                                     (1024, 1024))
        checker_mask = F.interpolate(update_mask_inner, (1024, 1024))
        only_old_mask = F.interpolate(torch.bitwise_and(improve_z_mask_inner == 1,
                                                        update_mask_base_inner == 0).float(), (1024, 1024))
        checker_mask[only_old_mask == 1] = checkerboard[only_old_mask == 1]
        return checker_mask
    
    def generate_mask(self, update_mask_inner, improve_z_mask_inner, update_mask_base_inner):
        mask = F.interpolate(update_mask_inner, (1024, 1024))
        only_old_mask = F.interpolate(torch.bitwise_and(improve_z_mask_inner == 1,
                                                        update_mask_base_inner == 0).float(), (1024, 1024))
        mask[only_old_mask == 1] = 1 
        return mask
    
    def project_back(self, render_cache: Dict[str, Any], background: Any, rgb_output: torch.Tensor,
                     object_mask: torch.Tensor, update_mask: torch.Tensor, z_normals: torch.Tensor,
                     z_normals_cache: torch.Tensor):
        object_mask = torch.from_numpy(
            cv2.erode(object_mask[0, 0].detach().cpu().numpy(), np.ones((5, 5), np.uint8))).to(
            object_mask.device).unsqueeze(0).unsqueeze(0)
        render_update_mask = object_mask.clone()
        render_update_mask[update_mask == 0] = 0

        blurred_render_update_mask = torch.from_numpy(
            cv2.dilate(render_update_mask[0, 0].detach().cpu().numpy(), np.ones((25, 25), np.uint8))).to(
            render_update_mask.device).unsqueeze(0).unsqueeze(0)

        # 전체 Gaussian blur
        blurred_render_update_mask = utils.gaussian_blur(blurred_render_update_mask, 51, 23)

        blurred_render_update_mask[object_mask == 0] = 0

        #strict constraint
        if self.cfg.guide.strict_projection:
            blurred_render_update_mask[blurred_render_update_mask < 0.5] = 0
            # Do not use bad normals
            z_was_better = z_normals + self.cfg.guide.z_update_thr < z_normals_cache[:, :1, :, :]
            blurred_render_update_mask[z_was_better] = 0
        #update
        render_update_mask = blurred_render_update_mask
        self.log_train_image(rgb_output * render_update_mask, 'project_back_input')

        # Update the normals (max value with two)
        z_normals_cache[:, 0, :, :] = torch.max(z_normals_cache[:, 0, :, :], z_normals[:, 0, :, :])
        # self.save_vu_image(z_normals_cache, 'z_normals_cache_updated')
        # self.save_vu_image(z_normals, 'z_normals')
        #Adam optimizer for updating model parameter
        optimizer = torch.optim.Adam(self.mesh_model.get_params(), lr=self.cfg.optim.lr, betas=(0.9, 0.99),
                                     eps=1e-15)
        #Optimize Mesh Colors 200 iteration
        for _ in tqdm(range(200), desc='fitting mesh colors'):
            optimizer.zero_grad()
            outputs = self.mesh_model.render(background=background,
                                             render_cache=render_cache)
            rgb_render = outputs['image']
            mask = render_update_mask.flatten()
            masked_pred = rgb_render.reshape(1, rgb_render.shape[1], -1)[:, :, mask > 0]
            masked_target = rgb_output.reshape(1, rgb_output.shape[1], -1)[:, :, mask > 0]
            masked_mask = mask[mask > 0]
            loss = ((masked_pred - masked_target.detach()).pow(2) * masked_mask).mean()

            meta_outputs = self.mesh_model.render(background=torch.Tensor([0, 0, 0]).to(self.device),
                                                  use_meta_texture=True, render_cache=render_cache)
            current_z_normals = meta_outputs['image']
            current_z_mask = meta_outputs['mask'].flatten()
            masked_current_z_normals = current_z_normals.reshape(1, current_z_normals.shape[1], -1)[:, :,
                                       current_z_mask == 1][:, :1]
            masked_last_z_normals = z_normals_cache.reshape(1, z_normals_cache.shape[1], -1)[:, :,
                                    current_z_mask == 1][:, :1]
            loss += (masked_current_z_normals - masked_last_z_normals.detach()).pow(2).mean()
            loss.backward()
            optimizer.step()

        return rgb_render, current_z_normals
    
    # def project_back_blend(self, render_cache: Dict[str, Any], background: Any, rgb_output: torch.Tensor,
    #              object_mask: torch.Tensor, update_mask: torch.Tensor, z_normals: torch.Tensor,
    #              z_normals_cache: torch.Tensor, refine_mask: torch.Tensor):
    #     object_mask = torch.from_numpy(
    #         cv2.erode(object_mask[0, 0].detach().cpu().numpy(), np.ones((5, 5), np.uint8))).to(
    #         object_mask.device).unsqueeze(0).unsqueeze(0)

    #     render_update_mask = object_mask.clone()
    #     render_update_mask[update_mask == 0] = 0

    #     blurred_render_update_mask = torch.from_numpy(
    #         cv2.dilate(render_update_mask[0, 0].detach().cpu().numpy(), np.ones((25, 25), np.uint8))).to(
    #         render_update_mask.device).unsqueeze(0).unsqueeze(0)
    #     blurred_render_update_mask = utils.gaussian_blur(blurred_render_update_mask, 51, 23)
    #     blurred_render_update_mask[object_mask == 0] = 0

    #     if self.cfg.guide.strict_projection:
    #         blurred_render_update_mask[blurred_render_update_mask < 0.5] = 0
    #         z_was_better = z_normals + self.cfg.guide.z_update_thr < z_normals_cache[:, :1, :, :]
    #         blurred_render_update_mask[z_was_better] = 0

    #     render_update_mask = blurred_render_update_mask
    #     self.log_train_image(rgb_output * render_update_mask, 'project_back_input')

    #     z_normals_cache[:, 0, :, :] = torch.max(z_normals_cache[:, 0, :, :], z_normals[:, 0, :, :])
    #     optimizer = torch.optim.Adam(self.mesh_model.get_params(), lr=self.cfg.optim.lr, betas=(0.9, 0.99), eps=1e-15)

    #     for _ in tqdm(range(200), desc='fitting mesh colors'):
    #         optimizer.zero_grad()
    #         outputs = self.mesh_model.render(background=background, render_cache=render_cache)
    #         rgb_render = outputs['image']

    #         mask = render_update_mask.flatten()
    #         masked_pred = rgb_render.reshape(1, rgb_render.shape[1], -1)[:, :, mask > 0]
    #         masked_target = rgb_output.reshape(1, rgb_output.shape[1], -1)[:, :, mask > 0]
    #         masked_mask = mask[mask > 0]

    #         loss = ((masked_pred - masked_target.detach()).pow(2) * masked_mask).mean()

    #         # Smooth loss for refine region
    #         if refine_mask is not None:
    #             blend_weight = 0.3
    #             refine_mask_exp = refine_mask.expand_as(rgb_output)
    #             blended_target = rgb_output.detach() * blend_weight + outputs['image'].detach() * (1 - blend_weight)

    #             refine_region = refine_mask_exp.bool()
    #             pred_refine = rgb_render[refine_region]
    #             target_refine = blended_target[refine_region]

    #             if pred_refine.numel() > 0:
    #                 smooth_loss = ((pred_refine - target_refine).pow(2)).mean()
    #                 loss += smooth_loss

    #         meta_outputs = self.mesh_model.render(background=torch.Tensor([0, 0, 0]).to(self.device),
    #                                             use_meta_texture=True, render_cache=render_cache)
    #         current_z_normals = meta_outputs['image']
    #         current_z_mask = meta_outputs['mask'].flatten()
    #         masked_current_z_normals = current_z_normals.reshape(1, current_z_normals.shape[1], -1)[:, :,
    #                                 current_z_mask == 1][:, :1]
    #         masked_last_z_normals = z_normals_cache.reshape(1, z_normals_cache.shape[1], -1)[:, :,
    #                                 current_z_mask == 1][:, :1]
    #         loss += (masked_current_z_normals - masked_last_z_normals.detach()).pow(2).mean()

    #         loss.backward()
    #         optimizer.step()

    #     return rgb_render, current_z_normals

    def project_back_ref(self, render_cache: Dict[str, Any], background: Any, rgb_output: torch.Tensor,
                 object_mask: torch.Tensor, update_mask: torch.Tensor, z_normals: torch.Tensor,
                 z_normals_cache: torch.Tensor, refine_mask: torch.Tensor):
        object_mask = torch.from_numpy(
            cv2.erode(object_mask[0, 0].detach().cpu().numpy(), np.ones((5, 5), np.uint8))).to(
            object_mask.device).unsqueeze(0).unsqueeze(0)

        render_update_mask = object_mask.clone()
        render_update_mask[update_mask == 0] = 0

        blurred_render_update_mask = torch.from_numpy(
            cv2.dilate(render_update_mask[0, 0].detach().cpu().numpy(), np.ones((25, 25), np.uint8))).to(
            render_update_mask.device).unsqueeze(0).unsqueeze(0)
        blurred_render_update_mask = utils.gaussian_blur(blurred_render_update_mask, 13, 7)
        blurred_render_update_mask[object_mask == 0] = 0

        if self.cfg.guide.strict_projection:
            blurred_render_update_mask[blurred_render_update_mask < 0.5] = 0
            z_was_better = z_normals + self.cfg.guide.z_update_thr < z_normals_cache[:, :1, :, :]
            blurred_render_update_mask[z_was_better] = 0

        render_update_mask = blurred_render_update_mask
        self.log_train_image(rgb_output * render_update_mask, 'project_back_input')

        z_normals_cache[:, 0, :, :] = torch.max(z_normals_cache[:, 0, :, :], z_normals[:, 0, :, :])
        optimizer = torch.optim.Adam(self.mesh_model.get_params(), lr=self.cfg.optim.lr, betas=(0.9, 0.99), eps=1e-15)

        for _ in tqdm(range(200), desc='fitting mesh colors'):
            optimizer.zero_grad()
            outputs = self.mesh_model.render(background=background, render_cache=render_cache)
            rgb_render = outputs['image']

            mask = render_update_mask.flatten()
            masked_pred = rgb_render.reshape(1, rgb_render.shape[1], -1)[:, :, mask > 0]
            masked_target = rgb_output.reshape(1, rgb_output.shape[1], -1)[:, :, mask > 0]
            masked_mask = mask[mask > 0]

            loss = ((masked_pred - masked_target.detach()).pow(2) * masked_mask).mean()

            # 추가 loss: refine 영역 외부의 gradient 보존 (keep mask 기준)
            if refine_mask is not None:
                refine_mask_exp = refine_mask.expand_as(rgb_render)
                keep_mask = (update_mask - refine_mask).clamp(min=0)
                keep_mask_exp = keep_mask.expand_as(rgb_render)

                grad_pred = rgb_render[:, :, 1:, :] - rgb_render[:, :, :-1, :]
                grad_keep = grad_pred * keep_mask[:, :, 1:, :]
                grad_loss = (grad_keep.pow(2)).mean()
                loss += grad_loss

            meta_outputs = self.mesh_model.render(background=torch.Tensor([0, 0, 0]).to(self.device),
                                                use_meta_texture=True, render_cache=render_cache)
            current_z_normals = meta_outputs['image']
            current_z_mask = meta_outputs['mask'].flatten()
            masked_current_z_normals = current_z_normals.reshape(1, current_z_normals.shape[1], -1)[:, :,
                                    current_z_mask == 1][:, :1]
            masked_last_z_normals = z_normals_cache.reshape(1, z_normals_cache.shape[1], -1)[:, :,
                                    current_z_mask == 1][:, :1]
            loss += (masked_current_z_normals - masked_last_z_normals.detach()).pow(2).mean()

            loss.backward()
            optimizer.step()

        return rgb_render, current_z_normals

    def log_train_image(self, tensor: torch.Tensor, name: str, colormap=False):
        if self.cfg.log.log_images:
            self.ncount += 1
            if colormap:
                tensor = cm.seismic(tensor.detach().cpu().numpy())[:, :, :3]
            else:
                tensor = einops.rearrange(tensor, '(1) c h w -> h w c').detach().cpu().numpy()
            Image.fromarray((tensor * 255).astype(np.uint8)).save(
                #self.train_renders_path / f'{self.paint_step:04d}_{name}.jpg')
                self.train_renders_path / f'{self.ncount:04d}_{self.paint_step:02d}_{name}.jpg')

    def log_diffusion_steps(self, intermediate_vis: List[Image.Image]):
        if len(intermediate_vis) > 0:
            step_folder = self.train_renders_path / f'{self.paint_step:04d}_diffusion_steps'
            step_folder.mkdir(exist_ok=True)
            for k, intermedia_res in enumerate(intermediate_vis):
                intermedia_res.save(
                    step_folder / f'{k:02d}_diffusion_step.jpg')

    def save_image(self, tensor: torch.Tensor, path: Path):
        if self.cfg.log.log_images:
            Image.fromarray(
                (einops.rearrange(tensor, '(1) c h w -> h w c').detach().cpu().numpy() * 255).astype(np.uint8)).save(
                path)
    
    def save_vu_image(self, tensor: torch.Tensor, name: str):
        self.ncount += 1
        # Save the image with the new naming format
        vutils.save_image(tensor, self.train_renders_path / f'{self.ncount:04d}_{self.paint_step:02d}_{name}.png')

    def save_uv_map(self, dataloader: DataLoader, save_path: Path, name: str = 'collapsed'):
        self.texturecount += 1
        logger.info(f'Saving UV maps to {save_path}')
        _, textures, _, _ = self.eval_render(next(iter(dataloader)))
        texture = tensor2numpy(textures[0])
        if name == 'collapsed':
            Image.fromarray(texture).save(save_path / f"step_{self.paint_step:02d}_{self.texturecount:03d}_collapsed_texture.png")
        elif name == 'initial':
            Image.fromarray(texture).save(save_path / f"step_{self.paint_step:02d}_{self.texturecount:03d}_initial_texture.png")
        else :
            Image.fromarray(texture).save(save_path / f"step_{self.paint_step:02d}_{self.texturecount:03d}_{name}_texture.png")
    

    def get_texture_mask_from_render(self, rgb_render: torch.Tensor, threshold: float = 0.05):
        default_color = torch.tensor(self.mesh_model.default_color).view(1, 3, 1, 1).to(rgb_render.device)
        diff = (rgb_render - default_color).abs().sum(dim=1, keepdim=True)  # [B,1,H,W]
        mask = (diff > threshold).float()  # 텍스처가 입혀진 영역: 1, 아닌 곳: 0
        return mask  # [B, 1, H, W]

