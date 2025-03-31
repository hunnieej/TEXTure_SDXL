#=========================Need to be modified========================#
# Class SDXL(nn.Module):
# def __init__
# def load_concept
# def get_text_embeds
# def img2img_step
# def produce_latent
# def decode_latent
# def encode_imgs
# def get_timesteps
# def prompt_to_img
# img2img_single_step
# def train_step
# def produce_latent
#====================================================================#

from diffusers import (AutoencoderKL, UNet2DConditionModel, ControlNetModel, 
                       StableDiffusionXLControlNetPipeline, StableDiffusionXLPipeline,
                       PNDMScheduler, LMSDiscreteScheduler, DDIMScheduler)
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from huggingface_hub import hf_hub_download
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
    logging,
    DPTFeatureExtractor, DPTForDepthEstimation,
)

######################################################################

# suppress partial model loading warning
from src import utils
from src.utils import seed_everything
logging.set_verbosity_error()

######################################################################

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from tqdm.auto import tqdm
import cv2
from PIL import Image
import matplotlib.pyplot as plt

class BlendedLatentDiffusionSDXL(StableDiffusionXLPipeline):
    def __init__(self, model_name, device="cuda"):
        # StableDiffusionXLPipeline의 모든 모델을 직접 로드
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        ).to(device)

        # 부모 클래스 초기화 (vae, unet, text_encoder 등의 파라미터 상속)
        super().__init__(
            vae=pipeline.vae,
            text_encoder=pipeline.text_encoder,
            text_encoder_2=pipeline.text_encoder_2,
            tokenizer=pipeline.tokenizer,
            tokenizer_2=pipeline.tokenizer_2,
            unet=pipeline.unet,
            scheduler=pipeline.scheduler,
            feature_extractor=pipeline.feature_extractor,
        )

        # ControlNet 추가
        self.controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-depth-sdxl-1.0",
            variant="fp16",
            use_safetensors=True,
            torch_dtype=torch.float16
        ).to(device)

    def parameters(self):
        """ StableDiffusionXLPipeline 내부 모듈들의 parameters() 호출 """
        params = []
        if hasattr(self, "unet"):
            params += list(self.unet.parameters())
        if hasattr(self, "vae"):
            params += list(self.vae.parameters())
        if hasattr(self, "text_encoder"):
            params += list(self.text_encoder.parameters())
        if hasattr(self, "text_encoder_2"):
            params += list(self.text_encoder_2.parameters())
        if hasattr(self, "controlnet"):
            params += list(self.controlnet.parameters())

        return params
    
    
    def img2img_step(self, text, inputs, depth_mask, guidance_scale=5.0, strength=0.5,
                     num_inference_steps=40, update_mask=None, latent_mode=False, check_mask=None,
                     fixed_seed=None, check_mask_iters=0.5, intermediate_vis=False,
                    all_mask=None, generate_mask=None):
        intermediate_results = []
        device="cuda"

        # print(depth_mask.shape) #[1 1 H W]
        # print(inputs.shape) #[1 3 H W]
        def sample(latents, depth_mask, controlnet_depth_mask, strength, num_inference_steps, update_mask=None, check_mask=None,
                   masked_latents=None, prompt_embeds=None, pooled_prompt_embeds=None, negative_prompt_embeds=None,
                   negative_pooled_prompt_embeds=None, all_mask=None, generate_mask=None,
                   blending_percentage=0.0, height=1024, width=1024, guidance_sclae=5.0):
            
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = self.scheduler.timesteps
            seed_everything(1)

            # Noise to target latent
            max_noise_timestep = int(len(timesteps) * blending_percentage)
            init_image = latents
            latents = self.scheduler.add_noise(
                latents, torch.randn_like(latents), self.scheduler.timesteps[max_noise_timestep].unsqueeze(0)
            )
            depth_mask = torch.cat([depth_mask] * 2) # [2 1 128 128]

            # 0. Default height and width to unet
            height = height or self.default_sample_size * self.vae_scale_factor
            width = width or self.default_sample_size * self.vae_scale_factor

            original_size = (height, width)
            target_size = (height, width)
            crops_coords_top_left = (0,0)
            
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim
            add_text_embeds = pooled_prompt_embeds
            add_time_ids = self._get_add_time_ids(
                original_size, crops_coords_top_left, target_size, dtype=prompt_embeds.dtype, text_encoder_projection_dim=text_encoder_projection_dim
            )

            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

            prompt_embeds = prompt_embeds.to(device)
            add_text_embeds = add_text_embeds.to(device)
            add_time_ids = add_time_ids.to(device).repeat(1, 1)

            num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

            with torch.autocast('cuda'):
                for i, t in tqdm(enumerate(timesteps[max_noise_timestep:])):
                    use_bld = i >= num_inference_steps * 0.0 and i <= num_inference_steps * 1.0

                    latent_model_input = torch.cat([latents] * 2) 
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    # predict the noise residual
                    added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                    cond_depth_mask = torch.cat([controlnet_depth_mask] * 3, dim=1)

                    controlnet_latent = self.controlnet(
                                latent_model_input,  # 현재 Latent 입력
                                t,  # 현재 timestep
                                encoder_hidden_states=prompt_embeds,  # Text Condition
                                controlnet_cond=cond_depth_mask,  # ControlNet에 Depth 정보를 입력
                                added_cond_kwargs=added_cond_kwargs,
                                return_dict=True  # NOTE : v2(250303) Dict 로 반환해서 down block, mid block 반환
                            )  # ControlNet이 변형 한 latent 출력

                    controlnet_scale = 0.5  
                    controlnet_down_features = [
                        feature * controlnet_scale for feature in controlnet_latent['down_block_res_samples']
                    ]
                    controlnet_mid_features = controlnet_latent['mid_block_res_sample'] * controlnet_scale

                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=None,
                        added_cond_kwargs=added_cond_kwargs,
                        down_block_additional_residuals=controlnet_down_features,  # ControlNet Downblock feature
                        mid_block_additional_residual=controlnet_mid_features,  # ControlNet Midblock feature
                        return_dict=False,
                    )[0]

                    # perform guidance
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    
                    extra_step_kwargs = self.prepare_extra_step_kwargs(None, 0.0)
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                    # BLD 시작
                    # Source latents와의 블렌딩
                    noise_source_latents = self.scheduler.add_noise(
                        init_image, torch.randn_like(latents), t.unsqueeze(0)
                    ) # [2, 4, 128, 128]

                    if use_bld and self.use_inpaint:
                        latents = latents * (generate_mask) + noise_source_latents * (1 - generate_mask)
            
            return latents

        controlnet_depth_mask = F.interpolate(depth_mask, size=(1024, 1024), mode='bicubic',
                                   align_corners=False)
        depth_mask = F.interpolate(depth_mask, size=(128, 128), mode='bicubic',
                                   align_corners=False)
        masked_latents = None
        if inputs is None:
            latents = None
        elif latent_mode:
            latents = inputs
        else:         
            #inputs_1024 = F.interpolate(inputs, (1024, 1024), mode='bilinear', align_corners=False)
            inputs_1024 = inputs.squeeze(0).permute(1, 2, 0).cpu().numpy()
            inputs_1024 = Image.fromarray((inputs_1024 * 255).clip(0, 255).astype(np.uint8))
            inputs_1024 = inputs_1024.resize((1024, 1024), Image.BILINEAR)
            inputs_1024 = np.array(inputs_1024)[:, :, :3]
            latents = self._image2latent(inputs_1024)

        if update_mask is not None:
            update_mask = F.interpolate(update_mask, (128, 128), mode='nearest')
        if check_mask is not None:
            check_mask = F.interpolate(check_mask, (128, 128), mode='nearest')
        if all_mask is not None:
            all_mask = F.interpolate(all_mask, (128, 128), mode='nearest')
        if generate_mask is not None:
            generate_mask = F.interpolate(generate_mask, (128, 128), mode='nearest')

        depth_mask = 2.0 * (depth_mask - depth_mask.min()) / (depth_mask.max() - depth_mask.min()) - 1.0
        controlnet_depth_mask = 2.0 * (controlnet_depth_mask - controlnet_depth_mask.min()) / (controlnet_depth_mask.max() - controlnet_depth_mask.min()) - 1.0

        # 3. Encode input prompt
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=text,
            prompt_2=None,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=None,
            negative_prompt_2=None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=None,
            negative_pooled_prompt_embeds=None,
            lora_scale=None,
        )

        with torch.no_grad():
            target_latents = sample(latents, depth_mask, controlnet_depth_mask, strength=strength, num_inference_steps=num_inference_steps,
                                    update_mask=update_mask, check_mask=check_mask, masked_latents=masked_latents, prompt_embeds=prompt_embeds,
                                    pooled_prompt_embeds=pooled_prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,
                                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds, all_mask=all_mask, generate_mask=generate_mask)
            self.upcast_vae()
            target_latents = target_latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
            target_rgb = self.vae.decode(target_latents/self.vae.config.scaling_factor, return_dict=False)[0]
            target_rgb = (target_rgb / 2 + 0.5).clamp(0, 1)

        if latent_mode:
            return target_rgb, target_latents
        else:
            return target_rgb, intermediate_results

    def decode_latents(self, latents):
        # latents = F.interpolate(latents, (64, 64), mode='bilinear', align_corners=False)
        # latents = 1 / 0.18215 * latents # SDv2
        # NOTE : Scaling Factor changed
        # SOURCE : https://github.com/huggingface/diffusers/issues/6923
        latents = 1 / 0.13025 * latents
        vae = self.vae.float()
        if latents.dtype == torch.float16:
            latents = latents.to(dtype=torch.float32)

        with torch.no_grad():
            imgs = vae.decode(latents).sample

        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    @torch.no_grad()
    def _image2latent(self, image):
        image = torch.from_numpy(image).float() / 127.5 - 1
        image = image.permute(2, 0, 1).unsqueeze(0).to("cuda")
        self.vae = self.vae.float()
        latents = self.vae.encode(image)["latent_dist"].mean
        self.vae = self.vae.half()
        latents = latents.half()
        latents = latents * self.vae.config.scaling_factor

        return latents