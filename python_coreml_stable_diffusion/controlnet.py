#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers import ModelMixin
import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet import Timesteps, TimestepEmbedding, get_down_block, UNetMidBlock2DCrossAttn, linear_to_conv2d_map

class ControlNetConditioningEmbedding(nn.Module):
    def __init__(self, conditioning_embedding_channels, conditioning_channels=3, block_out_channels=(16, 32, 96, 256)):
        super().__init__()
        self.conv_in = nn.Conv2d(conditioning_channels, block_out_channels[0], kernel_size=3, padding=1)
        self.blocks = nn.ModuleList()

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1))
            self.blocks.append(nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2))

        self.conv_out = nn.Conv2d(block_out_channels[-1], conditioning_embedding_channels, kernel_size=3, padding=1)

    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)

        embedding = self.conv_out(embedding)
        return embedding

class ControlNetModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, in_channels=4, flip_sin_to_cos=True, freq_shift=0, down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
                 only_cross_attention=False, block_out_channels=(320, 640, 1280, 1280), layers_per_block=2, downsample_padding=1, mid_block_scale_factor=1, act_fn="silu",
                 norm_num_groups=32, norm_eps=1e-5, cross_attention_dim=1280, transformer_layers_per_block=1, attention_head_dim=8, use_linear_projection=False,
                 upcast_attention=False, resnet_time_scale_shift="default", conditioning_embedding_out_channels=(16, 32, 96, 256), **kwargs):
        super().__init__()

        if len(block_out_channels) != len(down_block_types):
            raise ValueError(f"Must provide the same number of `block_out_channels` as `down_block_types`. `block_out_channels`: {block_out_channels}. `down_block_types`: {down_block_types}.")

        if isinstance(only_cross_attention, bool):
            only_cross_attention = [only_cross_attention] * len(down_block_types)

        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(down_block_types)

        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * len(down_block_types)

        self._register_load_state_dict_pre_hook(linear_to_conv2d_map)

        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=1)
        time_embed_dim = block_out_channels[0] * 4
        self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding(block_out_channels[0], time_embed_dim)

        self.controlnet_cond_embedding = ControlNetConditioningEmbedding(conditioning_embedding_channels=block_out_channels[0], block_out_channels=conditioning_embedding_out_channels)
        self.down_blocks = nn.ModuleList()
        self.controlnet_down_blocks = nn.ModuleList()

        output_channel = block_out_channels[0]
        self.controlnet_down_blocks.append(nn.Conv2d(output_channel, output_channel, kernel_size=1))

        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                transformer_layers_per_block=transformer_layers_per_block[i],
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim[i],
                downsample_padding=downsample_padding,
                add_downsample=not is_final_block,
            )
            self.down_blocks.append(down_block)

            for _ in range(layers_per_block):
                self.controlnet_down_blocks.append(nn.Conv2d(output_channel, output_channel, kernel_size=1))

            if not is_final_block:
                self.controlnet_down_blocks.append(nn.Conv2d(output_channel, output_channel, kernel_size=1))

        mid_block_channel = block_out_channels[-1]
        self.controlnet_mid_block = nn.Conv2d(mid_block_channel, mid_block_channel, kernel_size=1)
        self.mid_block = UNetMidBlock2DCrossAttn(
            in_channels=mid_block_channel,
            temb_channels=time_embed_dim,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            output_scale_factor=mid_block_scale_factor,
            resnet_time_scale_shift=resnet_time_scale_shift,
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attention_head_dim[-1],
            resnet_groups=norm_num_groups,
            use_linear_projection=use_linear_projection,
            upcast_attention=upcast_attention,
        )

    def get_num_residuals(self):
        num_res = 2 # initial sample + mid block
        for down_block in self.down_blocks:
            num_res += len(down_block.resnets)
            if hasattr(down_block, "downsamplers") and down_block.downsamplers is not None:
                num_res += len(down_block.downsamplers)
        return num_res

    def forward(self, sample, timestep, encoder_hidden_states, controlnet_cond):
        t_emb = self.time_proj(timestep)
        emb = self.time_embedding(t_emb)
        sample = self.conv_in(sample)
        controlnet_cond = self.controlnet_cond_embedding(controlnet_cond)
        sample += controlnet_cond

        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "attentions") and downsample_block.attentions is not None:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb, encoder_hidden_states=encoder_hidden_states)
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
            down_block_res_samples += res_samples

        if self.mid_block is not None:
            sample = self.mid_block(sample, emb, encoder_hidden_states=encoder_hidden_states)

        controlnet_down_block_res_samples = ()
        for down_block_res_sample, controlnet_block in zip(down_block_res_samples, self.controlnet_down_blocks):
            down_block_res_sample = controlnet_block(down_block_res_sample)
            controlnet_down_block_res_samples += (down_block_res_sample,)

        down_block_res_samples = controlnet_down_block_res_samples
        mid_block_res_sample = self.controlnet_mid_block(sample)
        return down_block_res_samples, mid_block_res_sample
