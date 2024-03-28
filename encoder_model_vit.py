import math

import timm
import torch
from typing import Union, List, Tuple
import torch.nn.modules.utils as nn_utils
import types


def NCHW_to_NHWC(input_tensor):
    return input_tensor.permute(0, 2, 3, 1)

def NHWC_to_NCHW(input_tensor):
    return input_tensor.permute(0, 3, 1, 2)

class feature_extractor_vit(torch.nn.Module):
    def __init__(self, choices=[3,5]):
        super().__init__()
        self.model = timm.create_model('vit_base_patch16_clip_224.laion2b', pretrained=True)
        self.model.eval()
        self.hook_handlers = []
        self.choices = choices
        self.p = self.model.patch_embed.patch_size[0]
        self.stride = self.model.patch_embed.proj.stride

        # Model parameters themselves do not require gradient
        for name, param in self.model.named_parameters():
                param.requires_grad = False

    def _get_hook(self, facet: str):
        """
        generate a hook method for a specific block and facet.
        """
        if facet in ['attn', 'token']:
            def _hook(model, input, output):
                self._feats.append(output)
            return _hook

        if facet == 'query':
            facet_idx = 0
        elif facet == 'key':
            facet_idx = 1
        elif facet == 'value':
            facet_idx = 2
        else:
            raise TypeError(f"{facet} is not a supported facet.")

        def _inner_hook(module, input, output):
            input_layer = input[0]
            B, N, C = input_layer.shape
            print(B, N, C, "SHAPES 1")
            print(input_layer[0,0,0])

            print(module.num_heads, "NUM HEADS")
            qkv = module.qkv(input_layer).reshape(B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
            self._feats.append(qkv[facet_idx]) #Bxhxtxd
        return _inner_hook

    def _get_hook_2(self, facet: str):
        """
        generate a hook method for a specific block and facet.
        """
        if facet in ['attn', 'token']:
            def _hook(model, input, output):
                self._feats.append(output)
            return _hook

        if facet == 'query':
            facet_idx = 0
        elif facet == 'key':
            facet_idx = 1
        elif facet == 'value':
            facet_idx = 2
        else:
            raise TypeError(f"{facet} is not a supported facet.")

        def _inner_hook(module, input_value, output):
            input_layer = input_value[0]
            B, N, C = input_layer.shape
            # print(B, N, C, "SHAPES 2 ")
            # print(input_layer[0,0,0])
            num_heads = 12
            qkv = output.reshape(B, N, 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)
            self._feats.append(qkv[facet_idx][:,:,1:,:].permute(0, 2, 3, 1).flatten(start_dim=-2, end_dim=-1)) #Bxhxtxd
        return _inner_hook

    def _register_hooks(self, layers: List[int], facet: str) -> None:
        """
        register hook to extract features.
        :param layers: layers from which to extract features.
        :param facet: facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token' | 'attn']
        """
        for block_idx, block in enumerate(self.model.blocks):
            if block_idx in layers:
                if facet == 'token':
                    raise ValueError
                    self.hook_handlers.append(block.register_forward_hook(self._get_hook(facet)))
                elif facet == 'attn':
                    raise ValueError
                    self.hook_handlers.append(block.attn.attn_drop.register_forward_hook(self._get_hook(facet)))
                elif facet in ['key', 'query', 'value']:
                    # self.hook_handlers.append(block.attn.register_forward_hook(self._get_hook(facet))); print("mode 1")
                    self.hook_handlers.append(block.attn.qkv.register_forward_hook(self._get_hook_2(facet)))
                else:
                    raise TypeError(f"{facet} is not a supported facet.")

    def _unregister_hooks(self) -> None:
        """
        unregisters the hooks. should be called after feature extraction.
        """
        for handle in self.hook_handlers:
            handle.remove()
        self.hook_handlers = []

    # def _extract_features(self, batch: torch.Tensor, layers: List[int] = [3,5,9], facet: str = 'key') -> List[torch.Tensor]:
    def _extract_features(self, batch: torch.Tensor, layers: List[int] = [3, 5], facet: str = 'key') -> List[torch.Tensor]:
        """
        extract features from the model
        :param batch: batch to extract features for. Has shape BxCxHxW.
        :param layers: layer to extract. A number between 0 to 11.
        :param facet: facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token' | 'attn']
        :return : tensor of features.
                  if facet is 'key' | 'query' | 'value' has shape Bxhxtxd
                  if facet is 'attn' has shape Bxhxtxt
                  if facet is 'token' has shape Bxtxd
        """
        B, C, H, W = batch.shape
        try:
            del self._feats
            # Memory leak?
        except:
            pass
        self._feats = []
        self._register_hooks(layers, facet)
        last_layer_output = self.model(batch)
        self._unregister_hooks()
        self.load_size = (H, W)
        self.num_patches = (1 + (H - self.p) // self.stride[0], 1 + (W - self.p) // self.stride[1])
        return self._feats, last_layer_output

    def forward(self, input_image):
        extracted_features = self._extract_features(input_image, layers=self.choices)
        return extracted_features[0], extracted_features[1]


class downproject_split(torch.nn.Module):
    def __init__(self, num_early_output=None, num_higher_output=None):
        super().__init__()

        ### CLIP conv layers to early visual ###
        self.reduce_1 = torch.nn.Linear(768, 160)
        self.reduce_2 = torch.nn.Linear(768, 160)
        self.fuse1 = torch.nn.Linear(160*14*14, 1024)

        self.reduce_3 = torch.nn.Linear(768, 160)
        self.reduce_4 = torch.nn.Linear(768, 160)
        self.fuse2 = torch.nn.Linear(160*14*14, 1024)

        self.squeeze = torch.nn.Sequential(torch.nn.Linear(2048, 128), torch.nn.GELU(approximate="tanh"), torch.nn.Linear(128, 2048), torch.nn.Sigmoid())
        # Identified by Aria in her "Incorporating natural language into vision models improves prediction and understanding of higher visual cortex" paper
        ### End CLIP conv layers to early visual ###

        ### CLIP later layers to non-early visual ###
        self.reduce_5 = torch.nn.Linear(768, 160)
        self.reduce_6 = torch.nn.Linear(768, 160)
        self.fuse3 = torch.nn.Linear(160*14*14, 1024)
        self.fuse3.weight.data = self.fuse3.weight.data * 0.05
        self.fuse3.bias.data = self.fuse3.bias.data * 0.05

        self.reduce_7 = torch.nn.Linear(512, 1024)
        torch.nn.init.eye_(self.reduce_7.weight.data)
        torch.nn.init.orthogonal_(self.reduce_7.weight.data[512:])
        self.reduce_7.weight.data = self.reduce_7.weight.data * 0.5

        self.squeeze_2 = torch.nn.Sequential(torch.nn.Linear(2048, 128), torch.nn.GELU(approximate="tanh"), torch.nn.Linear(128, 2048), torch.nn.Sigmoid())
        self.squeeze_2[-2].bias.data = self.squeeze_2[-2].bias.data*0.1
        self.squeeze_2[-2].weight.data = self.squeeze_2[-2].weight.data*0.1
        ### End CLIP later layers to non-early visual ###

        keys = sorted(num_early_output.keys())
        for k_i in keys:
            self.add_module("final_{}_early".format(k_i), torch.nn.Linear(2048, num_early_output[k_i]))
            getattr(self, "final_{}_early".format(k_i)).weight.data = getattr(self, "final_{}_early".format(k_i)).weight.data*0.5
            # torch.nn.init.kaiming_uniform_(getattr(self, "final_{}_early".format(k_i)).weight.data, a=math.sqrt(5), mode='fan_out')
            self.add_module("final_{}_higher".format(k_i), torch.nn.Linear(2048, num_higher_output[k_i]))
            getattr(self, "final_{}_higher".format(k_i)).weight.data = getattr(self, "final_{}_higher".format(k_i)).weight.data*0.5

        # non linearity helps
        self.act1 = torch.nn.GELU(approximate="tanh")
        self.act2 = torch.nn.GELU(approximate="tanh")
        self.act3 = torch.nn.GELU(approximate="tanh")
    def forward(self, first_layer_in, second_layer_in, third_layer_in, last_layer_in, key_order):
        drop_p = 0.2
        # in shape torch.Size([2, 196, 768])
        # permute 1 = torch.Size([2, 768, 196])
        # permute 2 = torch.Size([2, 768, 196])

        first_layer = torch.nn.functional.dropout1d(torch.permute(first_layer_in, (0,2,1)), p=0.2,training=self.training).permute(0,2,1)
        second_layer = torch.nn.functional.dropout1d(torch.permute(second_layer_in, (0,2,1)), p=0.2,training=self.training).permute(0,2,1)
        third_layer = torch.nn.functional.dropout1d(torch.permute(third_layer_in, (0,2,1)), p=0.2,training=self.training).permute(0,2,1)

        flipped_first_ = self.fuse1(torch.flatten(self.act1(self.reduce_1(first_layer)) + self.reduce_2(first_layer), start_dim=1))
        flipped_second_ = self.fuse2(torch.flatten(self.act2(self.reduce_3(second_layer)) + self.reduce_4(second_layer), start_dim=1))

        early_out = torch.nn.functional.dropout(torch.cat((flipped_first_, flipped_second_), dim=1), p=0.2, training=self.training)
        early_final = torch.nn.functional.dropout(early_out * (1.0 + self.squeeze(early_out)), p=0.2, training=self.training)

        flipped_third_ = self.fuse3(torch.flatten(self.act3(self.reduce_5(third_layer)) + self.reduce_6(third_layer), start_dim=1))
        clip_last_proj = self.reduce_7(last_layer_in)
        higher_out = torch.nn.functional.dropout(torch.cat((clip_last_proj, flipped_third_), dim=1), p=0.2, training=self.training)
        higher_final = torch.nn.functional.dropout(higher_out * (1.0 + self.squeeze_2(higher_out)), p=0.2, training=self.training)
        key_i = key_order[0]
        # print(torch.cat((getattr(self, "final_{}_early".format(str(key_i)))(early_final), getattr(self, "final_{}_higher".format(str(key_i)))(higher_final)), dim=1).shape, "CAT SHAPE")
        # exit()
        return torch.cat((getattr(self, "final_{}_early".format(str(key_i)))(early_final), getattr(self, "final_{}_higher".format(str(key_i)))(higher_final)), dim=1)
        # print(.shape, "EARLY SHAPE")
        # print(getattr(self, "final_{}_higher".format(str(key_i)))(higher_final), "LATE SHAPE")
        # exit()
        # torch.cat((getattr(self, "final_{}_early".format(str(key_i)))(early_final),
        #            getattr(self, "final_{}_higher".format(str(key_i)))(higher_final)), dim=0)
        results = []

        # We collapse along the batch dimension, can enable training of multiple subjects with different number of voxels
        # Suppose we have [subj1, subj1] with 1000 voxels
        # We will return a vector of shape 2000
        for count, key_i in enumerate(key_order):
            results.append(torch.cat((getattr(self, "final_{}_early".format(str(key_i)))(early_final[count]), getattr(self, "final_{}_higher".format(str(key_i)))(higher_final[count])), dim=0))
            # print(results[-1].shape)
        return torch.cat(results, dim=0)


class downproject_CLIP(torch.nn.Module):
    def __init__(self, num_early_output=None, num_higher_output=None):
        super().__init__()

        ### CLIP conv layers to early visual ###
        self.reduce_1 = torch.nn.Linear(768, 160)
        self.reduce_2 = torch.nn.Linear(768, 160)
        self.fuse1 = torch.nn.Linear(160*14*14, 1024)

        self.reduce_3 = torch.nn.Linear(768, 160)
        self.reduce_4 = torch.nn.Linear(768, 160)
        self.fuse2 = torch.nn.Linear(160*14*14, 1024)

        self.squeeze = torch.nn.Sequential(torch.nn.Linear(2048, 128), torch.nn.GELU(approximate="tanh"), torch.nn.Linear(128, 2048), torch.nn.Sigmoid())
        # Identified by Aria in her "Incorporating natural language into vision models improves prediction and understanding of higher visual cortex" paper
        ### End CLIP conv layers to early visual ###

        ### CLIP later layers to non-early visual ###
        # self.reduce_5 = torch.nn.Linear(768, 160)
        # self.reduce_6 = torch.nn.Linear(768, 160)
        # self.fuse3 = torch.nn.Linear(160*14*14, 1024)
        # self.fuse3.weight.data = self.fuse3.weight.data * 0.05
        # self.fuse3.bias.data = self.fuse3.bias.data * 0.05

        self.reduce_5 = torch.nn.Linear(512, 2048)
        torch.nn.init.eye_(self.reduce_5.weight.data)
        torch.nn.init.orthogonal_(self.reduce_5.weight.data[512:])
        self.reduce_5.weight.data[512:] = self.reduce_5.weight.data[512:] * 0.5
        # NaN during training otherwise :(

        self.squeeze_2 = torch.nn.Sequential(torch.nn.Linear(2048, 512), torch.nn.GELU(approximate="tanh"),torch.nn.Linear(512, 512), torch.nn.GELU(approximate="tanh"), torch.nn.Linear(512, 2048), torch.nn.Sigmoid())
        self.squeeze_2[0].bias.data = self.squeeze_2[0].bias.data*0.25
        self.squeeze_2[0].weight.data = self.squeeze_2[0].weight.data*0.25

        self.squeeze_2[2].bias.data = self.squeeze_2[2].bias.data*0.25
        self.squeeze_2[2].weight.data = self.squeeze_2[2].weight.data*0.25

        self.squeeze_2[4].bias.data = self.squeeze_2[4].bias.data*0.25
        self.squeeze_2[4].weight.data = self.squeeze_2[4].weight.data*0.25


        ### End CLIP later layers to non-early visual ###

        keys = sorted(num_early_output.keys())
        for k_i in keys:
            self.add_module("final_{}_early".format(k_i), torch.nn.Linear(2048, num_early_output[k_i]))
            getattr(self, "final_{}_early".format(k_i)).weight.data = getattr(self, "final_{}_early".format(k_i)).weight.data*0.5
            # torch.nn.init.kaiming_uniform_(getattr(self, "final_{}_early".format(k_i)).weight.data, a=math.sqrt(5), mode='fan_out')
            self.add_module("final_{}_higher".format(k_i), torch.nn.Linear(2048, num_higher_output[k_i]))
            getattr(self, "final_{}_higher".format(k_i)).weight.data = getattr(self, "final_{}_higher".format(k_i)).weight.data*0.5

        # non linearity helps
        self.act1 = torch.nn.GELU(approximate="tanh")
        self.act2 = torch.nn.GELU(approximate="tanh")
        self.act3 = torch.nn.GELU(approximate="tanh")
    def forward(self, first_layer_in, second_layer_in, last_layer_in, key_order):
        drop_p = 0.2
        # in shape torch.Size([2, 196, 768])
        # permute 1 = torch.Size([2, 768, 196])
        # permute 2 = torch.Size([2, 768, 196])

        first_layer = torch.nn.functional.dropout1d(torch.permute(first_layer_in, (0,2,1)), p=0.2,training=self.training).permute(0,2,1)
        second_layer = torch.nn.functional.dropout1d(torch.permute(second_layer_in, (0,2,1)), p=0.2,training=self.training).permute(0,2,1)

        flipped_first_ = self.fuse1(torch.flatten(self.act1(self.reduce_1(first_layer)) + self.reduce_2(first_layer), start_dim=1))
        flipped_second_ = self.fuse2(torch.flatten(self.act2(self.reduce_3(second_layer)) + self.reduce_4(second_layer), start_dim=1))

        early_out = torch.nn.functional.dropout(torch.cat((flipped_first_, flipped_second_), dim=1), p=0.2, training=self.training)
        early_final = torch.nn.functional.dropout(early_out * (1.0 + self.squeeze(early_out)), p=0.2, training=self.training)

        higher_out = torch.nn.functional.dropout(self.reduce_5(last_layer_in), p=0.1, training=self.training)
        higher_final = torch.nn.functional.dropout(higher_out * (1.0 + self.squeeze_2(higher_out)), p=0.1, training=self.training)
        key_i = key_order[0]
        return torch.cat((getattr(self, "final_{}_early".format(str(key_i)))(early_final), getattr(self, "final_{}_higher".format(str(key_i)))(higher_final)), dim=1)
        # print(.shape, "EARLY SHAPE")
        # print(getattr(self, "final_{}_higher".format(str(key_i)))(higher_final), "LATE SHAPE")
        # exit()
        # torch.cat((getattr(self, "final_{}_early".format(str(key_i)))(early_final),
        #            getattr(self, "final_{}_higher".format(str(key_i)))(higher_final)), dim=0)
        results = []

        # We collapse along the batch dimension, can enable training of multiple subjects with different number of voxels
        # Suppose we have [subj1, subj1] with 1000 voxels
        # We will return a vector of shape 2000
        for count, key_i in enumerate(key_order):
            results.append(torch.cat((getattr(self, "final_{}_early".format(str(key_i)))(early_final[count]), getattr(self, "final_{}_higher".format(str(key_i)))(higher_final[count])), dim=0))
            # print(results[-1].shape)
        return torch.cat(results, dim=0)


class downproject_CLIP_split(torch.nn.Module):
    def __init__(self, num_early_output=None, num_higher_output=None):
        super().__init__()

        ### CLIP conv layers to early visual ###
        self.reduce_1 = torch.nn.Linear(768, 160)
        self.reduce_2 = torch.nn.Linear(768, 160)
        self.fuse1 = torch.nn.Linear(160*14*14, 1024)

        self.reduce_3 = torch.nn.Linear(768, 160)
        self.reduce_4 = torch.nn.Linear(768, 160)
        self.fuse2 = torch.nn.Linear(160*14*14, 1024)

        self.squeeze = torch.nn.Sequential(torch.nn.Linear(2048, 128), torch.nn.GELU(approximate="tanh"), torch.nn.Linear(128, 2048), torch.nn.Sigmoid())
        # Identified by Aria in her "Incorporating natural language into vision models improves prediction and understanding of higher visual cortex" paper
        ### End CLIP conv layers to early visual ###

        ### CLIP later layers to non-early visual ###
        # self.reduce_5 = torch.nn.Linear(768, 160)
        # self.reduce_6 = torch.nn.Linear(768, 160)
        # self.fuse3 = torch.nn.Linear(160*14*14, 1024)
        # self.fuse3.weight.data = self.fuse3.weight.data * 0.05
        # self.fuse3.bias.data = self.fuse3.bias.data * 0.05

        self.reduce_5 = torch.nn.Linear(512, 2048-512)
        torch.nn.init.eye_(self.reduce_5.weight.data)*0.25
        torch.nn.init.orthogonal_(self.reduce_5.weight.data[512:])*0.25
        # NaN during training otherwise :(

        self.squeeze_2 = torch.nn.Sequential(torch.nn.Linear(2048-512, 128), torch.nn.GELU(approximate="tanh"), torch.nn.Linear(128, 2048-512), torch.nn.Sigmoid())
        self.squeeze_2[0].bias.data = self.squeeze_2[0].bias.data*0.2
        self.squeeze_2[0].weight.data = self.squeeze_2[0].weight.data*0.2

        self.squeeze_2[2].bias.data = self.squeeze_2[2].bias.data*0.2
        self.squeeze_2[2].weight.data = self.squeeze_2[2].weight.data*0.2

        ### End CLIP later layers to non-early visual ###

        keys = sorted(num_early_output.keys())
        for k_i in keys:
            self.add_module("final_{}_early".format(k_i), torch.nn.Linear(2048, num_early_output[k_i]))
            getattr(self, "final_{}_early".format(k_i)).weight.data = getattr(self, "final_{}_early".format(k_i)).weight.data*0.5
            # torch.nn.init.kaiming_uniform_(getattr(self, "final_{}_early".format(k_i)).weight.data, a=math.sqrt(5), mode='fan_out')
            self.add_module("final_{}_higher".format(k_i), torch.nn.Linear(2048, num_higher_output[k_i]))
            getattr(self, "final_{}_higher".format(k_i)).weight.data = getattr(self, "final_{}_higher".format(k_i)).weight.data*0.5

        # non linearity helps
        self.act1 = torch.nn.GELU(approximate="tanh")
        self.act2 = torch.nn.GELU(approximate="tanh")
        self.act3 = torch.nn.GELU(approximate="tanh")
    def forward(self, first_layer_in, second_layer_in, last_layer_in, key_order):
        drop_p = 0.2
        # in shape torch.Size([2, 196, 768])
        # permute 1 = torch.Size([2, 768, 196])
        # permute 2 = torch.Size([2, 768, 196])

        first_layer = torch.nn.functional.dropout1d(torch.permute(first_layer_in, (0,2,1)), p=0.2,training=self.training).permute(0,2,1)
        second_layer = torch.nn.functional.dropout1d(torch.permute(second_layer_in, (0,2,1)), p=0.2,training=self.training).permute(0,2,1)

        flipped_first_ = self.fuse1(torch.flatten(self.act1(self.reduce_1(first_layer)) + self.reduce_2(first_layer), start_dim=1))
        flipped_second_ = self.fuse2(torch.flatten(self.act2(self.reduce_3(second_layer)) + self.reduce_4(second_layer), start_dim=1))

        early_out = torch.nn.functional.dropout(torch.cat((flipped_first_, flipped_second_), dim=1), p=0.2, training=self.training)
        early_final = torch.nn.functional.dropout(early_out * (1.0 + self.squeeze(early_out)), p=0.2, training=self.training)

        last_layer_in_norm = last_layer_in/torch.norm(last_layer_in, dim=1, keepdim=True)
        # higher_out = self.reduce_5(last_layer_in)
        higher_out = self.reduce_5(last_layer_in_norm*12.0)
        higher_final = torch.nn.functional.dropout(torch.cat((higher_out * (1.0 + self.squeeze_2(higher_out)*0.0), last_layer_in_norm), dim=1), p=0.1, training=self.training)
        key_i = key_order[0]
        return torch.cat((getattr(self, "final_{}_early".format(str(key_i)))(early_final), getattr(self, "final_{}_higher".format(str(key_i)))(higher_final)), dim=1)
        # print(.shape, "EARLY SHAPE")
        # print(getattr(self, "final_{}_higher".format(str(key_i)))(higher_final), "LATE SHAPE")
        # exit()
        # torch.cat((getattr(self, "final_{}_early".format(str(key_i)))(early_final),
        #            getattr(self, "final_{}_higher".format(str(key_i)))(higher_final)), dim=0)
        results = []

        # We collapse along the batch dimension, can enable training of multiple subjects with different number of voxels
        # Suppose we have [subj1, subj1] with 1000 voxels
        # We will return a vector of shape 2000
        for count, key_i in enumerate(key_order):
            results.append(torch.cat((getattr(self, "final_{}_early".format(str(key_i)))(early_final[count]), getattr(self, "final_{}_higher".format(str(key_i)))(higher_final[count])), dim=0))
            # print(results[-1].shape)
        return torch.cat(results, dim=0)


# class downproject_CLIP_split_linear(torch.nn.Module):
#     def __init__(self, num_early_output=None, num_higher_output=None):
#         super().__init__()
#
#         ### CLIP conv layers to early visual ###
#         self.reduce_1 = torch.nn.Linear(768, 160)
#         self.reduce_2 = torch.nn.Linear(768, 160)
#         self.fuse1 = torch.nn.Linear(160*14*14, 1024)
#
#         self.reduce_3 = torch.nn.Linear(768, 160)
#         self.reduce_4 = torch.nn.Linear(768, 160)
#         self.fuse2 = torch.nn.Linear(160*14*14, 1024)
#
#         self.squeeze = torch.nn.Sequential(torch.nn.Linear(2048, 128), torch.nn.GELU(approximate="tanh"), torch.nn.Linear(128, 2048), torch.nn.Sigmoid())
#         # Identified by Aria in her "Incorporating natural language into vision models improves prediction and understanding of higher visual cortex" paper
#         ### End CLIP conv layers to early visual ###
#
#         ### CLIP later layers to non-early visual ###
#         # self.reduce_5 = torch.nn.Linear(768, 160)
#         # self.reduce_6 = torch.nn.Linear(768, 160)
#         # self.fuse3 = torch.nn.Linear(160*14*14, 1024)
#         # self.fuse3.weight.data = self.fuse3.weight.data * 0.05
#         # self.fuse3.bias.data = self.fuse3.bias.data * 0.05
#         ### End CLIP later layers to non-early visual ###
#
#         keys = sorted(num_early_output.keys())
#         for k_i in keys:
#             self.add_module("final_{}_early".format(k_i), torch.nn.Linear(2048, num_early_output[k_i]))
#             getattr(self, "final_{}_early".format(k_i)).weight.data = getattr(self, "final_{}_early".format(k_i)).weight.data*0.5
#             # torch.nn.init.kaiming_uniform_(getattr(self, "final_{}_early".format(k_i)).weight.data, a=math.sqrt(5), mode='fan_out')
#             self.add_module("final_{}_higher".format(k_i), torch.nn.Linear(512, num_higher_output[k_i]))
#             getattr(self, "final_{}_higher".format(k_i)).weight.data = getattr(self, "final_{}_higher".format(k_i)).weight.data*0.5
#
#         # non linearity helps
#         self.act1 = torch.nn.GELU(approximate="tanh")
#         self.act2 = torch.nn.GELU(approximate="tanh")
#     def forward(self, first_layer_in, second_layer_in, last_layer_in, key_order):
#         drop_p = 0.15
#         # in shape torch.Size([2, 196, 768])
#         # permute 1 = torch.Size([2, 768, 196])
#         # permute 2 = torch.Size([2, 768, 196])
#
#         first_layer = torch.nn.functional.dropout1d(torch.permute(first_layer_in, (0,2,1)), p=drop_p,training=self.training).permute(0,2,1)
#         second_layer = torch.nn.functional.dropout1d(torch.permute(second_layer_in, (0,2,1)), p=drop_p,training=self.training).permute(0,2,1)
#         flipped_first_ = self.fuse1(torch.flatten(self.act1(self.reduce_1(first_layer)) + self.reduce_2(first_layer), start_dim=1))
#         flipped_second_ = self.fuse2(torch.flatten(self.act2(self.reduce_3(second_layer)) + self.reduce_4(second_layer), start_dim=1))
#         early_out = torch.nn.functional.dropout(torch.cat((flipped_first_, flipped_second_), dim=1), p=drop_p, training=self.training)
#         early_final = torch.nn.functional.dropout(early_out * (1.0 + self.squeeze(early_out)), p=drop_p, training=self.training)
#
#         last_layer_in_norm = torch.nn.functional.normalize(last_layer_in, dim=1)
#         # higher_out = self.reduce_5(last_layer_in)
#         key_i = key_order[0]
#         return torch.cat((getattr(self, "final_{}_early".format(str(key_i)))(early_final), getattr(self, "final_{}_higher".format(str(key_i)))(last_layer_in_norm)), dim=1)
#
#     def forward_higher(self, first_layer_in, second_layer_in, last_layer_in, key_order):
#         key_i = key_order[0]
#         last_layer_in_norm = torch.nn.functional.normalize(last_layer_in, dim=1)
#         return getattr(self, "final_{}_higher".format(str(key_i)))(last_layer_in_norm)
#
#     def forward_early(self, first_layer_in, second_layer_in, last_layer_in, key_order):
#         drop_p = 0.15
#         assert not self.training
#         first_layer = torch.nn.functional.dropout1d(torch.permute(first_layer_in, (0,2,1)), p=drop_p,training=self.training).permute(0,2,1)
#         second_layer = torch.nn.functional.dropout1d(torch.permute(second_layer_in, (0,2,1)), p=drop_p,training=self.training).permute(0,2,1)
#         flipped_first_ = self.fuse1(torch.flatten(self.act1(self.reduce_1(first_layer)) + self.reduce_2(first_layer), start_dim=1))
#         flipped_second_ = self.fuse2(torch.flatten(self.act2(self.reduce_3(second_layer)) + self.reduce_4(second_layer), start_dim=1))
#         early_out = torch.nn.functional.dropout(torch.cat((flipped_first_, flipped_second_), dim=1), p=drop_p, training=self.training)
#         early_final = torch.nn.functional.dropout(early_out * (1.0 + self.squeeze(early_out)), p=drop_p, training=self.training)
#         key_i = key_order[0]
#         return getattr(self, "final_{}_early".format(str(key_i)))(early_final)
#


class downproject_CLIP_split_linear(torch.nn.Module):
    def __init__(self, num_early_output=None, num_higher_output=None):
        super().__init__()
        keys = sorted(num_early_output.keys())
        for k_i in keys:
            self.add_module("final_{}_higher".format(k_i), torch.nn.Linear(512, num_higher_output[k_i]))
            getattr(self, "final_{}_higher".format(k_i)).weight.data = getattr(self, "final_{}_higher".format(k_i)).weight.data*0.5

        # non linearity helps
    def forward(self, first_layer_in, second_layer_in, last_layer_in, key_order):
        drop_p = 0.15
        # in shape torch.Size([2, 196, 768])
        # permute 1 = torch.Size([2, 768, 196])
        # permute 2 = torch.Size([2, 768, 196])

        first_layer = torch.nn.functional.dropout1d(torch.permute(first_layer_in, (0,2,1)), p=drop_p,training=self.training).permute(0,2,1)
        second_layer = torch.nn.functional.dropout1d(torch.permute(second_layer_in, (0,2,1)), p=drop_p,training=self.training).permute(0,2,1)
        flipped_first_ = self.fuse1(torch.flatten(self.act1(self.reduce_1(first_layer)) + self.reduce_2(first_layer), start_dim=1))
        flipped_second_ = self.fuse2(torch.flatten(self.act2(self.reduce_3(second_layer)) + self.reduce_4(second_layer), start_dim=1))
        early_out = torch.nn.functional.dropout(torch.cat((flipped_first_, flipped_second_), dim=1), p=drop_p, training=self.training)
        early_final = torch.nn.functional.dropout(early_out * (1.0 + self.squeeze(early_out)), p=drop_p, training=self.training)

        last_layer_in_norm = torch.nn.functional.normalize(last_layer_in, dim=1)
        # higher_out = self.reduce_5(last_layer_in)
        key_i = key_order[0]
        return torch.cat((getattr(self, "final_{}_early".format(str(key_i)))(early_final), getattr(self, "final_{}_higher".format(str(key_i)))(last_layer_in_norm)), dim=1)

    def forward_higher(self, first_layer_in, second_layer_in, last_layer_in, key_order):
        key_i = key_order[0]
        last_layer_in_norm = torch.nn.functional.normalize(last_layer_in, dim=1)
        return getattr(self, "final_{}_higher".format(str(key_i)))(last_layer_in_norm)

    def forward_early(self, first_layer_in, second_layer_in, last_layer_in, key_order):
        drop_p = 0.15
        assert not self.training
        first_layer = torch.nn.functional.dropout1d(torch.permute(first_layer_in, (0,2,1)), p=drop_p,training=self.training).permute(0,2,1)
        second_layer = torch.nn.functional.dropout1d(torch.permute(second_layer_in, (0,2,1)), p=drop_p,training=self.training).permute(0,2,1)
        flipped_first_ = self.fuse1(torch.flatten(self.act1(self.reduce_1(first_layer)) + self.reduce_2(first_layer), start_dim=1))
        flipped_second_ = self.fuse2(torch.flatten(self.act2(self.reduce_3(second_layer)) + self.reduce_4(second_layer), start_dim=1))
        early_out = torch.nn.functional.dropout(torch.cat((flipped_first_, flipped_second_), dim=1), p=drop_p, training=self.training)
        early_final = torch.nn.functional.dropout(early_out * (1.0 + self.squeeze(early_out)), p=drop_p, training=self.training)
        key_i = key_order[0]
        return getattr(self, "final_{}_early".format(str(key_i)))(early_final)