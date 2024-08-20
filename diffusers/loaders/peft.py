# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List, Union

from ..utils import MIN_PEFT_VERSION, check_peft_version, is_peft_available


class PeftAdapterMixin:
    """
    A class containing all functions for loading and using adapters weights that are supported in PEFT library. For
    more details about adapters and injecting them in a transformer-based model, check out the PEFT [documentation](https://huggingface.co/docs/peft/index).

    Install the latest version of PEFT, and use this mixin to:

    - Attach new adapters in the model.
    - Attach multiple adapters and iteratively activate/deactivate them.
    - Activate/deactivate all adapters from the model.
    - Get a list of the active adapters.
    """

    _hf_peft_config_loaded = False
    
    
    def load_adapter(
        self,
        peft_model_id = None,
        adapter_name = None,
        revision = None,
        token = None,
        device_map = "auto",
        max_memory = None,
        offload_folder = None,
        offload_index = None,
        peft_config = None,
        adapter_state_dict= None,
        adapter_kwargs = None,
    ) -> None:
        """
        Load adapter weights from file or remote Hub folder. If you are not familiar with adapters and PEFT methods, we
        invite you to read more about them on PEFT official documentation: https://huggingface.co/docs/peft

        Requires peft as a backend to load the adapter weights.

        Args:
            peft_model_id (`str`, *optional*):
                The identifier of the model to look for on the Hub, or a local path to the saved adapter config file
                and adapter weights.
            adapter_name (`str`, *optional*):
                The adapter name to use. If not set, will use the default adapter.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.

                <Tip>

                To test a pull request you made on the Hub, you can pass `revision="refs/pr/<pr_number>".

                </Tip>

            token (`str`, `optional`):
                Whether to use authentication token to load the remote folder. Userful to load private repositories
                that are on HuggingFace Hub. You might need to call `huggingface-cli login` and paste your tokens to
                cache it.
            device_map (`str` or `Dict[str, Union[int, str, torch.device]]` or `int` or `torch.device`, *optional*):
                A map that specifies where each submodule should go. It doesn't need to be refined to each
                parameter/buffer name, once a given module name is inside, every submodule of it will be sent to the
                same device. If we only pass the device (*e.g.*, `"cpu"`, `"cuda:1"`, `"mps"`, or a GPU ordinal rank
                like `1`) on which the model will be allocated, the device map will map the entire model to this
                device. Passing `device_map = 0` means put the whole model on GPU 0.

                To have Accelerate compute the most optimized `device_map` automatically, set `device_map="auto"`. For
                more information about each option see [designing a device
                map](https://hf.co/docs/accelerate/main/en/usage_guides/big_modeling#designing-a-device-map).
            max_memory (`Dict`, *optional*):
                A dictionary device identifier to maximum memory. Will default to the maximum memory available for each
                GPU and the available CPU RAM if unset.
            offload_folder (`str` or `os.PathLike`, `optional`):
                If the `device_map` contains any value `"disk"`, the folder where we will offload weights.
            offload_index (`int`, `optional`):
                `offload_index` argument to be passed to `accelerate.dispatch_model` method.
            peft_config (`Dict[str, Any]`, *optional*):
                The configuration of the adapter to add, supported adapters are non-prefix tuning and adaption prompts
                methods. This argument is used in case users directly pass PEFT state dicts
            adapter_state_dict (`Dict[str, torch.Tensor]`, *optional*):
                The state dict of the adapter to load. This argument is used in case users directly pass PEFT state
                dicts
            adapter_kwargs (`Dict[str, Any]`, *optional*):
                Additional keyword arguments passed along to the `from_pretrained` method of the adapter config and
                `find_adapter_config_file` method.
        """
        
        
        
        check_peft_version(min_version=MIN_PEFT_VERSION)

        adapter_name = adapter_name if adapter_name is not None else "default"
        if adapter_kwargs is None:
            adapter_kwargs = {}

        from peft import PeftConfig, inject_adapter_in_model, load_peft_weights
        from peft.utils import set_peft_model_state_dict

        if self._hf_peft_config_loaded and adapter_name in self.peft_config:
            raise ValueError(f"Adapter with name {adapter_name} already exists. Please use a different name.")

        if peft_model_id is None and (adapter_state_dict is None and peft_config is None):
            raise ValueError(
                "You should either pass a `peft_model_id` or a `peft_config` and `adapter_state_dict` to load an adapter."
            )

        if "device" not in adapter_kwargs:
            device = self.device if not hasattr(self, "hf_device_map") else list(self.hf_device_map.values())[0]
        else:
            device = adapter_kwargs.pop("device")

        # To avoid PEFT errors later on with safetensors.


        # We keep `revision` in the signature for backward compatibility
        if revision is not None and "revision" not in adapter_kwargs:
            adapter_kwargs["revision"] = revision
        elif revision is not None and "revision" in adapter_kwargs and revision != adapter_kwargs["revision"]:
            logger.error(
                "You passed a `revision` argument both in `adapter_kwargs` and as a standalone argument. "
                "The one in `adapter_kwargs` will be used."
            )

        # Override token with adapter_kwargs' token
        if "token" in adapter_kwargs:
            token = adapter_kwargs.pop("token")

        if peft_config is None:
            adapter_config_file = find_adapter_config_file(
                peft_model_id,
                token=token,
                **adapter_kwargs,
            )

            if adapter_config_file is None:
                raise ValueError(
                    f"adapter model file not found in {peft_model_id}. Make sure you are passing the correct path to the "
                    "adapter model."
                )

            peft_config = PeftConfig.from_pretrained(
                peft_model_id,
                token=token,
                **adapter_kwargs,
            )

        # Create and add fresh new adapters into the model.
        inject_adapter_in_model(peft_config, self, adapter_name)

        if not self._hf_peft_config_loaded:
            self._hf_peft_config_loaded = True

        if peft_model_id is not None:
            adapter_state_dict = load_peft_weights(peft_model_id, token=token, device=device, **adapter_kwargs)

        # We need to pre-process the state dict to remove unneeded prefixes - for backward compatibility
        processed_adapter_state_dict = {}
        prefix = "base_model.model."
        for key, value in adapter_state_dict.items():
            if key.startswith(prefix):
                new_key = key[len(prefix) :]
            else:
                new_key = key
            processed_adapter_state_dict[new_key] = value

        # Load state dict
        incompatible_keys = set_peft_model_state_dict(self, processed_adapter_state_dict, adapter_name)

        if incompatible_keys is not None:
            # check only for unexpected keys
            if hasattr(incompatible_keys, "unexpected_keys") and len(incompatible_keys.unexpected_keys) > 0:
                logger.warning(
                    f"Loading adapter weights from {peft_model_id} led to unexpected keys not found in the model: "
                    f" {incompatible_keys.unexpected_keys}. "
                )

        # Re-dispatch model and hooks in case the model is offloaded to CPU / Disk.
        if (
            (getattr(self, "hf_device_map", None) is not None)
            and (len(set(self.hf_device_map.values()).intersection({"cpu", "disk"})) > 0)
            and len(self.peft_config) == 1
        ):
            self._dispatch_accelerate_model(
                device_map=device_map,
                max_memory=max_memory,
                offload_folder=offload_folder,
                offload_index=offload_index,
            )
            
    def add_adapter(self, adapter_config, adapter_name: str = "default") -> None:
        r"""
        Adds a new adapter to the current model for training. If no adapter name is passed, a default name is assigned
        to the adapter to follow the convention of the PEFT library.

        If you are not familiar with adapters and PEFT methods, we invite you to read more about them in the PEFT
        [documentation](https://huggingface.co/docs/peft).

        Args:
            adapter_config (`[~peft.PeftConfig]`):
                The configuration of the adapter to add; supported adapters are non-prefix tuning and adaption prompt
                methods.
            adapter_name (`str`, *optional*, defaults to `"default"`):
                The name of the adapter to add. If no name is passed, a default name is assigned to the adapter.
        """
        check_peft_version(min_version=MIN_PEFT_VERSION)

        if not is_peft_available():
            raise ImportError("PEFT is not available. Please install PEFT to use this function: `pip install peft`.")

        from peft import PeftConfig, inject_adapter_in_model

        if not self._hf_peft_config_loaded:
            self._hf_peft_config_loaded = True
        elif adapter_name in self.peft_config:
            raise ValueError(f"Adapter with name {adapter_name} already exists. Please use a different name.")

        if not isinstance(adapter_config, PeftConfig):
            raise ValueError(
                f"adapter_config should be an instance of PeftConfig. Got {type(adapter_config)} instead."
            )

        # Unlike transformers, here we don't need to retrieve the name_or_path of the unet as the loading logic is
        # handled by the `load_lora_layers` or `LoraLoaderMixin`. Therefore we set it to `None` here.
        adapter_config.base_model_name_or_path = None
        inject_adapter_in_model(adapter_config, self, adapter_name)
        self.set_adapter(adapter_name)

    def set_adapter(self, adapter_name: Union[str, List[str]]) -> None:
        """
        Sets a specific adapter by forcing the model to only use that adapter and disables the other adapters.

        If you are not familiar with adapters and PEFT methods, we invite you to read more about them on the PEFT
        [documentation](https://huggingface.co/docs/peft).

        Args:
            adapter_name (Union[str, List[str]])):
                The list of adapters to set or the adapter name in the case of a single adapter.
        """
        check_peft_version(min_version=MIN_PEFT_VERSION)

        if not self._hf_peft_config_loaded:
            raise ValueError("No adapter loaded. Please load an adapter first.")

        if isinstance(adapter_name, str):
            adapter_name = [adapter_name]

        missing = set(adapter_name) - set(self.peft_config)
        if len(missing) > 0:
            raise ValueError(
                f"Following adapter(s) could not be found: {', '.join(missing)}. Make sure you are passing the correct adapter name(s)."
                f" current loaded adapters are: {list(self.peft_config.keys())}"
            )

        from peft.tuners.tuners_utils import BaseTunerLayer

        _adapters_has_been_set = False
        ms = self.named_modules()
        for names, module in ms:
            # if names.__contains__("to_out.0"):
            #     print()
            # if names.__contains__("to_out.1"):
            #     print()
            # if names.__contains__("to_out"):
            #     print()
            if isinstance(module, BaseTunerLayer):
                if hasattr(module, "set_adapter"):
                    module.set_adapter(adapter_name)
                # Previous versions of PEFT does not support multi-adapter inference
                elif not hasattr(module, "set_adapter") and len(adapter_name) != 1:
                    raise ValueError(
                        "You are trying to set multiple adapters and you have a PEFT version that does not support multi-adapter inference. Please upgrade to the latest version of PEFT."
                        " `pip install -U peft` or `pip install -U git+https://github.com/huggingface/peft.git`"
                    )
                else:
                    module.active_adapter = adapter_name
                _adapters_has_been_set = True

        if not _adapters_has_been_set:
            raise ValueError(
                "Did not succeeded in setting the adapter. Please make sure you are using a model that supports adapters."
            )

    def disable_adapters(self) -> None:
        r"""
        Disable all adapters attached to the model and fallback to inference with the base model only.

        If you are not familiar with adapters and PEFT methods, we invite you to read more about them on the PEFT
        [documentation](https://huggingface.co/docs/peft).
        """
        check_peft_version(min_version=MIN_PEFT_VERSION)

        if not self._hf_peft_config_loaded:
            raise ValueError("No adapter loaded. Please load an adapter first.")

        from peft.tuners.tuners_utils import BaseTunerLayer

        for _, module in self.named_modules():
            if isinstance(module, BaseTunerLayer):
                if hasattr(module, "enable_adapters"):
                    module.enable_adapters(enabled=False)
                else:
                    # support for older PEFT versions
                    module.disable_adapters = True

    def enable_adapters(self) -> None:
        """
        Enable adapters that are attached to the model. The model uses `self.active_adapters()` to retrieve the
        list of adapters to enable.

        If you are not familiar with adapters and PEFT methods, we invite you to read more about them on the PEFT
        [documentation](https://huggingface.co/docs/peft).
        """
        check_peft_version(min_version=MIN_PEFT_VERSION)

        if not self._hf_peft_config_loaded:
            raise ValueError("No adapter loaded. Please load an adapter first.")

        from peft.tuners.tuners_utils import BaseTunerLayer

        for _, module in self.named_modules():
            if isinstance(module, BaseTunerLayer):
                if hasattr(module, "enable_adapters"):
                    module.enable_adapters(enabled=True)
                else:
                    # support for older PEFT versions
                    module.disable_adapters = False

    def active_adapters(self) -> List[str]:
        """
        Gets the current list of active adapters of the model.

        If you are not familiar with adapters and PEFT methods, we invite you to read more about them on the PEFT
        [documentation](https://huggingface.co/docs/peft).
        """
        check_peft_version(min_version=MIN_PEFT_VERSION)

        if not is_peft_available():
            raise ImportError("PEFT is not available. Please install PEFT to use this function: `pip install peft`.")

        if not self._hf_peft_config_loaded:
            raise ValueError("No adapter loaded. Please load an adapter first.")

        from peft.tuners.tuners_utils import BaseTunerLayer

        for _, module in self.named_modules():
            if isinstance(module, BaseTunerLayer):
                return module.active_adapter
