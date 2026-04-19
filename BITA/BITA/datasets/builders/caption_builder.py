from .base_dataset_builder import BaseDatasetBuilder
from ..datasets.rsicd_caption_datasets import (
    RSICDCapDataset,
    RSICDCapEvalDataset
)
from ..datasets.nwpu_caption_datasets import (
    NWPUCapDataset,
    NWPUCapEvalDataset
)

from ..datasets.ucm_caption_datasets import (
    UCMCapDataset,
    UCMCapEvalDataset
)

from BITA.common.registry import Registry


@Registry.register_dataset_builder("rsicd_caption")
class RSICDCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = RSICDCapDataset
    eval_dataset_cls = RSICDCapEvalDataset

    # 使用相对路径，提高可移植性
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/rsicd/defaults_cap.yaml"
    }


@Registry.register_dataset_builder("nwpu_caption")
class NWPUCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = NWPUCapDataset
    eval_dataset_cls = NWPUCapEvalDataset

    # 使用相对路径，提高可移植性
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/nwpu/defaults_cap.yaml"
    }


@Registry.register_dataset_builder("ucm_caption")
class UCMCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = UCMCapDataset
    eval_dataset_cls = UCMCapEvalDataset

    # 使用相对路径，提高可移植性
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/ucm/defaults_cap.yaml"
    }