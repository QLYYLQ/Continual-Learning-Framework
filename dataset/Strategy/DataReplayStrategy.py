from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from typing import List, TypeVar, Generic
from ..LoadStrategy.BaseLoader import LoadProtocol

# _T = TypeVar('_T', covariant=True)

class BaseReplay(ABC, Dataset):
    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx):
        raise NotImplementedError

    @abstractmethod
    def update_replay_state(self, image_list: List) -> List:
        raise NotImplementedError

    @abstractmethod
    def update_strategy(self, *args, **kwargs):
        """
        In this method, you should implement your own strategy for replay, and return a list which can be used to
        update the replay.
        """
        raise NotImplementedError


from __future__ import annotations
import abc
from typing import Any, Tuple, Protocol, Optional, Dict
import torch
from torch.utils.data import Dataset
from PIL import Image
import logging


# ----------------------
# 抽象接口层 (Contracts)
# ----------------------

class BaseLoader(Protocol):
    """数据加载器抽象协议（依赖注入接口）"""

    def load_image(self, identifier: Any) -> Image.Image:
        ...

    def load_mask(self, identifier: Any) -> Image.Image:
        ...

    @property
    def class_mapping(self) -> Dict[int, int]:
        ...


class ReplayStrategy(abc.ABC):
    """回放策略核心协议"""

    @abc.abstractmethod
    def acquire_samples(self,
                        model: Optional[torch.nn.Module],
                        device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        ...

    @abc.abstractmethod
    def update_memory(self,
                      data: Tuple[torch.Tensor, torch.Tensor],
                      model: Optional[torch.nn.Module]) -> None:
        ...


# --------------------------
# 基础设施层 (Infrastructure)
# --------------------------

class DataLoaderRegistry:
    """数据加载器注册中心（实现依赖注入）"""
    _loaders: Dict[str, BaseLoader] = {}

    @classmethod
    def register(cls, name: str, loader: BaseLoader):
        cls._loaders[name] = loader
        logging.info(f"Registered loader: {name}")

    @classmethod
    def get(cls, name: str) -> BaseLoader:
        if name not in cls._loaders:
            raise KeyError(f"LoadProtocol {name} not registered")
        return cls._loaders[name]


# ----------------------
# 核心实现层 (Core)
# ----------------------

class BaseReplayBuffer(Dataset, abc.ABC):
    """回放缓冲基础类"""

    def __init__(self,
                 loader: BaseLoader,
                 transform: Optional[torch.nn.Module] = None,
                 target_transform: Optional[torch.nn.Module] = None):
        self.loader = loader
        self.transform = transform
        self.target_transform = target_transform
        self._storage = []

    def __len__(self) -> int:
        return len(self._storage)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        identifier = self._storage[idx]
        img = self.loader.load_image(identifier)
        mask = self.loader.load_mask(identifier)

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            mask = self.target_transform(mask)
        return img, mask


class SampleReplayStrategy(ReplayStrategy):
    """样本回放策略实现基类"""

    def __init__(self,
                 buffer: BaseReplayBuffer,
                 selection_policy: str = "random"):
        self.buffer = buffer
        self.selection_policy = selection_policy

    def acquire_samples(self,
                        model: Optional[torch.nn.Module] = None,
                        device: torch.device = torch.device('cpu')) -> Tuple[torch.Tensor, torch.Tensor]:
        indices = self._select_indices(len(self.buffer))
        return self.buffer[indices]

    @abc.abstractmethod
    def _select_indices(self, buffer_size: int) -> torch.Tensor:
        ...


class GenerativeReplayStrategy(ReplayStrategy):
    """生成回放策略实现基类"""

    def __init__(self,
                 generator: torch.nn.Module,
                 latent_dim: int = 128):
        self.generator = generator
        self.latent_dim = latent_dim

    def acquire_samples(self,
                        model: torch.nn.Module,
                        device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        z = torch.randn(1, self.latent_dim, device=device)
        with torch.no_grad():
            fake_img = self.generator(z)
            pseudo_mask = model(fake_img).argmax(1)
        return fake_img, pseudo_mask


# ----------------------
# 应用层 (Application)
# ----------------------

class ContinuousLearningDataset(Dataset):
    """持续学习数据聚合器"""

    def __init__(self,
                 original: Dataset,
                 replay_strategies: Dict[str, ReplayStrategy],
                 sampling_weights: Dict[str, float],
                 device: torch.device):
        self.original = original
        self.strategies = replay_strategies
        self.weights = torch.tensor(list(sampling_weights.values()))
        self.device = device

    def __len__(self) -> int:
        return len(self.original)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if torch.rand(1) < self._get_dynamic_ratio():  # 动态调整比例
            strategy_name = self._select_strategy()
            return self.strategies[strategy_name].acquire_samples(device=self.device)
        return self.original[idx]

    def _select_strategy(self) -> str:
        return list(self.strategies.keys())[torch.multinomial(self.weights, 1).item()]

    def _get_dynamic_ratio(self) -> float:
        # 实现动态调整逻辑（例如基于训练进度）
        return 0.3


# ----------------------
# 具体实现示例
# ----------------------

class DiskLoader(BaseLoader):
    """本地磁盘加载器实现"""

    def __init__(self, root_dir: str, class_map: Dict[int, int]):
        self.root_dir = root_dir
        self._class_map = class_map

    def load_image(self, identifier: str) -> Image.Image:
        return Image.open(f"{self.root_dir}/images/{identifier}.jpg")

    def load_mask(self, identifier: str) -> Image.Image:
        return Image.open(f"{self.root_dir}/masks/{identifier}.png")

    @property
    def class_mapping(self) -> Dict[int, int]:
        return self._class_map


class FeatureBasedBuffer(BaseReplayBuffer):
    """基于特征重要性的缓冲实现"""

    def __init__(self,
                 loader: BaseLoader,
                 feature_extractor: torch.nn.Module,
                 capacity: int = 1000):
        super().__init__(loader)
        self.feature_extractor = feature_extractor
        self.capacity = capacity
        self._importance_scores = []

    def update_memory(self,
                      data: Tuple[torch.Tensor, torch.Tensor],
                      model: Optional[torch.nn.Module] = None):
        # 实现基于特征重要性的更新策略
        with torch.no_grad():
            features = self.feature_extractor(data[0])
            importance = torch.norm(features, dim=1).mean().item()

        if len(self._storage) >= self.capacity:
            # 替换重要性最低的样本
            min_idx = torch.argmin(torch.tensor(self._importance_scores))
            self._storage[min_idx] = data
            self._importance_scores[min_idx] = importance
        else:
            self._storage.append(data)
            self._importance_scores.append(importance)


# ----------------------
# 初始化与使用示例
# ----------------------

# 注册数据加载器
DataLoaderRegistry.register(
    "cityscapes",
    DiskLoader(
        root_dir="/data/cityscapes",
        class_map={0: 0, 1: 1, ...}
    )
)

# 构建系统组件
loader = DataLoaderRegistry.get("cityscapes")
feature_extractor = torch.hub.load('pytorch/vision', 'resnet50', pretrained=True)
buffer = FeatureBasedBuffer(loader, feature_extractor)

replay_strategies = {
    "sample": SampleReplayStrategy(buffer),
    "generative": GenerativeReplayStrategy(generator=GANGenerator())
}

dataset = ContinuousLearningDataset(
    original=CityscapesDataset(),
    replay_strategies=replay_strategies,
    sampling_weights={"sample": 0.7, "generative": 0.3},
    device=torch.device('cuda')
)

# 训练循环
for epoch in range(100):
    for inputs, targets in DataLoader(dataset, batch_size=32):
        # 训练逻辑...
        # 更新回放缓冲
        buffer.update_memory((inputs, targets))
