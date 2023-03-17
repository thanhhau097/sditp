from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    """
    Arguments relating to model.
    """
    model_type: str = field(default="timm", metadata={"help": "timm or sd"})
    model_name: str = field(default="resnet50", metadata={"help": "timm model name"})
    resume: Optional[str] = field(
        default=None, metadata={"help": "Path of model checkpoint"}
    )
    objective: str = field(default="cosine", metadata={"help": "cosine or contrastive"})
