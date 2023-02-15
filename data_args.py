from dataclasses import dataclass, field


@dataclass
class DataArguments:
    """
    Arguments relating to data.
    """

    pairs_path: str = field(
        default="./data/pairs.csv", metadata={"help": "path to pairs csv file"}
    )
    prompt_path: str = field(
        default="./data/prompt.csv", metadata={"help": "path to prompt csv file"}
    )
    image_path: str = field(
        default="./data/image.csv", metadata={"help": "path to image csv file"}
    )
    correlation_path: str = field(
        default="./data/correlation.csv",
        metadata={"help": "correlation between prompt and images"},
    )
    image_folder: str = field(
        default="./data", metadata={"help": "path to image folder"}
    )
    fold: int = field(default=0, metadata={"help": "fold for validation"})
    top_k_neighbors: int = field(
        default=50,
        metadata={
            "help": "select top_k nearest neighbors for training and valiation set"
        },
    )
