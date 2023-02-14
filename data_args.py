from dataclasses import dataclass, field


@dataclass
class DataArguments:
    """
    Arguments relating to data.
    """

    prompt_path: str = field(
        default="./data/supervised_correlations.csv", metadata={"help": "path to prompt csv file"}
    )
    image_path: str = field(
        default="./data/image.csv", metadata={"help": "path to image csv file"}
    )
    correlation_path: str = field(
        default="./data/correlations.csv", metadata={"help": "correlation between prompt and images"}
    )
    image_folder: str = field(
        default="./data/images", metadata={"help": "path to image folder"}
    )
    top_k_neighbors: int = field(default=50, metadata={"help": "select top_k nearest neighbors for training and valiation set"})