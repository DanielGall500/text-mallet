from dataloaders.base import DataLoader


class TxtLoader(DataLoader):
    def load(self, path_to_dataset: str) -> str:
        with open(path_to_dataset, "r") as f:
            full_text = f.read()
            return full_text
