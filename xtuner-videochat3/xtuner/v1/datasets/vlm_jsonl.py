import json
import os

from xtuner.v1.utils import get_logger

from .jsonl import JsonlDataset


logger = get_logger()


class VLMJsonlDataset(JsonlDataset):
    def __init__(
        self,
        *args,
        media_root: str | None = "",
        **kwargs,
    ):
        if media_root is None:
            media_root = ""
        self.media_root = media_root

        super().__init__(*args, **kwargs)

        self.fake_data = {
            "id": -1,
            "messages": [
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": "你好呀！很高兴为你服务～有什么问题或需要帮忙的地方，随时告诉我哦！"},
            ],
        }

    def __getitem__(self, item):
        try:
            with open(self.path, "rb") as f:
                f.seek(self.offsets[item])
                line = f.readline().decode("utf-8")

            raw_data = json.loads(line)

            if self.tokenize_fn:
                tokenized_data = self.tokenize_fn(raw_data, media_root=self.media_root)
                data = tokenized_data
            else:
                data = raw_data
        except Exception as e:
            import traceback
            logger.warning(
                f"[{os.path.basename(self.path)}] Error at item {item}:\n"
                f"Exception: {e}\n"
                f"Line content (first 200 chars): {repr(line[:200]) if 'line' in locals() else 'N/A'}\n"
                f"Traceback:\n{traceback.format_exc()}\n"
                "Dumping a fake data!"
            )
            data = self.tokenize_fn(self.fake_data)
            assert isinstance(data, dict), f"Expected dict, got {type(data)}"
            if "labels" in data:
                data["labels"] = len(data["input_ids"]) * [-100]
        return data
