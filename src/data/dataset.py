"""Tokenize complete paired sentences and report rows exceeding the model limits."""
from torch.utils.data import Dataset
from transformers import DataCollatorForSeq2Seq


class SpellingDataset(Dataset):
    def __init__(self, rows, tokenizer, max_source_length, max_target_length, prefix="spelling: "):
        self.samples = []
        self.excluded_ids = []
        for row in rows:
            source = tokenizer(prefix + row["input_text"], truncation=False, verbose=False)
            target = tokenizer(text_target=row["target_text"], truncation=False, verbose=False)
            if len(source["input_ids"]) > max_source_length or len(target["input_ids"]) > max_target_length:
                self.excluded_ids.append(row["id"])
                continue
            source["labels"] = target["input_ids"]
            self.samples.append(source)
        if not self.samples:
            raise ValueError("Không có cặp câu nào vừa giới hạn token")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def build_data_collator(tokenizer, model):
    return DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True, label_pad_token_id=-100)
