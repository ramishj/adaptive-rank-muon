"""
Arrow-based packed dataset loader for pre-tokenized data.

Expects a directory containing data-*.arrow files, each with an "input_ids" column
of pre-tokenized, packed sequences (e.g., OpenWebText tokenized with Pythia tokenizer,
packed to seq_len=2048).

Design notes:
  - Lazy loading: only one arrow file is memory-mapped at a time
  - Supports both IPC stream and file formats (tries stream first)
  - num_workers=0 recommended for NFS mounts (temp file cleanup issues on full disks)
"""

import glob
import os

import torch
from torch.utils.data import Dataset

try:
    import pyarrow as pa
except ImportError:
    pa = None


class ArrowPackedDataset(Dataset):
    """
    PyTorch Dataset backed by Arrow IPC files.

    Args:
        data_path: Directory containing data-*.arrow files.

    Each item returns {"input_ids": List[int]} of length seq_len.
    """

    def __init__(self, data_path: str):
        if pa is None:
            raise ImportError("pyarrow is required for ArrowPackedDataset")

        self.arrow_files = sorted(glob.glob(os.path.join(data_path, "data-*.arrow")))
        if not self.arrow_files:
            raise FileNotFoundError(f"No data-*.arrow files in {data_path}")

        # Build cumulative index for file-level sharding
        self._cumulative = [0]
        for fp in self.arrow_files:
            table = self._read_arrow(fp)
            self._cumulative.append(self._cumulative[-1] + len(table))

        self._total = self._cumulative[-1]
        self._cur_idx = -1
        self._cur_table = None

    @staticmethod
    def _read_arrow(filepath: str):
        """Read an Arrow file, trying stream format first, then file format."""
        with open(filepath, "rb") as f:
            try:
                reader = pa.ipc.RecordBatchStreamReader(f)
                return reader.read_all()
            except Exception:
                f.seek(0)
                reader = pa.ipc.RecordBatchFileReader(f)
                return reader.read_all()

    def __len__(self) -> int:
        return self._total

    def __getitem__(self, idx: int) -> dict:
        # Find which file contains this index
        fi = 0
        while idx >= self._cumulative[fi + 1]:
            fi += 1

        # Lazy-load the file if not already cached
        if fi != self._cur_idx:
            self._cur_table = self._read_arrow(self.arrow_files[fi])
            self._cur_idx = fi

        row = self._cur_table["input_ids"][idx - self._cumulative[fi]]
        ids = row.as_py() if hasattr(row, "as_py") else list(row)
        return {"input_ids": ids}


def collate_packed(batch: list[dict]) -> torch.Tensor:
    """Collate function for packed datasets. Returns (batch_size, seq_len) tensor."""
    return torch.stack([torch.tensor(x["input_ids"], dtype=torch.long) for x in batch])
