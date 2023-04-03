"""
Generate permuations
"""
import json
import logging
import os
import shutil
from itertools import product
from pathlib import Path
from typing import Iterator, List

from omegaconf import OmegaConf

from utils import Singleton

log = logging.getLogger(__name__)


# pylint: disable = too-few-public-methods
class Permutation(metaclass=Singleton):
    """Permutation"""

    def __init__(
        self, directory: str, file_max_chunking: int, valid_qiskit_rs: List[str], n_qubits: int
    ) -> None:
        self.directory = directory
        self.file_max_chunking = file_max_chunking
        self.valid_qiskit_rs = valid_qiskit_rs
        self.n_qubits = n_qubits

        self.valid_cnots = [[i, i + 1] for i in range(self.n_qubits - 1)]
        self.cnots_space = [
            [y for y in CNOTs if y is not None]
            for CNOTs in list(product(*([x, None] for x in self.valid_cnots)))
        ]
        self.rs_qiskit_space = list(product(*([self.valid_qiskit_rs] * self.n_qubits)))
        self.nas_search_qiskit_space = product(self.rs_qiskit_space, self.cnots_space)

        self.nas_search_qiskit_space_count = 0

        # check directory exist and previously already ran permutation
        dirpath = Path(self.directory)
        if dirpath.exists() and dirpath.is_dir():
            metadata = json.load(
                open(
                    os.path.join(self.directory, "permutation_info.txt"),
                    mode="r",
                    encoding="utf-8",
                )
            )
            if (
                metadata.get("n_qubits", None) is None
                or metadata["n_qubits"] != self.n_qubits
                or metadata.get("file_max_chunking", None) is None
                or metadata["file_max_chunking"] != self.file_max_chunking
                or metadata.get("valid_qiskit_rs", None) is None
                or metadata["valid_qiskit_rs"] != self.valid_qiskit_rs
            ):
                log.info("re-run permutation and clear directory %s", self.directory)
                shutil.rmtree(dirpath)

                self.generate()
            else:
                # skip
                log.info("Re-use and skip running permutation with n_qubits= %d", self.n_qubits)
        else:
            self.generate()

    def _chunker(self, iterator: Iterator, chunk_size: int) -> Iterator[List[str]]:
        while True:
            chunk = []
            try:
                for _ in range(chunk_size):
                    line = json.dumps(next(iterator)) + "\n"
                    chunk.append(line)
                yield chunk
            except StopIteration:
                if chunk:
                    yield chunk
                return

    def generate(self) -> None:
        """Write permutation to file"""
        Path(self.directory).mkdir(parents=True, exist_ok=True)
        log.info("write permutation to directory %s", self.directory)
        for index, text in enumerate(
            self._chunker(self.nas_search_qiskit_space, self.file_max_chunking)
        ):
            file_path = os.path.join(self.directory, f"ansatz_output_{index}.txt")
            with open(file_path, mode="wt", encoding="utf-8") as write_file:
                write_file.write("".join(text))
            self.nas_search_qiskit_space_count += len(text)

        # write metadata
        with open(
            os.path.join(self.directory, "permutation_info.txt"), mode="w", encoding="utf-8"
        ) as permutation_info_writer:
            permutation_info_writer.write(
                json.dumps(
                    {
                        "NAS_search_qiskit_space_len": self.nas_search_qiskit_space_count,
                        "file_max_chunking": self.file_max_chunking,
                        "CNOTs_space_len": len(self.cnots_space),
                        "Rs_qiskit_space_len": len(self.rs_qiskit_space),
                        "n_qubits": self.n_qubits,
                        "valid_qiskit_rs": OmegaConf.to_object(self.valid_qiskit_rs),
                    },
                    indent=2,
                )
            )
