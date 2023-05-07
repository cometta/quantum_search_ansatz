""" Circuit Model """
import json
import logging
import os
import time
import zipfile
from itertools import islice
from typing import List

import numpy as np
from cachetools import LRUCache, cached
from cachetools.keys import hashkey
from nptyping import Float, Int, NDArray, Shape
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.providers.fake_provider.fake_backend import FakeBackend

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def read_line(
    line_number_zero_based: int, file_max_chunking: int, permutation_directory: str = ""
) -> List[tuple[List[str], List[tuple[int, int]]]]:
    """read permutations from file

    Reference:
        https://stackoverflow.com/questions/19189961/python-fastest-access-to-nth-line-in-huge-file

    Args:
        line_number_zero_based (int): line number, start counting from zero
        file_max_chunking (int): maximum number of line to chunk
        permutation_directory (str): permutation directory

    Returns:
        List[tuple[List[str], List[tuple[List[int]]]]]: each line from the permutation file
    """
    index = line_number_zero_based // file_max_chunking
    offset = line_number_zero_based % file_max_chunking

    with zipfile.ZipFile(
        os.path.join(permutation_directory, "ansatz_output.zip"), mode="r"
    ) as open_zip:
        with open_zip.open(f"ansatz_output_{index}.txt") as open_permutation_file:
            line = list(
                islice(open_permutation_file, offset, offset + 1)
            )  # islice start index from zero
            return json.loads(line[0])


@cached(
    cache=LRUCache(maxsize=640 * 1024, getsizeof=len),
    key=lambda n_qubits, n_layers, arch, seed, metadata, p_dir, device, transpile_q_layout: hashkey(
        n_qubits, n_layers, str(arch), seed, p_dir, str(transpile_q_layout)
    ),
)
def circuit_search(
    n_qubits: int,
    n_layers: int,
    arch: List[int],
    seed: int,
    metadata: dict,
    p_dir: str,
    device: FakeBackend,
    transpile_q_layout: List[int],
) -> QuantumCircuit:
    """Create ansatz cached with 640K

    Args:
        n_qubits (int): Number of qubits.
        n_layers (int): Number of layers.
        arch (List[int]): Subnet list.
        seed (int): Random seed.
        metadata (dict): Permutation metadata.
        p_dir (str): Permutation output directory.
        device (FakeBackend): Qiskit backend v1.
        transpile_q_layout (List[int]): Initial layout for transpiler.

    Returns:
        QuantumCircuit: Qiskit Quantum Circuit
    """

    ansatz_custom = QuantumCircuit(n_qubits)

    for layer in range(n_layers):
        line = read_line(arch[layer], metadata["file_max_chunking"], p_dir)

        for qubit_count, gate in enumerate(line[0]):
            if gate == "Y":
                ansatz_custom.ry(Parameter(f"{qubit_count}_" + str(layer)), [qubit_count])
            elif gate == "Z":
                ansatz_custom.rz(Parameter(f"{qubit_count}_" + str(layer)), [qubit_count])
            elif gate == "X":
                ansatz_custom.rx(Parameter(f"{qubit_count}_" + str(layer)), [qubit_count])

        # for cnot_element in  NAS_search_qiskit_space[arch[layer]][1]:
        for cnot_element in line[1]:
            ansatz_custom.cnot(*cnot_element)

        ansatz_custom.barrier()
    # log.info(f"original circuit {ansatz_custom.draw('text')}")

    ansatz_custom = transpile(
        ansatz_custom,
        seed_transpiler=seed,
        backend=device,
        initial_layout=None
        if isinstance(transpile_q_layout, str) and transpile_q_layout == "None"
        else transpile_q_layout,
        optimization_level=3,
    )
    # log.info(f"after transpiled {ansatz_custom.draw('text')} ")

    return ansatz_custom


class CircuitSearchModel:
    """
    Circuit Model
    """

    def __init__(
        self,
        n_qubits: int,
        n_layers: int,
        n_experts: int,
        rng: np.random.Generator,
        seed: int,
        metadata: dict,
        permutation_directory: str,
        device: FakeBackend,
        transpile_q_layout: List[int],
    ):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_experts = n_experts

        self.params_space: NDArray[Shape["*, *, *, *"], Float] = rng.uniform(
            0, np.pi * 2, (n_experts, n_layers, metadata["Rs_qiskit_space_len"], n_qubits)
        )
        self.params: NDArray[Shape["*, *"], Float]
        self.seed = seed
        self.metadata = metadata
        self.permutation_directory = permutation_directory
        self.device = device
        self.transpile_q_layout = transpile_q_layout

    def get_params(
        self, subnet: NDArray[Shape["*"], Int], expert_idx: int
    ) -> NDArray[Shape["*, *"], Float]:
        """
        Each time get_params is called, generate new new ansatz
        """

        self.subnet = subnet
        self.expert_idx = expert_idx

        params = []
        for j in range(self.n_layers):
            r_idx = subnet[j] // self.metadata["CNOTs_space_len"]
            # self.params_space.shape (5, 3, 16, 4)
            params.append(self.params_space[expert_idx, j, r_idx : r_idx + 1])
        # np.array(params).shape=(3, 1, 4)  np.concatenate(params, axis=0)=(3, 4)
        # (layer, an item from rs_space, qubits)
        return np.concatenate(params, axis=0)

    def set_params(self, params: NDArray[Shape["*, *"], Float]) -> None:
        """Set the paramaters to the param space

        Args:
            params (NDArray N X M ): parameters
        """
        for j in range(self.n_layers):
            r_idx = self.subnet[j] // self.metadata["CNOTs_space_len"]
            self.params_space[self.expert_idx, j, r_idx : r_idx + 1] = params[j, :]

    def __call__(self, transpile_q_layout=None):
        return circuit_search(
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            arch=self.subnet,
            seed=self.seed,
            metadata=self.metadata,
            p_dir=self.permutation_directory,
            device=self.device,
            transpile_q_layout=transpile_q_layout
            if transpile_q_layout
            else self.transpile_q_layout,
        )
