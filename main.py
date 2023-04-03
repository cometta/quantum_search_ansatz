""" Main search program """
import json
import logging
import os
from dataclasses import dataclass
from functools import partial
from typing import Callable, List, cast

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from nptyping import Int, NDArray, Shape
from omegaconf import DictConfig, OmegaConf
from qiskit import qpy
from qiskit.algorithms import NumPyEigensolver
from qiskit.opflow.primitive_ops import PauliSumOp
from qiskit.primitives import BackendEstimator
from qiskit.providers.fake_provider.fake_backend import FakeBackend
from qiskit.utils import algorithm_globals
from qiskit_aer.primitives import Estimator as Aer_Estimator

from evolution.evolution_sampler import EvolutionSampler
from model import CircuitSearchModel

disable_log = logging.getLogger("qiskit")
disable_log.setLevel(logging.WARN)

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def create_estimator(
    seed: int, is_noisy: bool, device: FakeBackend
) -> Aer_Estimator | BackendEstimator:
    """create a noise-less or noisy estimator

    Args:
        seed (int):  Random seed
        is_noisy (bool): Is noisy estimator
        device (FakeBackend): Qiskit backend v1

    Returns:
        Aer_Estimator | BackendEstimator: Estimator
    """
    if is_noisy:
        estimator = BackendEstimator(backend=device)
        log.info("using noisy estimator")
    else:
        estimator = Aer_Estimator(
            backend_options={"method": "statevector"},
            run_options={"shots": None, "seed": seed},
            transpile_options={"seed_transpiler": seed},
            approximation=True,
        )
    return estimator


@dataclass
class ExpertInfo:
    """
    Expert info
    """

    subnet: NDArray[Shape["*"], Int]
    n_experts: int


@dataclass
class TranspileInfo:
    """
    Transpilation info
    """

    hamiltonian_for_device: PauliSumOp
    transpile_q_layout: List[int]


def expert_evaluator(
    model: CircuitSearchModel,
    expert_info: ExpertInfo,
    estimator: Aer_Estimator | BackendEstimator,
    traspile_info: TranspileInfo,
):
    """In this function, we locate the expert that achieves the minimum loss,
    where such an expert is the best choice
    for the given subset

    Args:
        model (CircuitSearchModel): Search Model
        expert_info (ExpertInfo): Expert information
        estimator (Aer_Estimator | BackendEstimator): Estimator
        transpile_info (TranspileInfo) Transpiler information. \
            Hamiltonian and initial position of virtual qubits on physical qubits

    Returns:
        target_expert: best expert
    """
    target_expert = 0
    target_loss = None

    for i in range(expert_info.n_experts):
        model.params = model.get_params(expert_info.subnet, i)

        job = estimator.run(
            model(traspile_info.transpile_q_layout),
            traspile_info.hamiltonian_for_device,
            model.params.flatten(),
        )
        result = job.result()

        temp_loss = result.values[0]

        if target_loss is None or temp_loss < target_loss:
            target_loss = temp_loss
            target_expert = i
    return target_expert


def calculate_exact_value(cfg: DictConfig, hamiltonian_numpy_eigen: PauliSumOp) -> float:
    """Calculate exact eigen value

    Args:
        cfg (DictConfig): Hydra config
        hamiltonian_numpy_eigen (PauliSumOp): hamitonian

    Raises:
        TypeError: When enter invalid value in the config

    Returns:
        float: electronic ground state value
    """
    if isinstance(cfg.search.exact_value, str) and cfg.search.exact_value.lower() == "auto":
        exact_solver = NumPyEigensolver(k=1)
        exact_result = exact_solver.compute_eigenvalues(hamiltonian_numpy_eigen)
        exact_value = np.round(exact_result.eigenvalues[0], 4)
        log.info(
            "Computed electronic ground state energy using NumPyEigensolver exact_value= %f",
            exact_value,
        )
    elif isinstance(cfg.search.exact_value, (int, float)):
        exact_value = cfg.search.exact_value
    else:
        raise TypeError(
            "please enter 'auto' or exact electronic ground state energy value for 'exact_value'"
        )

    return exact_value


def get_device_hamiltonian_and_layout(
    cfg: DictConfig,
    hamiltonian_numpy_eigen: PauliSumOp,
    device_n_qubits: int,
    model: CircuitSearchModel,
) -> TranspileInfo:
    """Auto select best layout for the device

    Args:
        cfg (DictConfig): Hydra config
        hamiltonian_numpy_eigen (PauliSumOp): hamitonian
        device_n_qubits (int): number of qubits for the device
        model (CircuitSearchModel): Cicruit model

    Raises:
        ValueError: When device has less number of qubits than the problem

    Returns:
        TranspileInfo: Inflated hamitonian and best layout
    """
    transpile_q_layout: List[int] = []
    if (
        device_n_qubits > hamiltonian_numpy_eigen.num_qubits
        and isinstance(cfg.search.transpile_q_layout, str)
        and cfg.search.transpile_q_layout == "None"
    ):
        model.get_params(np.zeros(cfg.search.circuit.n_layers, dtype=int), 0)  # set self.subnet
        ansatz_inflated = (
            model()
        )  # put None as argument so that transpiler auto decide which qubit to use

        for qubit_index in range(ansatz_inflated.num_qubits):
            # pylint: disable=W0212
            if "ancilla" not in str(ansatz_inflated._layout.initial_layout[qubit_index]):
                transpile_q_layout.append(qubit_index)
        log.info("detect_initial_layout %s", transpile_q_layout)
        hamiltonian_for_device = hamiltonian_numpy_eigen.permute(transpile_q_layout)
    elif device_n_qubits > hamiltonian_numpy_eigen.num_qubits and not isinstance(
        cfg.search.transpile_q_layout, str
    ):
        log.info("use transpile_q_layout %s from config", cfg.search.transpile_q_layout)
        hamiltonian_for_device = hamiltonian_numpy_eigen.permute(cfg.search.transpile_q_layout)
        transpile_q_layout = cast(List[int], OmegaConf.to_object(cfg.search.transpile_q_layout))
    elif device_n_qubits < hamiltonian_numpy_eigen.num_qubits:
        raise ValueError("the device has less number of qubits than the problem")
    else:
        transpile_q_layout = cast(List[int], OmegaConf.to_object(cfg.search.transpile_q_layout))
        hamiltonian_for_device = hamiltonian_numpy_eigen

    return TranspileInfo(hamiltonian_for_device, transpile_q_layout)


def rel_err(target: float, measured: float) -> float:
    """
    relative error

    Args:
        target (float): target value
        measured (float): measured value

    Returns:
        float: relative error
    """
    return abs((target - measured) / target)


def report(
    expert_info: ExpertInfo,
    sorted_result: List[tuple[str, tuple[float, float]]],
    model: CircuitSearchModel,
    estimator_func: Aer_Estimator | BackendEstimator,
    transpile_info: TranspileInfo,
) -> None:
    """Generate report by outputing files

    Args:
        expert_info (ExpertInfo): Expert information
        sorted_result (List[tuple[str, tuple[float,float]]]): Sorted result
        model (CircuitSearchModel): Circuit Model
        estimator_func (Aer_Estimator | BackendEstimator): a noise-less or noisy estimator
        transpile_info (TranspileInfo): Transpile information
    """
    # sorted_result = list(result.items())
    # sorted_result.sort(key=lambda x: x[1], reverse=True)
    with open(
        os.path.join(HydraConfig.get().runtime.output_dir, "nas_result_sorted.txt"),
        mode="w",
        encoding="utf-8",
    ) as write_nas:
        write_nas.write("\n".join([f"{x[0]} {x[1]}" for x in sorted_result]))

    with open(
        os.path.join(HydraConfig.get().runtime.output_dir, "params_selected.txt"),
        mode="wt",
        encoding="utf-8",
    ) as write_params:
        subnet_int = np.array([int(x) for x in sorted_result[0][0].strip("[]").split()])
        expert_idx = expert_evaluator(model, expert_info, estimator_func(), transpile_info)
        model.params = model.get_params(subnet_int, expert_idx)
        params_selected = {
            "subnet": subnet_int.tolist(),
            "params": model.params.tolist(),
            "transpile_q_layout": transpile_info.transpile_q_layout,
            "result": sorted_result[0],
        }

        write_params.write(json.dumps(params_selected))
        log.info("params_selected %s", params_selected)
        # save circuit, must call after get_params
        circuit_found = model(transpile_info.transpile_q_layout)
        log.info(circuit_found.draw("text"))

        with open(
            os.path.join(HydraConfig.get().runtime.output_dir, "circuit_selected.qpy"), "wb"
        ) as circuit_selected_writer:
            qpy.dump(
                circuit_found,
                circuit_selected_writer,
            )


def generate_permutations(cfg: DictConfig, n_qubits: int) -> dict:
    """Generate permutation or re-use permutations

    Args:
        cfg (DictConfig): Hydra config
        n_qubits (int): number of qubits

    Returns:
        dict: Metadata information in dictionary
    """

    hydra.utils.instantiate(cfg.search.permutation.generator, _partial_=True)(n_qubits=n_qubits)

    with open(
        os.path.join(cfg.search.permutation.generator.directory, "permutation_info.txt"),
        mode="r",
        encoding="utf-8",
    ) as permutation_info_writer:
        metadata = json.load(permutation_info_writer)
        log.info("permutation metadata info: %s", metadata)

    return metadata


def set_seed(seed: int) -> np.random.Generator:
    """Set Random Seed

    Args:
        seed (int): number for setting seed

    Returns:
        np.random.Generator: Numpy generator
    """
    algorithm_globals.random_seed = seed
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    return rng


def test_subnet_evolution_wrapper(
    estimator_func: Aer_Estimator | BackendEstimator,
    model: CircuitSearchModel,
    n_experts: int,
    transpile_info: TranspileInfo,
    exact_value: float,
) -> Callable:
    """For evolution to test subnet

    Args:
        estimator_func (Aer_Estimator | BackendEstimator): a noise-less or noisy estimator
        model (CircuitSearchModel): Circuit model
        n_experts (int): number of experts
        transpile_info (TranspileInfo): Transpile information
        exact_value (float): electronic ground state value

    Returns:
        Callable: return test_subnet_evolution function
    """

    def test_subnet_evolution(subnet: NDArray[Shape["*"], Int]) -> tuple[float, float]:
        estimator = estimator_func()

        expert_idx = expert_evaluator(
            model,
            ExpertInfo(subnet, n_experts),
            estimator,
            transpile_info,
        )
        model.params = model.get_params(subnet, expert_idx)
        job = estimator.run(
            model(transpile_info.transpile_q_layout),
            transpile_info.hamiltonian_for_device,
            model.params.flatten(),
        )

        result = job.result()
        energy = result.values[0]
        score = -np.abs(energy - (exact_value))
        return score, energy  # higher is better

    return test_subnet_evolution


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main function"""
    rng = set_seed(cfg.search.seed)

    device = hydra.utils.instantiate(cfg.search.device)
    log.info("device selected is %s number of qubits %d", device, device.configuration().n_qubits)

    hamiltonian_numpy_eigen = hydra.utils.instantiate(cfg.search.hamiltonian).create()
    n_qubits: int = 0
    if device.configuration().n_qubits == hamiltonian_numpy_eigen.num_qubits and not isinstance(
        cfg.search.transpile_q_layout, str
    ):
        n_qubits = len(cfg.search.transpile_q_layout)
        log.info(
            "user provided hamiltonian with the same number of qubits for the device %s", device
        )
    elif (
        device.configuration().n_qubits == hamiltonian_numpy_eigen.num_qubits
        and isinstance(cfg.search.transpile_q_layout, str)
        and cfg.search.transpile_q_layout == "None"
    ):
        # because can't find out exact number of qubits before inflate
        raise ValueError("This is not yet supported")
    else:
        n_qubits = hamiltonian_numpy_eigen.num_qubits

    metadata = generate_permutations(cfg, n_qubits)

    model = CircuitSearchModel(
        n_qubits,
        cfg.search.circuit.n_layers,
        cfg.search.qas.n_experts,
        rng,
        cfg.search.seed,
        metadata=metadata,
        permutation_directory=cfg.search.permutation.generator.directory,
        device=device,
        transpile_q_layout=cfg.search.transpile_q_layout,
    )

    transpile_info = get_device_hamiltonian_and_layout(
        cfg, hamiltonian_numpy_eigen, device.configuration().n_qubits, model
    )

    estimator_func = partial(create_estimator, cfg.search.seed, cfg.search.noise, device)

    # train
    for epoch in range(cfg.search.epochs):
        estimator = estimator_func()

        subnet = rng.integers(
            0, metadata["NAS_search_qiskit_space_len"], (cfg.search.circuit.n_layers,)
        )
        # find the expert with minimal loss w.r.t. subnet
        if epoch < cfg.search.warmup_epochs:
            expert_idx = rng.integers(cfg.search.qas.n_experts)
        else:
            expert_idx = expert_evaluator(
                model,
                ExpertInfo(subnet, cfg.search.qas.n_experts),
                estimator,
                transpile_info,
            )

        log.info("subnet: %s, epoch: %d  expert_idx: %d", subnet, epoch, expert_idx)

        params = model.get_params(subnet, expert_idx)

        custom_vqe = hydra.utils.instantiate(
            cfg.search.vqe,
            estimator,
            model(transpile_info.transpile_q_layout),
            params=params.flatten(),
        )

        result = custom_vqe.compute_minimum_eigenvalue(transpile_info.hamiltonian_for_device)
        model.params = result.optimal_parameters.reshape(
            cfg.search.circuit.n_layers,
            n_qubits,
        )

        model.set_params(model.params)

    if cfg.search.qas.searcher == "evolution":
        log.info("Starting evolution phrase")

        sampler = EvolutionSampler(
            pop_size=cfg.search.qas.ea_pop_size,
            n_gens=cfg.search.qas.ea_gens,
            n_layers=cfg.search.circuit.n_layers,
            n_blocks=metadata["NAS_search_qiskit_space_len"],
        )

        exact_value = calculate_exact_value(cfg, hamiltonian_numpy_eigen)
        sorted_result = sampler.sample(
            test_subnet_evolution_wrapper(
                estimator_func,
                model,
                cfg.search.qas.n_experts,
                transpile_info,
                exact_value,
            )
        )
        # result = sampler.subnet_eval_dict

        log.info("Completed running evolution")

        report(
            ExpertInfo(subnet, cfg.search.qas.n_experts),
            sorted_result,
            model,
            estimator_func,
            transpile_info,
        )

        # Compute the relative error between the expected ground state energy and the measured
        found_eigenvalue: float = sorted_result[0][1][1]
        log.info("Expected electronic ground state energy: %.10f", exact_value)
        log.info("Computed electronic ground state energy: %.10f", found_eigenvalue)
        log.info("Relative error: %.8f", rel_err(exact_value, found_eigenvalue))
        log.info("Relative error: %.8f %%", 100 * rel_err(exact_value, found_eigenvalue))


if __name__ == "__main__":
    main()  # pylint: disable = no-value-for-parameter
