# pylint: skip-file
from qiskit.opflow import I
from qiskit.opflow.primitive_ops import PauliSumOp
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.units import DistanceUnit

from utils import Singleton


class H2(metaclass=Singleton):
    def __init__(self) -> None:
        pass

    def create(self) -> PauliSumOp:
        driver = PySCFDriver(
            atom="H 0 0 0; H 0 0 0.735",
            basis="sto3g",
            charge=0,
            spin=0,
            unit=DistanceUnit.ANGSTROM,
        )
        problem = driver.run()

        mapper = JordanWignerMapper()
        hamiltonian_numpy_eigen = mapper.map(problem.hamiltonian.second_q_op())

        return hamiltonian_numpy_eigen
