""" Custom VQE """
from typing import Any, Callable, Optional

from nptyping import Float, NDArray
from qiskit import QuantumCircuit
from qiskit.algorithms import MinimumEigensolver, VQEResult
from qiskit.algorithms.list_or_dict import ListOrDict
from qiskit.algorithms.minimum_eigen_solvers import MinimumEigensolverResult
from qiskit.algorithms.optimizers import Optimizer
from qiskit.opflow import OperatorBase
from qiskit.primitives import BackendEstimator
from qiskit_aer.primitives import Estimator as Aer_Estimator


class CustomVQE(MinimumEigensolver):
    """Custom VQE"""

    def __init__(
        self,
        estimator: Aer_Estimator | BackendEstimator,
        circuit: QuantumCircuit,
        optimizer: Optimizer,
        params: NDArray[Any, Float],
        callback: Callable | None = None,
    ) -> None:
        self._estimator = estimator
        self._circuit = circuit
        self._optimizer = optimizer
        self._callback = callback
        self._params = params

    def compute_minimum_eigenvalue(
        self, operator: OperatorBase, aux_operators: Optional[ListOrDict[OperatorBase]] = None
    ) -> MinimumEigensolverResult:
        # Define objective function to classically minimize over
        # pylint: disable = invalid-name
        def objective(x):
            # Execute job with estimator primitive
            job = self._estimator.run([self._circuit], [operator], [x])
            # Get results from jobs
            est_result = job.result()
            # Get the measured energy value
            value = est_result.values[0]
            # Save result information using callback function
            if self._callback is not None:
                self._callback(value)
            return value

        x0 = self._params  # pylint: disable = invalid-name

        # Run optimization
        res = self._optimizer.minimize(objective, x0=x0)

        # Populate VQE result
        result = VQEResult()
        result.cost_function_evals = res.nfev
        result.eigenvalue = res.fun
        result.optimal_parameters = res.x
        return result
