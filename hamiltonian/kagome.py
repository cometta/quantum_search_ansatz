# pylint: skip-file
import rustworkx as rx
from qiskit.opflow.primitive_ops import PauliSumOp
from qiskit_nature.mappers.second_quantization import LogarithmicMapper
from qiskit_nature.problems.second_quantization.lattice import Lattice

from hamiltonian.heisenberg_model import HeisenbergModel
from utils import Singleton


class Kagome(metaclass=Singleton):
    def __init__(self) -> None:
        pass

    def create(self) -> PauliSumOp:
        # Kagome unit cell
        num_qubits = 16
        # Edge weight
        t = 1.0

        # Generate graph of kagome unit cell
        # Start by defining all the edges
        graph_16 = rx.PyGraph(multigraph=False)
        graph_16.add_nodes_from(range(num_qubits))
        edge_list = [
            (1, 2, t),
            (2, 3, t),
            (3, 5, t),
            (5, 8, t),
            (8, 11, t),
            (11, 14, t),
            (14, 13, t),
            (13, 12, t),
            (12, 10, t),
            (10, 7, t),
            (7, 4, t),
            (4, 1, t),
            (4, 2, t),
            (2, 5, t),
            (5, 11, t),
            (11, 13, t),
            (13, 10, t),
            (10, 4, t),
        ]
        # Generate graph from the list of edges
        graph_16.add_edges_from(edge_list)

        # Make a Lattice from graph
        kagome_unit_cell_16 = Lattice(graph_16)

        # Draw Lattice and include labels to check we exclude the right spins
        # Specify node locations for better visualizations
        kagome_pos = {
            0: [1, -1],
            6: [1.5, -1],
            9: [2, -1],
            15: [2.5, -1],
            1: [0, -0.8],
            2: [-0.6, 1],
            4: [0.6, 1],
            10: [1.2, 3],
            13: [0.6, 5],
            11: [-0.6, 5],
            5: [-1.2, 3],
            3: [-1.8, 0.9],
            8: [-1.8, 5.1],
            14: [0, 6.8],
            7: [1.8, 0.9],
            12: [1.8, 5.1],
        }
        kagome_unit_cell_16.draw(
            style={
                "with_labels": True,
                "font_color": "white",
                "node_color": "purple",
                "pos": kagome_pos,
            }
        )
        # plt.show()

        # Build Hamiltonian from graph edges
        heis_16 = HeisenbergModel.uniform_parameters(
            lattice=kagome_unit_cell_16,
            uniform_interaction=t,
            uniform_onsite_potential=0.0,  # No singe site external field
        )

        # Map from SpinOp to qubits just as before.
        log_mapper = LogarithmicMapper()
        ham_16 = 4 * log_mapper.map(heis_16.second_q_ops().simplify())
        # Print Hamiltonian to check it's what we expect:
        # 18 ZZ, 18 YY, and 18 XX terms over 16 qubits instead of over 12 qubits

        return ham_16
