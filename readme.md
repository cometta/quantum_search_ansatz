


## Quantum Ansatz Search for ground state energy with Evolution 
Find the best ansatz for ground state energy using genetic algorithm. Able to run on any Qiskit V1 fake device with and without noise. Capable to search bigger hibert space compared to original implemention on the paper. Tested up to 12 qubits Hamiltonian run on the 16 qubits device.

### How to use
 - install below packages
    ```
    sudo apt install cmake
    sudo apt-get install libopenblas-dev
    ```

 - create python 3.11 environment
 - `pip install -r requirements.txt`
 - add your problem in the directory `hamiltonian/<new problem>.py`
 - add your problem config file in the directory `conf/search/<new problem>.yaml`
 - edit `config.yaml` file, change the  `search` value based on your problem
 - run the search by `python main.py`
 - output are stored at `outputs/<date>/<time>/*`
   - `circuit_selected.qpy` Ansatz found is stored in Qiskit binary serializer format
   - `nas_result_sorted.txt` Best result is on the top, sorted in the format `[subnet] (score, electric ground state value)`
   - `param_selected.txt` best parameters found

### How to use the ansatz

    ```
    from qiskit import qpy
    ansatz_custom = qpy.load(open(os.path.join('<path>/circuit_selected.qpy'), 'rb'))[0] 
    ```

### Supported
 - `hamiltonian qubits < device qubits` supported, `transpile_q_layout` can be `None` or `List integer`

 - `hamiltonian qubits = device qubits` in config file, need to fill in `transpile_q_layout`

 - `hamiltonian qubits = device qubits and transpile_q_layout = None` known issue, does not supported



### Credits
```
@article{du2020quantum,
  title={Quantum circuit architecture search: error mitigation and trainability enhancement for variational quantum solvers},
  author={Du, Yuxuan and Huang, Tao and You, Shan and Hsieh, Min-Hsiu and Tao, Dacheng},
  journal={arXiv preprint arXiv:2010.10217},
  year={2020}
}
```
fork of https://github.com/yuxuan-du/Quantum_architecture_search