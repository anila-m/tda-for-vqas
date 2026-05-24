# Exploring Cost Landscapes of Variational Quantum Algorithms via Topological Data Analysis

Experiment/Code for reproduction of results for the research project "Exploring Cost Landscapes of Variational Quantum Algorithms via Topological Data Analysis" by Alina Mürwald (2026). 

It uses topological data analysis to analyze the structure of cost landscapes of Variational Quantum Algorithms (VQAs), like Quantum Neural Networks (QNNs) and the Quantum Approximate Optimizer Algorithm (QAOA). 

## Directory

All experiment results are available in the folder experiment_results.

All files needed to reproduce the results are available in the folder experiments. 


## Sources

The source code for the VQA cost landscapes were adapted from existing open-source projects and research:
* The **QNN** source code: Adapted from the repository accompanying the master thesis *Analyzing the Effect of Entanglement of Training Samples on the Loss Landscape of Quantum Neural Networks* of Ülger ([GitHub Repository](https://github.com/vic-it/master-thesis)).
* The **QAOA** source code: Adapted from the repository accompanying the paper *Connecting the Hamiltonian structure to the QAOA energy and Fourier landscape structure.* by Stęchły, et al. ([GitHub Repository](https://github.com/Boniface316/qaoa_landscape)).

The implementations of the roughness metrics (Total Variation, Absolute Scalar Curvature and Fourier Density) were taken from the master thesis *Analyzing the Effect of Entanglement of Training Samples on the Loss Landscape of Quantum Neural Networks* of Ülger ([GitHub Repository](https://github.com/vic-it/master-thesis)).

The Scikit-TDA and Giotto-tda libraries were used to compute persistence diagrams, landscapes and distances.

## Dependencies

The experiments were run using Python 3.10.11.
All required packages are detailed in `requirements_3.10.11.txt`. Install them by running the following command: ``pip install -r requirements_3.10.11.txt``.
The orqviz package has to be upgraded manually: ``pip install --upgrade --no-cache-dir --use-deprecated=legacy-resolver orqviz``.

## Disclaimer of Warranty

Unless required by applicable law or agreed to in writing, Licensor provides the Work (and each Contributor provides its
Contributions) on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied, including,
without limitation, any warranties or conditions of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
PARTICULAR PURPOSE. You are solely responsible for determining the appropriateness of using or redistributing the Work
and assume any risks associated with Your exercise of permissions under this License.

## Haftungsausschluss

Dies ist ein Forschungsprototyp. Die Haftung für entgangenen Gewinn, Produktionsausfall, Betriebsunterbrechung,
entgangene Nutzungen, Verlust von Daten und Informationen, Finanzierungsaufwendungen sowie sonstige Vermögens- und
Folgeschäden ist, außer in Fällen von grober Fahrlässigkeit, Vorsatz und Personenschäden, ausgeschlossen.
