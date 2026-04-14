from resonator_tools import circuit
import numpy as np
import pandas as pd

from _example_setup import data_path, ensure_repo_root_on_path

ensure_repo_root_on_path()

port1 = circuit.notch_port()
# port1 = circuit.reflection_port()
port1.add_froms2p(
    str(data_path("s2210dBmm.s1p")), 1, 2, "realimag", fdata_unit=1e9, delimiter=None
)
port1.GUIfit()