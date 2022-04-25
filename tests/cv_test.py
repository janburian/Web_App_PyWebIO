import os

import pytest
import detectron2_testovaci
from pathlib import Path

def test_cv():
    #fn = Path(__file__).parent / "0000.jpg"
    fn = Path(__file__).parent
    fn_out = Path(__file__).parent

    detectron2_testovaci.predict((fn), (fn_out))

    assert Path(fn_out / "output.jpg").exists() # testuje, zda "output.jpg" existuje

