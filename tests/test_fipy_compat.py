from pathlib import Path
import dis
import pytest
from fipy import CellVariable, DiffusionTermCorrection
from flowdesk.meshing import MeshProject
from flowdesk.unstructured import fipy_mesh, system
from flowdesk.solver import Config, Solver


def test_nonorthogonal_assembly_does_not_decompile_python_bytecode(monkeypatch):
    path = Path(__file__).resolve().parent.parent/"examples/channel.flowmesh"
    mesh, _ = fipy_mesh(MeshProject.load(path))
    original = dis.get_instructions
    def unsupported_new_opcode(code, *args, **kwargs):
        if "/fipy/variables/" in code.co_filename.replace("\\", "/"):
            raise SyntaxError("Unknown instruction: LOAD_FAST_BORROW_LOAD_FAST_BORROW")
        return original(code, *args, **kwargs)
    monkeypatch.setattr(dis, "get_instructions", unsupported_new_opcode)
    # The upstream implementation reproduces the failure before solving anything.
    with pytest.raises(SyntaxError, match="Unknown instruction"):
        system(-DiffusionTermCorrection(coeff=1.), CellVariable(mesh=mesh, value=1.))
    # FlowDesk retains the same nonorthogonal correction without invoking the parser.
    result = Solver(Config(scenario="custom", mesh_path=str(path), mode="steady",
                           max_iterations=10)).run()
    assert result.diagnostics["iterations"] == 10
    assert result.diagnostics["mass_imbalance"] < 1e-7
