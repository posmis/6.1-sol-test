"""FiPy 4.0.3 nonorthogonal diffusion without Python-bytecode-based naming.

The upstream method obtains an automatic name for ``var + 0`` while building its
gradient. FiPy's name disassembler does not understand CPython 3.14 opcodes.
Assigning a name preserves the same lazy arithmetic, gradients and correction.
"""
from fipy import DiffusionTermCorrection


class NamedDiffusionTermCorrection(DiffusionTermCorrection):
    def _calcAnisotropySource(self, coeff, var):
        if not hasattr(self, "anisotropySource") and len(coeff) > 1:
            from fipy.variables.addOverFacesVariable import _AddOverFacesVariable
            unconstrained = var+0
            unconstrained.name = "FlowDesk unconstrained field"
            gradients = unconstrained.grad.harmonicFaceValue.dot(self._getRotationTensor(var.mesh))
            self.anisotropySource = _AddOverFacesVariable(gradients[1:].dot(coeff[1:]))*var.mesh.cellVolumes
