"""
Runs the reliability systematic step. Runs current framework
on prior datasets to ensure output is still the same/consistent
with the original.
"""

from tahoe100m_reliability.lib.step.base_systematic_step import BaseSystematicStep

class RegressionSystematicStep(BaseSystematicStep):
	pass