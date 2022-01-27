"""
empty
"""


class PolymerNetwork:
    """
    empty
    """

    def __init__(
        self,
        openmm_forcefield,
        initial_topology,
        group_a_select_string,
        group_b_select_string,
    ):
        self.openmm_forcefield = openmm_forcefield
        self.initial_topology = initial_topology
        self.group_a_select_string = group_a_select_string
        self.group_b_select_string = group_b_select_string
        self.u = PolymerNetwork._topology_to_universe(self.initial_topology)
        # self.group_a = self.u.select_atoms(group_a_select_string)
        # self.group_b = self.u.select_atoms(group_b_select_string)

    @staticmethod
    def _topology_to_universe(topology):
        return 1
