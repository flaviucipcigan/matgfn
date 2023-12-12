# MOF
from pymatgen.core.structure import Structure
import pormake as pm
pm.log.logger.setLevel(pm.log.logging.CRITICAL)
import os

# Standard library
import io
import tempfile
import random

# User Pormake to create a Pymatgen Structure
class PormakeStructureBuilder():
    def __init__(self, topology_string, include_edges = True, block_rmsd_cutoff=0.3):
        self.database = pm.Database()
        self.builder = pm.Builder()
        self.topology = self.database.get_topo(topology_string)

        self.token_vocabulary = self.database._get_bb_list()

        if include_edges==True:
            self.token_vocabulary.append('none')

        self.termination_token = "[TER]"

        if block_rmsd_cutoff is not None:
            self.mask = MOFMask(topology_string, self.token_vocabulary, include_edges, block_rmsd_cutoff)
        else:
            self.mask = None

        self.n_nodes = len(self.topology.unique_cn)

        if include_edges==False:
            self.n_slots = self.n_nodes
            self.unique_edge_types=[]
        else:
            self.n_slots = self.n_nodes + self.topology.n_edge_types
            self.unique_edge_types=self.topology.unique_edge_types        

    def random_sequence(self):
        sequence = []
        for i in range(0, self.n_slots):
            allowed_building_blocks = [self.token_vocabulary[a] for a in self.mask.forward_actions_at_each_slot[i]]
            sequence.append(random.choice(allowed_building_blocks))

        return sequence

    def make_pormake_mof(self, sequence):
        # Remove termination token if it's there
        if sequence[-1] == self.termination_token:
            sequence = sequence[:-1]
        
        # Split the sequence into nodes and edges
        edges={}

        ## we don't have any edges
        if len(sequence) == self.n_nodes:
            nodes = sequence
        ## we have filled all the edge slots
        else:
            # nodes are first in the sequence, then edges, then TER
            offset = len(self.unique_edge_types) 
            nodes = sequence[:-offset]

            edge_bb_names = sequence[self.n_nodes:]
            edge_connections_to_delete=[]

            for i in range(len(self.unique_edge_types)):
                edge_connections = (self.unique_edge_types[i][0],self.unique_edge_types[i][1])
                edges[edge_connections] = edge_bb_names[i] 

                if edge_bb_names[i]=='none':
                    edge_connections_to_delete.append(edge_connections)

            for item in edge_connections_to_delete:
                del edges[item]

        return self._make_pormake_mof(nodes, edges)

    def _make_pormake_mof(self, nodes, edges):
        node_bbs=[]

        for node in nodes:
            node_bbs.append(self.database.get_bb(node))

        for key in edges:
            edges[key] = self.database.get_building_block(edges[key])

        return self.builder.build_by_type(topology = self.topology, node_bbs = node_bbs, edge_bbs=edges)
    
    def make_cif(self, sequence):
        mof = self.make_pormake_mof(sequence)
        cif_string = self.mof_to_cif_string(mof)

        return cif_string
    
    def make_structure(self, sequence):
        cif_string = self.make_cif(sequence)
        return Structure.from_str(cif_string, fmt="cif")

    # Adapted from framework.write_cif in pormake
    # We only write symmetry information and atom coordinates
    def mof_to_cif_string(self,mof):
        f=io.StringIO("")

        f.write("data_{}\n")

        f.write("_symmetry_space_group_name_H-M    P1\n")
        f.write("_symmetry_Int_Tables_number       1\n")
        f.write("_symmetry_cell_setting            triclinic\n")

        f.write("loop_\n")
        f.write("_symmetry_equiv_pos_as_xyz\n")
        f.write("'x, y, z'\n")

        a, b, c, alpha, beta, gamma = \
            mof.atoms.get_cell_lengths_and_angles()

        f.write("_cell_length_a     {:.3f}\n".format(a))
        f.write("_cell_length_b     {:.3f}\n".format(b))
        f.write("_cell_length_c     {:.3f}\n".format(c))
        f.write("_cell_angle_alpha  {:.3f}\n".format(alpha))
        f.write("_cell_angle_beta   {:.3f}\n".format(beta))
        f.write("_cell_angle_gamma  {:.3f}\n".format(gamma))

        f.write("loop_\n")
        f.write("_atom_site_label\n")
        f.write("_atom_site_type_symbol\n")
        f.write("_atom_site_fract_x\n")
        f.write("_atom_site_fract_y\n")
        f.write("_atom_site_fract_z\n")
        f.write("_atom_type_partial_charge\n")

        symbols = mof.atoms.symbols
        frac_coords = mof.atoms.get_scaled_positions()
        for i, (sym, pos) in enumerate(zip(symbols, frac_coords)):
            label = "{}{}".format(sym, i)
            f.write("{} {} {:.5f} {:.5f} {:.5f} 0.0\n".
                    format(label, sym, *pos))

        return f.getvalue()


class MOFMask():

    def __init__(self,topology_name,vocab,include_edges, block_rmsd_cutoff):
        self.database = pm.Database()
        self.topology_name=topology_name
        self.topology=self.database.get_topology(self.topology_name)
        self.vocab=vocab

        self.connection_points_at_each_position=self.topology.unique_cn
        self.n_nodes=len(self.connection_points_at_each_position)

        self.local_structures = self.topology.unique_local_structures
        self.locator = pm.Locator()

        self.include_edges=include_edges
        self.block_rmsd_cutoff=block_rmsd_cutoff

        if self.include_edges==False:
            self.n_slots=self.n_nodes

        else:
            self.n_slots=self.n_nodes + self.topology.n_edge_types
            self.edge_vocab=[]

            for i in range(len(self.vocab)):

                if self.vocab[i]=='none':
                    self.edge_vocab.append(i)

                else:

                    bb=self.database.get_building_block(self.vocab[i])

                    if bb.n_connection_points==2:
                        self.edge_vocab.append(i)

        self._get_forward_actions_for_all_slots()

    # calculate allowed forward actions for each slot of the topology
    # this is called once, just after initialisation of MOFMask
    def _get_forward_actions_for_all_slots(self):

        self.forward_actions_at_each_slot=[]

        for position in range(self.n_nodes):

            allowed_actions = []

            for i in range(len(self.vocab)):

                if self.vocab[i] != 'none':
                    bb=self.database.get_building_block(self.vocab[i])

                    if bb.n_connection_points==self.connection_points_at_each_position[position]:
                        structure=self.local_structures[position]
                        RMSD = self.locator.calculate_rmsd(structure,bb)

                        if RMSD < self.block_rmsd_cutoff:
                            allowed_actions.append(i)

            self.forward_actions_at_each_slot.append(allowed_actions)

        if self.include_edges==True:
            for i in range(self.topology.n_edge_types):
                self.forward_actions_at_each_slot.append(self.edge_vocab)

        return self.forward_actions_at_each_slot
    
    # this is called at every step of building a MOF
    def allowed_forward_actions(self,sequence):

        position=len(sequence)

        return self.forward_actions_at_each_slot[position]
