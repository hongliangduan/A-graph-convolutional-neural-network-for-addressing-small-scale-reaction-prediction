import sys
sys.path.append("../rexgen_direct/")

import tensorflow as tf
from rexgen_direct.core_wln_global.nn import linearND, linear
from rexgen_direct.core_wln_global.models import *
from rexgen_direct.core_wln_global.ioutils_direct import *
import math, sys, random
from collections import Counter
from optparse import OptionParser
from functools import partial
import threading
from multiprocessing import Queue
import os

'''
This module defines the DirectCoreFinder class, which is for deploying the core finding model
'''

NK3 = 80
batch_size = 2 # just fake it, make two 
hidden_size = 300
depth = 3
model_path = os.path.join(os.path.dirname(__file__), "model-300-3-direct/model.ckpt-140000")
from rexgen_direct.core_wln_global.mol_graph import atom_feature_dimension as adim, bond_feature_dimension as bdim, max_nb, smiles2graph_list as _s2g
smiles2graph_batch = partial(_s2g, idxfunc=lambda x:x.GetIntProp('molAtomMapNumber') - 1)

class DirectCoreFinder():
    def __init__(self, hidden_size=hidden_size, batch_size=batch_size, 
            depth=depth):
        self.hidden_size = hidden_size 
        self.batch_size = batch_size 
        self.depth = depth 

    def load_model(self, model_path=model_path):
        hidden_size = self.hidden_size 
        vbatch_size = self.batch_size 
        depth = self.depth 

        self.graph = tf.Graph()
        with self.graph.as_default():
            input_atom = tf.placeholder(tf.float32, [batch_size, None, adim])
            input_bond = tf.placeholder(tf.float32, [batch_size, None, bdim])
            atom_graph = tf.placeholder(tf.int32, [batch_size, None, max_nb, 2])
            bond_graph = tf.placeholder(tf.int32, [batch_size, None, max_nb, 2])
            num_nbs = tf.placeholder(tf.int32, [batch_size, None])
            node_mask = tf.placeholder(tf.float32, [batch_size, None])
            self.src_holder = [input_atom, input_bond, atom_graph, bond_graph, num_nbs, node_mask]
            self.label = tf.placeholder(tf.int32, [batch_size, None])
            self.binary = tf.placeholder(tf.float32, [batch_size, None, None, binary_feature_dimension])

            node_mask = tf.expand_dims(node_mask, -1)

            graph_inputs = (input_atom, input_bond, atom_graph, bond_graph, num_nbs, node_mask)
            # input_atom: (2, 32, 82)

            with tf.variable_scope("encoder"):
                atom_hiddens, _ = rcnn_wl_last(graph_inputs, batch_size=batch_size, hidden_size=hidden_size, depth=depth)

            atom_hiddens1 = tf.reshape(atom_hiddens, [batch_size, 1, -1, hidden_size])
            atom_hiddens2 = tf.reshape(atom_hiddens, [batch_size, -1, 1, hidden_size])
            atom_pair = atom_hiddens1 + atom_hiddens2  #(2, 32, 32, 300)

            attention_hidden = tf.nn.relu(linearND(atom_pair, hidden_size, scope="att_atom_feature", init_bias=None) +
                                          linearND(self.binary, hidden_size, scope="att_bin_feature"))
            attention_score = linearND(attention_hidden, 1, scope="att_scores")
            attention_score = tf.nn.sigmoid(attention_score) #(2, 32, 32, 1)

            att_context = attention_score * atom_hiddens1
            att_context = tf.reduce_sum(att_context, 2)

            attention_context1 = tf.reshape(att_context, [batch_size, 1, -1, hidden_size])
            attention_context2 = tf.reshape(att_context, [batch_size, -1, 1, hidden_size])
            attention_pair = attention_context1 + attention_context2


            pair_hidden0 = linearND(atom_pair, hidden_size, scope="atom_feature", init_bias=None) + \
                           linearND(self.binary, hidden_size, scope="bin_feature", init_bias=None) + \
                           linearND(attention_pair, hidden_size, scope="ctx_feature")
            # attention_pair is added here. pair_hidden0: (2, 32, 32, 300)    binary: (2, 32, 32, 10)
            pair_hidden = tf.nn.relu(pair_hidden0)
            pair_hidden = tf.reshape(pair_hidden, [batch_size, -1, hidden_size])  #(2, 1024, 300) 1024=32*32

            score = linearND(pair_hidden, 5, scope="scores")  #(2, 1024, 5), 300---->5
            score = tf.reshape(score, [batch_size, -1])   #(2, 5120)  5120=1024*5
            bmask = tf.to_float(tf.equal(self.label, INVALID_BOND)) * 10000

            topk_scores, topk = tf.nn.top_k(score - bmask, k=NK3)
            label_dim = tf.shape(self.label)[1]
            
            # What will be used for inference?
            self.predict_vars = [topk, topk_scores, label_dim, attention_score]
            self.predict_vars_two = [score, pair_hidden0, pair_hidden, attention_pair, atom_hiddens, atom_hiddens1, atom_hiddens2, atom_pair, input_atom, atom_graph]
            
            # Restore
            self.session = tf.Session()
            saver = tf.train.Saver()
            saver.restore(self.session, model_path)
        
    def predict(self, reactants_smi):

        bo_to_index  = {0.0: 0, 1.0:1, 2.0:2, 3.0:3, 1.5:4}
        bindex_to_o = {val:key for key, val in bo_to_index.items()}
        nbos = len(bo_to_index)

        src_batch, edit_batch = [], []
        mol = Chem.MolFromSmiles(reactants_smi)

        if any(not a.HasProp('molAtomMapNumber') for a in mol.GetAtoms()):
            mapnum = 1
            for a in mol.GetAtoms():
                a.SetIntProp('molAtomMapNumber', mapnum)
                mapnum += 1
        react = Chem.MolToSmiles(mol)

        src_batch.append(react)
        src_batch.append(react)
        edit_batch.append('0-1-0.0') # dummy edits
        edit_batch.append('0-1-0.0') # dummy edits

        src_tuple = smiles2graph_batch(src_batch)
        cur_bin, cur_label, sp_label = get_all_batch(zip(src_batch, edit_batch))
        feed_map = {x:y for x,y in zip(self.src_holder, src_tuple)}
        feed_map.update({self.label:cur_label, self.binary:cur_bin})

        cur_topk, cur_score, cur_dim, cur_att_score = self.session.run(self.predict_vars, feed_dict=feed_map)
        score, pair_hidden0, cur_pair_hidden,  cur_att_pair, atom_hiddens, atom_hiddens1, atom_hiddens2, atom_pair, input_atom, atom_graph = self.session.run(self.predict_vars_two,
            feed_dict=feed_map)
        binary = self.session.run(self.binary, feed_dict=feed_map)

        print("**************************************************************************************************")
        print("score:")
        print(score.shape) #(2, 5120)   5120=32*32*5
        print(score)
        print("**************************************************************************************************")

        print("input_atom:")
        print(input_atom.shape)
        print(input_atom[0][0])
        print(input_atom[0][1])
        print(input_atom)

        print("atom_graph:")
        print(atom_graph.shape)
        print(atom_graph)

        print("binary:")
        print(binary.shape) #(2, 32, 32, 10)
        print(binary)

        print("pair_hidden0:")
        print(pair_hidden0.shape)
        print(pair_hidden0)

        print("cur_pair_hidden:")
        print(cur_pair_hidden.shape) #(2, 1024, 300)
        print(cur_pair_hidden)
        print("**************************************************************************************************")

        print("atom_hiddens:")
        print(atom_hiddens.shape)
        print(atom_hiddens)
        print("atom_hiddens1:")
        print(atom_hiddens1.shape)
        print(atom_hiddens1)
        print("atom_hiddens2:")
        print(atom_hiddens2.shape)
        print(atom_hiddens2)

        print("atom_pair:")
        print(atom_pair.shape)
        print(atom_pair)

        print("cur_attention_pair:")
        print(cur_att_pair.shape)  #(2, 32, 32, 300)
        print(cur_att_pair)
        print("**************************************************************************************************")

        print("cur_dim:")
        print(cur_dim)  # 5120
        print("**************************************************************************************************")

        print("cur_attention_score:")
        print(cur_att_score.shape)  # (2, 32, 32, 1)
        print(cur_att_score)
        print("**************************************************************************************************")

        cur_dim = int(math.sqrt(cur_dim/5)) # important! changed to divide by 5      5120---->32

        cur_topk = cur_topk[0,:]
        cur_score = cur_score[0]
        cur_att_score = cur_att_score[0, :, :]

        bond_preds = []
        bond_scores = []

        # NOTE: we don't filter out molecules known to be reagents, but during training, 
        # molecules known to be reagents/solvents are not allowed to be involved with bond
        # changes.

        print("cur_topk are: ")
        print(cur_topk.shape)
        print(cur_topk)



        for j in range(NK3):
            k = cur_topk[j]
            bindex = k % nbos   # nbos=5
            y = ((k - bindex) / nbos) % cur_dim + 1  # cur_dim=32
            x = (k - bindex - (y-1) * nbos) / cur_dim / nbos + 1


            if x < y: # keep canonical
                # x = k / cur_dim + 1 # was for 2D case
                # y = k % cur_dim + 1 # was for 2D case
                bo = bindex_to_o[bindex]
                bond_preds.append("{}-{}-{:.1f}".format(x, y, bo))
                bond_scores.append(cur_score[j])

        return (react, bond_preds, bond_scores, cur_att_score)

if __name__ == '__main__':
    directcorefinder = DirectCoreFinder()
    directcorefinder.load_model()
    react = '[F:1][C:2]([C:3](=[C:4]([F:5])[F:6])[F:7])([F:8])[F:9].[H:10][H:11]'
    react, bond_preds, bond_scores, cur_att_score = directcorefinder.predict(react)

    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print(react)
    print("bond_predicts:")
    print(bond_preds)  # 40
    print("bond_scores:")
    print(bond_scores)  # 40
    print("cur_attention_score:")
    print(cur_att_score) # (11, 11) 11 atoms.