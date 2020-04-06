import pickle

import numpy as np
import os

UNK_TOKEN = "<UNK>"
PAD_TOKEN = "<PAD>"
BOS_TOKEN = "시"
EOS_TOKEN = "끝"
SPLIT_TOKEN = "▁"

class G2pM(object):
    # one-layer bi-LSTM with two layered FC
    def __init__(self):
        self.cedict = pickle.load(open(os.path.dirname(os.path.abspath(__file__)) + '/digest_cedict.pkl', 'rb'))
        self.char2idx = pickle.load(open(os.path.dirname(os.path.abspath(__file__)) + '/char2idx.pkl', 'rb'))
        class2idx = pickle.load(open(os.path.dirname(os.path.abspath(__file__)) + '/class2idx.pkl', 'rb'))
        state_dict = pickle.load(open(os.path.dirname(os.path.abspath(__file__)) + '/np_ckpt.pkl', 'rb'))
        
        self.load_variable(state_dict)
        self.idx2class = {idx: pron for pron, idx in class2idx.items()}

    def load_variable(self, state_dict):
        self.embeddings = state_dict["embedding.weight"]
        self.weight_ih = state_dict["lstm.weight_ih_l0"]
        self.weight_hh = state_dict["lstm.weight_hh_l0"]
        self.bias_ih = state_dict["lstm.bias_ih_l0"]
        self.bias_hh = state_dict["lstm.bias_hh_l0"]

        self.weight_ih_reverse = state_dict["lstm.weight_ih_l0_reverse"]
        self.weight_hh_reverse = state_dict["lstm.weight_hh_l0_reverse"]
        self.bias_ih_reverse = state_dict["lstm.bias_ih_l0_reverse"]
        self.bias_hh_reverse = state_dict["lstm.bias_hh_l0_reverse"]

        self.hidden_weight_l0 = state_dict["logit_layer.0.weight"]
        self.hidden_bias_l0 = state_dict["logit_layer.0.bias"]
        self.hidden_weight_l1 = state_dict["logit_layer.2.weight"]
        self.hidden_bias_l1 = state_dict["logit_layer.2.bias"]

    def reverse_sequence(self, inputs, lengths):
        # inputs : numpy 2d array
        # lengths: list
        rev_seqs = []
        max_length = max(lengths)
        is_1d = len(inputs.shape) == 2
        for seq, length in zip(inputs, lengths):
            end = length-1

            if is_1d:
                zeros = np.array([0] * (max_length-length), dtype=np.int32)
            else:
                d = inputs.shape[-1]
                zeros = np.zeros((max_length-length, d), dtype=np.float64)
            rev_seq = np.concatenate([seq[end::-1], zeros], axis=0)
            rev_seqs.append(rev_seq)
        rev_seqs = np.stack(rev_seqs, axis=0)

        return rev_seqs

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def relu(self, x):
        return x * (x > 0)

    def get_embedding(self, inputs):
        return np.take(self.embeddings, inputs, axis=0)

    def fw_lstm_cell(self, inputs, init_states=None):
        # inputs : [b,d]
        if init_states is None:
            init_h = np.zeros((inputs.shape[0], self.weight_hh.shape[1]))
            init_c = np.zeros_like(init_h)
            init_states = (init_h, init_c)
        states = init_states
        (prev_h, prev_c) = states
        ifgo_ih = np.matmul(
            inputs, self.weight_ih.T) + self.bias_ih
        ifgo_hh = np.matmul(
            prev_h, self.weight_hh.T) + self.bias_hh

        if_ih = ifgo_ih[:, :ifgo_ih.shape[-1] * 2 // 4]
        go_ih = ifgo_ih[:, ifgo_ih.shape[-1] * 2 // 4:]

        if_hh = ifgo_hh[:, :ifgo_hh.shape[-1] * 2 // 4]
        go_hh = ifgo_hh[:, ifgo_hh.shape[-1] * 2 // 4:]

        if_gate = self.sigmoid(if_ih + if_hh)

        i, f = if_gate[:, :if_gate.shape[-1] //
                       2], if_gate[:, if_gate.shape[-1] // 2:]

        g_ih, o_ih = go_ih[:, :go_ih.shape[-1] //
                           2], go_ih[:, go_ih.shape[-1] // 2:]
        g_hh, o_hh = go_hh[:, :go_hh.shape[-1] //
                           2], go_hh[:, go_hh.shape[-1] // 2:]

        g = np.tanh(g_ih + g_hh)
        o = self.sigmoid(o_ih + o_hh)
        c = f * prev_c + i * g
        h = o * np.tanh(c)

        return (h, c)

    def bw_lstm_cell(self, inputs, init_states=None):
        # inputs : [b,d]
        if init_states is None:
            init_h = np.zeros((inputs.shape[0], self.weight_hh.shape[1]))
            init_c = np.zeros_like(init_h)
            init_states = (init_h, init_c)
        states = init_states
        (prev_h, prev_c) = states
        ifgo_ih = np.matmul(
            inputs, self.weight_ih_reverse.T) + self.bias_ih_reverse
        ifgo_hh = np.matmul(
            prev_h, self.weight_hh_reverse.T) + self.bias_hh_reverse

        if_ih = ifgo_ih[:, :ifgo_ih.shape[-1] * 2 // 4]
        go_ih = ifgo_ih[:, ifgo_ih.shape[-1] * 2 // 4:]

        if_hh = ifgo_hh[:, :ifgo_hh.shape[-1] * 2 // 4]
        go_hh = ifgo_hh[:, ifgo_hh.shape[-1] * 2 // 4:]

        if_gate = self.sigmoid(if_ih + if_hh)
        i, f = if_gate[:, :if_gate.shape[-1] //
                       2:], if_gate[:, if_gate.shape[-1] // 2:]

        g_ih, o_ih = go_ih[:, :go_ih.shape[-1] //
                           2], go_ih[:, go_ih.shape[-1] // 2:]
        g_hh, o_hh = go_hh[:, :go_hh.shape[-1] //
                           2], go_hh[:, go_hh.shape[-1] // 2:]

        g = np.tanh(g_ih + g_hh)
        o = self.sigmoid(o_ih + o_hh)
        c = f * prev_c + i * g
        h = o * np.tanh(c)

        return (h, c)

    def fc_layer(self, inputs):
        # inputs : [b,t,d]
        hidden_l0 = np.matmul(
            inputs, self.hidden_weight_l0.T) + self.hidden_bias_l0
        hidden_l0 = self.relu(hidden_l0)

        logits = np.matmul(
            hidden_l0, self.hidden_weight_l1.T) + self.hidden_bias_l1

        return logits

    def predict(self, inputs, target_idx):
        lengths = np.sum(np.sign(inputs), axis=1)
        max_length = max(lengths)

        rev_seq = self.reverse_sequence(inputs, lengths)
        fw_emb = self.get_embedding(inputs)  # [b,t,d]
        bw_emb = self.get_embedding(rev_seq)

        fw_states, bw_states = None, None
        fw_hs = []
        bw_hs = []
        for i in range(max_length):
            fw_input = fw_emb[:, i, :]
            bw_input = bw_emb[:, i, :]
            fw_states = self.fw_lstm_cell(fw_input, fw_states)
            bw_states = self.bw_lstm_cell(bw_input, bw_states)

            fw_hs.append(fw_states[0])
            bw_hs.append(bw_states[0])
        fw_hiddens = np.stack(fw_hs, axis=1)
        bw_hiddens = np.stack(bw_hs, axis=1)
        bw_hiddens = self.reverse_sequence(bw_hiddens, lengths)

        outputs = np.concatenate([fw_hiddens, bw_hiddens], axis=2)  # [b,t,d]
        batch_size = outputs.shape[0]
        if batch_size == 1:
            outputs = np.squeeze(outputs, axis=0)  # [t,d]
            target_hidden = outputs[target_idx]
        else:
            target_hidden = outputs[np.arange(len(lengths)), target_idx]

        logits = self.fc_layer(target_hidden)
        preds = np.argmax(logits, axis=1)

        return preds

    def __call__(self, sent):
        input_ids = []
        poly_indices = []
        pros_lst = []
        for idx, char in enumerate(sent):
            if char in self.char2idx:
                char_id = self.char2idx[char]
            else:
                char_id = self.char2idx[UNK_TOKEN]
            input_ids.append(char_id)

            if char in self.cedict:
                prons = self.cedict[char]

                # polyphonic character
                if len(prons) > 1:
                    poly_indices.append(idx)
                    pros_lst.append(SPLIT_TOKEN)
                else:
                    pros_lst.append(prons[0])
            else:
                pros_lst.append(char)

        if len(poly_indices) > 0:
            # insert and append BOS, EOS ID
            BOS_ID = self.char2idx[BOS_TOKEN]
            EOS_ID = self.char2idx[EOS_TOKEN]
            input_ids.insert(0, BOS_ID)
            input_ids.append(EOS_ID)
            # BOS_ID is inserted at the first position, so +1 for poly idx
            _poly_indices = [idx + 1 for idx in poly_indices]

            input_ids = np.array(input_ids, dtype=np.int32)
            input_ids = np.expand_dims(input_ids, axis=0)
            # input_ids = np.tile(input_ids, (len(poly_indices), 1))
            # polyphone disambiguation
            preds = self.predict(input_ids, _poly_indices)

            for idx, pred in zip(poly_indices, preds):
                pron = self.idx2class[pred]
                pros_lst[idx] = pron

        pron_str = ""
        for pro in pros_lst:
            if len(pro) == 1:
                pron_str += pro
            else:
                if len(pron_str) > 0 and pron_str[-1] != "-":
                    pro = "-" + pro
                pron_str += pro + "-"
        if pron_str[-1] == "-":
            pron_str = pron_str[:-1]
        ret = pron_str.split("-")
        
        return ret
