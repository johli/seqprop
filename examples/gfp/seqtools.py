import numpy as np


class SequenceTools(object):
    dna2gray_ = {'c': (0, 0), 't': (1, 0), 'g': (1, 1), 'a': (0, 1)}
    gray2dna_ = {(0, 0): 'c', (1, 0): 't', (1, 1): 'g', (0, 1): 'a'}

    codon2protein_ = {'ttt': 'f', 'ttc': 'f', 'tta': 'l', 'ttg': 'l', 'tct': 's', 'tcc': 's', 'tca': 's',
                      'tcg': 's', 'tat': 'y', 'tac': 'y', 'taa': '!', 'tag': '!', 'tgt': 'c', 'tgc': 'c',
                      'tga': '!', 'tgg': 'w', 'ctt': 'l', 'ctc': 'l', 'cta': 'l', 'ctg': 'l', 'cct': 'p',
                      'ccc': 'p', 'cca': 'p', 'ccg': 'p', 'cat': 'h', 'cac': 'h', 'caa': 'q', 'cag': 'q',
                      'cgt': 'r', 'cgc': 'r', 'cga': 'r', 'cgg': 'r', 'att': 'i', 'atc': 'i', 'ata': 'i',
                      'atg': 'm', 'act': 't', 'acc': 't', 'aca': 't', 'acg': 't', 'aat': 'n', 'aac': 'n',
                      'aaa': 'k', 'aag': 'k', 'agt': 's', 'agc': 's', 'aga': 'r', 'agg': 'r', 'gtt': 'v',
                      'gtc': 'v', 'gta': 'v', 'gtg': 'v', 'gct': 'a', 'gcc': 'a', 'gca': 'a', 'gcg': 'a',
                      'gat': 'd', 'gac': 'd', 'gaa': 'e', 'gag': 'e', 'ggt': 'g', 'ggc': 'g', 'gga': 'g',
                      'ggg': 'g'}

    protein2codon_ = {
        'l': ['tta', 'ttg', 'ctt', 'ctc', 'cta', 'ctg'],
        's': ['tct', 'tcc', 'tca', 'tcg', 'agt', 'agc'],
        'r': ['cgt', 'cgc', 'cga', 'cgg', 'aga', 'agg'],
        'v': ['gtt', 'gtc', 'gta', 'gtg'],
        'a': ['gct', 'gcc', 'gca', 'gcg'],
        'p': ['cct', 'ccc', 'cca', 'ccg'],
        't': ['act', 'acc', 'aca', 'acg'],
        'g': ['ggt', 'ggc', 'gga', 'ggg'],
        'stop': ['taa', 'tag', 'tga'],
        'i': ['att', 'atc', 'ata'],
        'y': ['tat', 'tac'],
        'f': ['ttt', 'ttc'],
        'c': ['tgt', 'tgc'],
        'h': ['cat', 'cac'],
        'q': ['caa', 'cag'],
        'n': ['aat', 'aac'],
        'k': ['aaa', 'aag'],
        'd': ['gat', 'gac'],
        'e': ['gaa', 'gag'],
        'w': ['tgg'],
        'm': ['atg']
    }

    protein2constraint_ = {
        'l': {(1,): {('t',)}, (0, 2): {('t', 'a'), ('t', 'g'), ('c', 't'), ('c', 'c'), ('c', 'a'), ('c', 'g')}},
        's': {(0, 1, 2): {('t', 'c', 't'), ('t', 'c', 'c'), ('t', 'c', 'a'), ('t', 'c', 'g'), ('a', 'g', 't'),
                          ('a', 'g', 'c')}},
        'r': {(1,): {('g',)}, (0, 2): {('c', 't'), ('c', 'c'), ('c', 'a'), ('c', 'g'), ('a', 'a'), ('a', 'g')}},
        'v': {(0,): {('g',)}, (1,): {('t',)}, (2,): {('g',), ('t',), ('a',), ('c',)}},
        'a': {(0,): {('g',)}, (1,): {('c',)}, (2,): {('g',), ('t',), ('a',), ('c',)}},
        'p': {(0,): {('c',)}, (1,): {('c',)}, (2,): {('g',), ('t',), ('a',), ('c',)}},
        't': {(0,): {('a',)}, (1,): {('c',)}, (2,): {('g',), ('t',), ('a',), ('c',)}},
        'g': {(0,): {('g',)}, (1,): {('g',)}, (2,): {('g',), ('t',), ('a',), ('c',)}},
        'stop': {(0,): {('t',)}, (1, 2): {('a', 'a'), ('a', 'g'), ('g', 'a')}},
        'i': {(0,): {('a',)}, (1,): {('t',)}, (2,): {('t',), ('a',), ('c',)}},
        'y': {(0,): {('t',)}, (1,): {('a',)}, (2,): {('t',), ('c',)}},
        'f': {(0,): {('t',)}, (1,): {('t',)}, (2,): {('t',), ('c',)}},
        'c': {(0,): {('t',)}, (1,): {('g',)}, (2,): {('t',), ('c',)}},
        'h': {(0,): {('c',)}, (1,): {('a',)}, (2,): {('t',), ('c',)}},
        'q': {(0,): {('c',)}, (1,): {('a',)}, (2,): {('a',), ('g',)}},
        'n': {(0,): {('a',)}, (1,): {('a',)}, (2,): {('t',), ('c',)}},
        'k': {(0,): {('a',)}, (1,): {('a',)}, (2,): {('a',), ('g',)}},
        'd': {(0,): {('g',)}, (1,): {('a',)}, (2,): {('t',), ('c',)}},
        'e': {(0,): {('g',)}, (1,): {('a',)}, (2,): {('a',), ('g',)}},
        'w': {(0,): {('t',)}, (1,): {('g',)}, (2,): {('g',)}},
        'm': {(0,): {('a',)}, (1,): {('t',)}, (2,): {('g',)}},
    }

    # Integer mapping from Fernandes and Vinga (2016)
    codon2idx_ = {'aaa': 1, 'aac': 2, 'aag': 3, 'aat': 4, 'aca': 5, 'acc': 6, 'acg': 7, 'act': 8, 'aga': 9,
                  'agc': 10, 'agg': 11, 'agt': 12, 'ata': 13, 'atc': 14, 'atg': 15, 'att': 16, 'caa': 17,
                  'cac': 18, 'cag': 19, 'cat': 20, 'cca': 21, 'ccc': 22, 'ccg': 23, 'cct': 24, 'cga': 25,
                  'cgc': 26, 'cgg': 27, 'cgt': 28, 'cta': 29, 'ctc': 30, 'ctg': 31, 'ctt': 32, 'gaa': 33,
                  'gac': 34, 'gag': 35, 'gat': 36, 'gca': 37, 'gcc': 38, 'gcg': 39, 'gct': 40, 'gga': 41,
                  'ggc': 42, 'ggg': 43, 'ggt': 44, 'gta': 45, 'gtc': 46, 'gtg': 47, 'gtt': 48, 'taa': 49,
                  'tac': 50, 'tag': 51, 'tat': 52, 'tca': 53, 'tcc': 54, 'tcg': 55, 'tct': 56, 'tga': 57,
                  'tgc': 58, 'tgg': 59, 'tgt': 60, 'tta': 61, 'ttc': 62, 'ttg': 63, 'ttt': 64}

    @staticmethod
    def convert_dna_to_rna(seq):
        dna2rna = {'t': 'u', 'a': 'a', 'g': 'g', 'c': 'c'}
        return "".join([dna2rna[s] for s in seq])

    @staticmethod
    def convert_dna_arr_to_str(dna_arr, base_order='ATCG'):
        """ Convert N x 4 tokenized array into length N string """
        dna_seq_str = ''
        for i in range(dna_arr.shape[0]):
            token = np.argmax(dna_arr[i, :])
            dna_seq_str += base_order[token]
        return dna_seq_str

    @staticmethod
    def get_aa_codons():
        aa_list = sorted(list(SequenceTools.protein2codon_.keys()))
        aa_codons = np.zeros((len(aa_list), 6, 3, 4))
        i = 0
        for aa in aa_list:
            cods = SequenceTools.protein2codon_[aa]
            j = 0
            for c in cods:
                cod_arr = SequenceTools.convert_dna_str_to_arr(c)
                aa_codons[i, j] = cod_arr
                j += 1
            i += 1
        return aa_codons

    @staticmethod
    def convert_dna_str_to_arr(dna_str, base_order='ATCG'):
        """ Convert length N string into N x 4 tokenized array"""
        dna_str = dna_str.upper()
        N = len(dna_str)
        dna_arr = np.zeros((N, 4))
        for i in range(N):
            idx = base_order.index(dna_str[i])
            dna_arr[i, idx] = 1.
        return dna_arr

    @staticmethod
    def convert_dna_arr_to_gray(dna_arr, base_order='ATCG'):
        """ Convert N x 4 tokenized array into 2N x 2 tokenized gray code array"""
        N = dna_arr.shape[0]
        gray_arr = np.zeros((2 * N, 2))
        for i in range(N):
            token = np.argmax(dna_arr[i, :])
            dna_i = base_order[token]
            gray_i = SequenceTools.dna2gray_[dna_i]
            for j in range(2):
                gray_arr[2 * i + j, gray_i[j]] = 1
        return gray_arr

    @staticmethod
    def convert_gray_to_dna_str(gray_arr):
        Ngray = gray_arr.shape[0]
        dna_str = ''
        i = 0
        while i < Ngray:
            g1 = int(np.argmax(gray_arr[i, :]))
            g2 = int(np.argmax(gray_arr[i + 1, :]))
            dna_str += SequenceTools.gray2dna_[(g1, g2)]
            i += 2
        return dna_str

    @staticmethod
    def convert_dna_str_to_gray(dna_str):
        """Convert length N string into 2N x 2 tokenized gray code array"""
        dna_str = dna_str.lower()
        N = len(dna_str)
        gray_arr = np.zeros((2 * N, 2))
        for i in range(N):
            gray_i = SequenceTools.dna2gray_[dna_str[i]]
            for j in range(2):
                gray_arr[2 * i + j, gray_i[j]] = 1
        return gray_arr

    @staticmethod
    def convert_rna_to_dna(seq):
        rna2dna = {'u': 't', 'a': 'a', 'g': 'g', 'c': 'c'}
        return "".join([rna2dna[s] for s in seq])

    @classmethod
    def get_codon_from_idx(cls, idx):
        idx2codon = {val: key for key, val in SequenceTools.codon2idx_.items()}
        return idx2codon[idx]

    @classmethod
    def get_start_codon_int(cls):
        return SequenceTools.codon2idx_['atg']

    @classmethod
    def get_stop_codon_ints(cls):
        stop_codons = SequenceTools.protein2codon_['stop']
        return [SequenceTools.codon2idx_[s] for s in stop_codons]

    @classmethod
    def translate_dna_str(cls, dna_seq):
        dna_seq = dna_seq.lower()
        prot_seq = []
        i = 0
        while i < len(dna_seq):
            cod = dna_seq[i:i + 3]
            prot_seq.append(SequenceTools.codon2protein_[cod])
            i += 3
        prot_seq = "".join(prot_seq)
        return prot_seq
