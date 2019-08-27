import random
import torch
import numpy as np
import pandas as pd
from multiprocessing import Pool
from collections import UserList, defaultdict, OrderedDict
from rdkit import rdBase
from rdkit import Chem

from tqdm.autonotebook import tqdm
from selfies import encoder, decoder
from joblib import Parallel, delayed
import multiprocessing


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SS:
    bos = '<bos>'
    eos = '<eos>'
    pad = '<pad>'


class CharVocab:

    @classmethod
    def from_data(cls, data, *args, **kwargs):
        chars = set()
        for string in data:
            chars.update(string)

        return cls(chars, *args, **kwargs)

    def __init__(self, chars, ss=SS):
        if (ss.bos in chars) or (ss.eos in chars) or \
                (ss.pad in chars):
            raise ValueError('SS in chars')

        all_syms = sorted(list(chars)) + [ss.bos, ss.eos, ss.pad]  # ss.unk

        self.ss = ss
        self.c2i = {c: i for i, c in enumerate(all_syms)}
        self.i2c = {i: c for i, c in enumerate(all_syms)}

    def __len__(self):
        return len(self.c2i)

    @property
    def bos(self):
        return self.c2i[self.ss.bos]

    @property
    def eos(self):
        return self.c2i[self.ss.eos]

    @property
    def pad(self):
        return self.c2i[self.ss.pad]

    def char2id(self, char):
        if char not in self.c2i:
            raise ValueError('{} not understood'.format(char))

        return self.c2i[char]

    def id2char(self, id):
        if id not in self.i2c:
            raise ValueError('{} not understood'.format(id))

        return self.i2c[id]

    def string2ids(self, string, add_bos=False, add_eos=False):
        ids = [self.char2id(c) for c in string]

        if add_bos:
            ids = [self.bos] + ids
        if add_eos:
            ids = ids + [self.eos]

        return ids

    def ids2string(self, ids, rem_bos=True, rem_eos=True):
        if len(ids) == 0:
            return ''
        if rem_bos and ids[0] == self.bos:
            ids = ids[1:]
        if rem_eos and ids[-1] == self.eos:
            ids = ids[:-1]

        string = ''.join([self.id2char(id) for id in ids])

        return string


class OneHotVocab(CharVocab):

    def __init__(self, *args, **kwargs):
        super(OneHotVocab, self).__init__(*args, **kwargs)
        self.vectors = torch.eye(len(self.c2i))


def mapper(n_jobs):
    '''
    Returns function for map call.
    If n_jobs == 1, will use standard map
    If n_jobs > 1, will use multiprocessing pool
    If n_jobs is a pool object, will return its map function
    '''
    if n_jobs == 1:
        def _mapper(*args, **kwargs):
            return list(map(*args, **kwargs))

        return _mapper
    elif isinstance(n_jobs, int):
        pool = Pool(n_jobs)

        def _mapper(*args, **kwargs):
            try:
                result = pool.map(*args, **kwargs)
            finally:
                pool.terminate()
            return result

        return _mapper
    else:
        return n_jobs.map


class Logger(UserList):

    def __init__(self, data=None):
        super().__init__()
        self.sdata = defaultdict(list)
        for step in (data or []):
            self.append(step)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.data[key]
        elif isinstance(key, slice):
            return Logger(self.data[key])
        else:
            ldata = self.sdata[key]
            if isinstance(ldata[0], dict):
                return Logger(ldata)
            else:
                return ldata

    def append(self, step_dict):
        super().append(step_dict)
        for k, v in step_dict.items():
            self.sdata[k].append(v)

    def save(self, path):
        df = pd.DataFrame(list(self))
        df.to_csv(path, index=None)


class CircularBuffer:

    def __init__(self, size):
        self.max_size = size
        self.data = np.zeros(self.max_size)
        self.size = 0
        self.pointer = -1

    def add(self, element):
        self.size = min(self.size + 1, self.max_size)
        self.pointer = (self.pointer + 1) % self.max_size
        self.data[self.pointer] = element
        return element

    def last(self):
        assert self.pointer != -1, "Can't get an element from an empty buffer!"
        return self.data[self.pointer]

    def mean(self):
        if self.size > 0:
            return self.data[:self.size].mean()
        else:
            return 0.0


def disable_rdkit_log():
    rdBase.DisableLog('rdApp.*')


def enable_rdkit_log():
    rdBase.EnableLog('rdApp.*')


class AttributeDict(OrderedDict):

    def __init__(self, *args, **kwargs):
        super(AttributeDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def run_multiprocessing(input_list, fn):
    n_jobs = multiprocessing.cpu_count()
    result = Parallel(n_jobs=n_jobs)(delayed(fn)(x) for x in tqdm(input_list))
    return result


class SELFIESVocab(OneHotVocab):

    @classmethod
    def from_data(cls, smiles_list, *args, **kwargs):
        alphabet = set()

        def update_alphabet(x): return set(cls.smiles2selfies_list(x))

        uniq_chars = run_multiprocessing(smiles_list, update_alphabet)
        for aset in uniq_chars:
            alphabet.update(aset)
        print('Alphabet size is {}'.format(len(alphabet)))
        return cls(alphabet, *args, **kwargs)

    @classmethod
    def smiles2selfies_list(cls, smiles):
        selfies = str(encoder(smiles))
        selfies_list = selfies.replace('[', '').split(']')[:-1]
        return selfies_list

    @classmethod
    def selfies_list2smiles(cls, selfies_list):
        selfies = '[' + ']['.join(selfies_list) + ']'
        smiles = str(decoder(selfies))
        return smiles

    def string2ids(self, string, add_bos=False, add_eos=False):
        selfies_list = self.smiles2selfies_list(string)
        ids = [self.char2id(c) for c in selfies_list]

        if add_bos:
            ids = [self.bos] + ids
        if add_eos:
            ids = ids + [self.eos]

        return ids

    def ids2string(self, ids, rem_bos=True, rem_eos=True):
        if len(ids) == 0:
            return ''
        if rem_bos and ids[0] == self.bos:
            ids = ids[1:]
        if rem_eos and ids[-1] == self.eos:
            ids = ids[:-1]

        selfies_list = [self.id2char(id) for id in ids]
        smiles = self.selfies_list2smiles(selfies_list)
        return smiles


def valid_smiles(smiles):
    if len(smiles) == 0:
        return False
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    try:
        Chem.SanitizeMol(mol)
    except ValueError:
        return False
    return True


def print_config(namespace, title='config'):
    print('{:=^80}'.format(' ' + title + ' '))
    for key in namespace.__dict__:
        value = namespace.__dict__[key]
        if value is not None:
            print('{:>20s}:{:>20s}'.format(key, str(value)))


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]
