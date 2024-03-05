import h5py
import numpy as np
import Train
with h5py.File('GIB-UVA ERP-BCI.hdf5', 'r') as f:
    for key in f.keys():
        print(key)
    data = f['features']
    labels = f['erp_labels']
    target = f['target']
    m_d = f['matrix_dims']
    r_i = f['run_indexes']
    sequences = f['sequences']
    subjects = f['subjects']
    trials = f['trials']
    print(target)
    print(labels)
    print(data)
    print(m_d)
    print(sequences)
    print(r_i)
    print(subjects)
    print(trials)
    l = labels[0:10000]
    d = data[0:10000, 0:128, 0:8]
    t = target[4000:5000, 2]
    sq = sequences[0:1000]
    ri = r_i[0:1000]
    sb = subjects[1000:2000]
    tr = trials[701000:701615]
Fs = 128
# Train.Neuronet_0(d, l, Fs)
Train.CNN(d, l, Fs)