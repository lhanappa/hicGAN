import os, time, pickle, random, time, sys, math
from datetime import datetime
import numpy as np
from time import localtime, strftime
import logging, scipy
import hickle as hkl
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

except_chr = {'hsa': {'X': 23, 23: 'X'}, 'mouse': {'X': 20, 20: 'X'}}

def hicGAN_g(t_image, is_train=False, reuse=False):
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None  # tf.constant_initializer(value=0.0)
    g_init = tf.random_normal_initializer(1., 0.02)
    with tf.variable_scope("hicGAN_g", reuse=reuse) as vs:
        n = InputLayer(t_image, name='in')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='n64s1/c')
        temp = n
        # B residual blocks
        for i in range(5):
            nn = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c1/%s' % i)
            nn = BatchNormLayer(nn, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='n64s1/b1/%s' % i)
            nn = Conv2d(nn, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c2/%s' % i)
            nn = BatchNormLayer(nn, is_train=is_train, gamma_init=g_init, name='n64s1/b2/%s' % i)
            nn = ElementwiseLayer([n, nn], tf.add, name='b_residual_add/%s' % i)
            n = nn
        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c/m')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n64s1/b/m')
        n = ElementwiseLayer([n, temp], tf.add, name='add3')
        # B residual blacks end. output shape: (None,w,h,64)
        n = Conv2d(n, 128, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='n256s1/1')
        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='n256s1/2')
        n = Conv2d(n, 1, (1, 1), (1, 1), act=tf.nn.tanh, padding='SAME', W_init=w_init, name='out')
        return n

def hicgan_predictor(lr_mats, model_name, batch=64):
    t_image = tf.placeholder('float32', [None, None, None, 1], name='image_input')
    net_g = hicGAN_g(t_image, is_train=False, reuse=False)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    tl.files.load_and_assign_npz(sess=sess, name=model_name, network=net_g)
    out = np.zeros(lr_mats.shape)
    if out.shape[0] <= batch:
        out = sess.run(net_g.outputs, {t_image: lr_mats})
        return out
    else:
        for i in range(out.shape[0] // batch):
            out[batch*i:batch*(i+1)] = sess.run(net_g.outputs, {t_image: lr_mats[batch*i:batch*(i+1)]})
        out[batch*(i+1):] = sess.run(net_g.outputs, {t_image: lr_mats[batch*(i+1):]})
        return out

get_digit = lambda x: int(''.join(list(filter(str.isdigit, x))))
def filename_parser(filename):
    info_str = filename.split('.')[0].split('_')[2:-1]
    chunk = get_digit(info_str[0])
    stride = get_digit(info_str[1])
    bound = get_digit(info_str[2])
    return chunk, stride, bound

def data_info(data):
    indices = data['inds']
    compacts = data['compacts'][()]
    sizes = data['sizes'][()]
    return indices, compacts, sizes

def together(matlist, indices, corp=0, species='hsa', tag='HiC'):
    chr_nums = sorted(list(np.unique(indices[:,0])))
    print(chr_nums)
    # convert last element to str 'X'
    if chr_nums[-1] in except_chr[species]: chr_nums[-1] = except_chr[species][chr_nums[-1]]
    print(f'{tag} data contain {chr_nums} chromosomes')
    h, w = matlist[0].shape
    results = dict.fromkeys(chr_nums)
    for n in chr_nums:
        # convert str 'X' to 23
        num = except_chr[species][n] if isinstance(n, str) else n
        loci = np.where(indices[:,0] == num)[0]
        sub_mats = matlist[loci]
        index = indices[loci]
        width = index[0,1]
        full_mat = np.zeros((width, width))
        for sub, pos in zip(sub_mats, index):
            i, j = pos[-2], pos[-1]
            if corp > 0:
                sub = sub[:, corp:-corp, corp:-corp]
                h, w = sub.shape
            full_mat[i:i+h, j:j+w] = sub
        results[n] = full_mat
    return results

def compactM(matrix, compact_idx, verbose=False):
    """compacting matrix according to the index list."""
    compact_size = len(compact_idx)
    result = np.zeros((compact_size, compact_size)).astype(matrix.dtype)
    if verbose: print('Compacting a', matrix.shape, 'shaped matrix to', result.shape, 'shaped!')
    for i, idx in enumerate(compact_idx):
        result[i, :] = matrix[idx][compact_idx]
    return result

def save_data_n(key, hics, compacts, sizes, low_res, out_dir):
    file = os.path.join(out_dir, f'predict_chr{key}_{low_res}.npz')
    save_data(hics[key], compacts[key], sizes[key], file)

def save_data(hic, compact, size, file):
    hic = spreadM(hic, compact, size, convert_int=False, verbose=True)
    np.savez_compressed(file, hic=hic, compact=compact)
    print('Saving file:', file)

def predict(data_dir, model_name, out_dir, lr=40000, cuda=0):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda)
    # constuct lr_mats by your own if you want to using custom data.
    # lr_mats with shape (N,m,m,1)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    low_res = str(lr)
    files = [f for f in os.listdir(data_dir) if f.find(low_res) >= 0]
    hicgan_file = [f for f in files if f.find('.npz') >= 0][0]
    chunk, stride, bound = filename_parser(hicgan_file)
    #sr_mats_pre = predictor(lr_mats,model_name)

    start = time.time()
    print(f'Loading data[HiCGAN]: {hicgan_file}')
    hicgan_data = np.load(os.path.join(data_dir, hicgan_file), allow_pickle=True)
    inputs = np.array(hicgan_data['lr_data'], dtype=float)
    
    indices, compacts, sizes = data_info(hicgan_data)
    hicgan_hics = inputs #hicgan_predictor(inputs, model_name)
    hicgan_hics = np.squeeze(hicgan_hics, axis=-1)
    print(hicgan_hics.shape)
    print(indices.shape)
    print(indices[0:4,0])
    result_data = hicgan_hics # np.concatenate(hicgan_hics, axis=0)
    result_inds = indices # np.concatenate(indices, axis=0)
    hicgans = together(result_data, result_inds, tag='Reconstructing: ')

    print(f'Start saving predicted data')
    print(f'Output path: {out_dir}')
    for key in compacts.keys():
        save_data_n(key,deep_hics, compacts, sizes, low_res, out_dir)
    
    print(f'All data saved. Running cost is {(time.time()-start)/60:.1f} min.')
