import tensorflow as tf
import numpy as np
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'tf_ops'))
sys.path.append(os.path.join(BASE_DIR, 'tf_ops/grouping'))
import tf_util
import pointfly as pf
from tf_grouping import group_point, knn_point

# A shape is (N, P, C)
def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    cls_labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl, labels_pl, cls_labels_pl

def RIConv(pts, fts_prev, qrs, is_training, tag, K, D, P, C, with_local, bn_decay=None):
    # pts: points input with batch size (?,256,3)---the second layer
    # fts_prev: features from previous layer (?,256,128)---the second layer: 128 is the number of channels(C)
    # qrs: points sampled by farthest point sampling (?,128,3)---the second layer: 128 is the number of sampling point(P)
    # K: K nearest neighborhood; D: number of bins; P: number of sampling points; C: number of channel

    indices = pf.knn_indices_general(qrs, pts, int(K), True) # indices for k nearest neighborhood (?,128,32,2)----the second layer: 128 is the number of point; 32 is the nearest neighborhood
    nn_pts = tf.gather_nd(pts, indices, name=tag + 'nn_pts') # coordinate for k nearest neighborhood (?,128,32,3)
    
    nn_pts_center = tf.expand_dims(qrs, axis=2, name=tag + 'nn_pts_center') # expand the dimension of qrs (fps points) (?,128,3)----(?,128,1,3)
    #nn_pts_local = tf.subtract(nn_pts, nn_pts_center, name=tag + 'nn_pts_local') # coordinate difference between fps point and local point (?,128,32,3)
    #dists_local = tf.norm(nn_pts_local, axis=-1, keepdims=True)  # dist to center(fps) (?,128,32,1)

    mean_local = tf.reduce_mean(nn_pts, axis=-2, keepdims=True) # mean of local region: (?,128,1,3)---32-1
    mean_global = tf.reduce_mean(pts, axis=-2, keepdims=True) # (?,256,3)---(?,1,3)
    mean_global = tf.expand_dims(mean_global, axis=-2) # mean of global region: (?,1,3)----(?,1,1,3)

    #nn_pts_local_mean = tf.subtract(nn_pts, mean_local, name=tag + 'nn_pts_local_mean') # (?,128,32,3)
    #dists_local_mean = tf.norm(nn_pts_local_mean, axis=-1, keepdims=True) # dist to local mean # (?,128,32,1)
    
    ## seven distances
    fps_global = qrs[:,0,:]
    fps_global = tf.expand_dims(tf.expand_dims(fps_global,axis=-2), axis=-2)
    # local distance
    dist1 = tf.norm( tf.subtract(nn_pts, nn_pts_center) , axis=-1, keepdims=True)
    dist2 = tf.norm( tf.subtract(nn_pts, mean_local) , axis=-1, keepdims=True)
    dist3 = tf.norm( tf.subtract(mean_local, nn_pts_center) , axis=-1, keepdims=True)
    #dist3 = tf.repeat( tf.expand_dims(dist3, axis=-2), [1,1,K,1] )
    dist3 = tf.tile( dist3, [1,1,K,1] )
    # connection between local and global distances
    dist4 = tf.norm( tf.subtract(nn_pts, nn_pts_center) , axis=-1, keepdims=True)
    #dist4 = tf.repeat( tf.expand_dims(dist4, axis=-2), [1,1,K,1] )
    # global distance
    dist5 = tf.norm( tf.subtract(nn_pts, mean_global) , axis=-1, keepdims=True)
    dist6 = tf.norm( tf.subtract(nn_pts, fps_global) , axis=-1, keepdims=True)
    dist7 = tf.norm( tf.subtract(mean_global, fps_global) , axis=-1, keepdims=True)
    #dist7 = tf.repeat( tf.expand_dims(dist7, axis=-2), [1,np.shape(dist6)[1],K,1] )
    dist7 = tf.tile( dist7, [1,np.shape(dist6)[1],K,1] )
    #nn_fts = tf.concat([dist1, dist2, dist3, dist4, dist5, dist6, dist7], axis=-1)
    nn_fts = tf.concat([dist1, dist2, dist3, dist4, dist5], axis=-1)
    ##

    #vec = mean_local - nn_pts_center # (?,128,1,3)-(?,128,1,3): local mean - FPS sampling points
    #vec_dist = tf.norm(vec, axis=-1, keepdims =True) # distance between local mean and FPS sampling points (?,128,1,1)
    #vec_norm = tf.divide(vec, vec_dist) # normalize vec (?,128,1,3): cos(alpha), cos(beta), cos(gamma)
    #vec_norm = tf.where(tf.is_nan(vec_norm), tf.ones_like(vec_norm) * 0, vec_norm) # replace nan with 0 (?,128,1,3)

    #nn_pts_local_proj = tf.matmul(nn_pts_local, vec_norm, transpose_b=True) # projection of 32 points along the aforementioned line: (?,128,32,3)*(?,128.1.3)'=(?,128.32.1)
    #nn_pts_local_proj_dot = tf.divide(nn_pts_local_proj, dists_local) # noralize across the distance (?,128,32,1)/(?,128,32,1)=(?,128,32,1): actually the dot product of normal vector(vec_norm) and normalized nn_pts_local
    #nn_pts_local_proj_dot = tf.where(tf.is_nan(nn_pts_local_proj_dot), tf.ones_like(nn_pts_local_proj_dot) * 0, nn_pts_local_proj_dot)  # check nan

    #nn_pts_local_proj_2 = tf.matmul(nn_pts_local_mean, vec_norm, transpose_b=True)
    #nn_pts_local_proj_dot_2 = tf.divide(nn_pts_local_proj_2, dists_local_mean) # actually the dot product of normal vector(vec_norm) and normalized nn_pts_local_mean
    #nn_pts_local_proj_dot_2 = tf.where(tf.is_nan(nn_pts_local_proj_dot_2), tf.ones_like(nn_pts_local_proj_dot_2) * 0, nn_pts_local_proj_dot_2)  # check nan

    #nn_fts = tf.concat([dists_local, dists_local_mean, nn_pts_local_proj_dot, nn_pts_local_proj_dot_2], axis=-1) # d0 d1 a0 a1 
    # concate the information together [(?,128,32,1),(?,128,32,1),(?,128,32,1),(?,128,32,1)] = (?,128,32,4)

    # compute indices from nn_pts_local_proj
    #vec = mean_global - nn_pts_center # distance between global mean and samling points: (?,1,1,3)-(?,128,1,3) = (?,128,1,3)
    #vec_dist = tf.norm(vec, axis=-1, keepdims =True) #(?,128,1,1)
    #vec_norm = tf.divide(vec, vec_dist) # normalize (?,128,1,3)
    #nn_pts_local_proj = tf.matmul(nn_pts_local, vec_norm, transpose_b=True) 
    # projection: coordinate of local point (relative to sampling points) and normal vector (between global mean and sampling point) (?,128,32,3)*(?,128,1,3)'=(?,128,32,1)

    #proj_min = tf.reduce_min(nn_pts_local_proj, axis=-2, keepdims=True) # find minimum (?,128,1,1)
    #proj_max = tf.reduce_max(nn_pts_local_proj, axis=-2, keepdims=True) # find maximum (?,128,1,1)
    #seg = (proj_max - proj_min) / D # average segmentation (?,128,1,1)

    #vec_tmp = tf.range(0, D, 1, dtype=tf.float32)
    #vec_tmp = tf.reshape(vec_tmp, (1,1,1,D)) # divide (0,1) into 2 parts (1,1,1,2)

    #limit_bottom = vec_tmp * seg + proj_min # lower limit (1,1,1,2)*(?,128,1,1)+(?,128,1,1) = (?,128,1,2)
    #limit_up = limit_bottom + seg # upper limit: difference a segmentation

    #idx_up = nn_pts_local_proj <= limit_up # find smaller than limit_up (?,128.32.1)<(?,128,1,2)=(?,128,32,2)
    #idx_bottom = nn_pts_local_proj >= limit_bottom # find larger than limit_bottom (?,128,32,2)
    #idx = tf.to_float(tf.equal(idx_bottom, idx_up)) # (?,128,32,2)
    #idx_expand = tf.expand_dims(idx, axis=-1) # (?,128,32,2,1)
    
    [N,P,K,dim] = nn_fts.shape # (N, P, K, 4) N: batch size; P: number of sampling point K: k-nearest neighborhood
    nn_fts_local = None
    if with_local: # true: whu with local?
        C_pts_fts = 64
        nn_fts_local_reshape = tf.reshape(nn_fts, (-1,P*K,dim,1)) # (N,P*K,dim,1)----(?,4096,4,1)
        # inputs,num_output_channels,kernel_size # (?,4096,1,32)
        nn_fts_local = tf_util.conv2d(nn_fts_local_reshape, C_pts_fts//2, [1,dim],
                         padding='VALID', stride=[1,1], 
                         bn=True, is_training=is_training,
                         scope=tag+'conv_pts_fts_0', bn_decay=bn_decay)
        
        # inputs,num_output_channels,kernel_size # (?,4096,1,64)
        nn_fts_local = tf_util.conv2d(nn_fts_local, C_pts_fts, [1,1],
                         padding='VALID', stride=[1,1], 
                         bn=True, is_training=is_training,
                         scope=tag+'conv_pts_fts_1', bn_decay=bn_decay)
        # (N,P,K,C_pts_fts)----(?,128,32,64)
        nn_fts_local = tf.reshape(nn_fts_local, (-1,P,K,C_pts_fts))
    else:
        nn_fts_local = nn_fts
    
    if fts_prev is not None:
        fts_prev = tf.gather_nd(fts_prev, indices, name=tag + 'fts_prev')  # (N, P, K, C_prev)---(?,128,32,64)
        pts_X_0 = tf.concat([nn_fts_local,fts_prev], axis=-1) # (N, P, K, C_prev+C_pts_fts)---(?,128,32,192)
    else:
        pts_X_0 = nn_fts_local # (N,P,K,C_pts_fts)----(?,128,32,64)

    pts_X_0_expand = tf.expand_dims(pts_X_0, axis=-2) # (?,128,32,1,192)
    #nn_fts_rect = pts_X_0_expand * idx_expand # (?,128,32,1,192) * (?,128,32,2,1) = (?,128,32,2,192)
    nn_fts_rect = pts_X_0_expand
    nn_fts_rect = tf.reduce_max(nn_fts_rect, axis=-3) # (?,128,2,192) 
    
    # nn_fts_rect = tf.matmul(idx_mean, pts_X_0, transpose_a = True)

    # inputs,num_output_channels,kernel_size # (?,128,1,256)    
    fts_X = tf_util.conv2d(nn_fts_rect, C, [1,nn_fts_rect.shape[-2].value],
                         padding='VALID', stride=[1,1], 
                         bn=True, is_training=is_training,
                         scope=tag+'conv', bn_decay=bn_decay)
    return tf.squeeze(fts_X, axis=-2) # (?,128,256)  

def get_model(layer_pts, is_training, RIconv_params, RIdconv_params, fc_params, sampling='fps', weight_decay=0.0, bn_decay=None, part_num=50):
    
    if sampling == 'fps':
        sys.path.append(os.path.join(BASE_DIR, 'tf_ops/sampling'))
        from tf_sampling import farthest_point_sample, gather_point

    layer_fts_list = [None]
    layer_pts_list = [layer_pts]
    for layer_idx, layer_param in enumerate(RIconv_params):
        tag = 'xconv_' + str(layer_idx + 1) + '_'
        K = layer_param['K']
        D = layer_param['D']
        P = layer_param['P']
        C = layer_param['C']
        # qrs = layer_pts if P == -1 else layer_pts[:,:P,:]  # (N, P, 3)

        if P == -1:
            qrs = layer_pts
        else:
            if sampling == 'fps':
                qrs = gather_point(layer_pts, farthest_point_sample(P, layer_pts))
            elif sampling == 'random':
                qrs = tf.slice(layer_pts, (0, 0, 0), (-1, P, -1), name=tag + 'qrs')  # (N, P, 3)
            else:
                print('Unknown sampling method!')
                exit()
        layer_fts= RIConv(layer_pts_list[-1], layer_fts_list[-1], qrs, is_training, tag, K, D, P, C, True, bn_decay)
        
        layer_pts = qrs
        layer_pts_list.append(qrs)
        layer_fts_list.append(layer_fts)
  
    if RIdconv_params is not None:
        fts = layer_fts_list[-1]
        for layer_idx, layer_param in enumerate(RIdconv_params):
            tag = 'xdconv_' + str(layer_idx + 1) + '_'
            K = layer_param['K'] 
            D = layer_param['D'] 
            pts_layer_idx = layer_param['pts_layer_idx']  # 2 1 0 
            qrs_layer_idx = layer_param['qrs_layer_idx']  # 1 0 -1

            pts = layer_pts_list[pts_layer_idx + 1]
            qrs = layer_pts_list[qrs_layer_idx + 1]
            fts_qrs = layer_fts_list[qrs_layer_idx + 1]

            C = fts_qrs.get_shape()[-1].value if fts_qrs is not None else C//2
            P = qrs.get_shape()[1].value
            
            layer_fts= RIConv(pts, fts, qrs, is_training, tag, K, D, P, C, True, bn_decay)
            if fts_qrs is not None: # this is for last layer
                fts_concat = tf.concat([layer_fts, fts_qrs], axis=-1, name=tag + 'fts_concat')
                fts = pf.dense(fts_concat, C, tag + 'fts_fuse', is_training)
            else:
                fts = layer_fts
        
    features = layer_fts  # the feature for similarity search
    for layer_idx, layer_param in enumerate(fc_params):
        C = layer_param['C']
        dropout_rate = layer_param['dropout_rate']
        # input, output, name, is_training, reuse=None, with_bn=True, activation=tf.nn.relu
        layer_fts = pf.dense(layer_fts, C, 'fc{:d}'.format(layer_idx), is_training)
        layer_fts = tf.layers.dropout(layer_fts, dropout_rate, training=is_training, name='fc{:d}_drop'.format(layer_idx)) #(?,64,512)--(?,64,256)
    logits_seg = pf.dense(layer_fts, part_num, 'logits', is_training, with_bn=False, activation=None) #(?,64,40)
    return features, logits_seg

def get_loss(seg_pred, seg_label):
    """ pred: BxNxC,
        label: BxN, """
    

    # size of seg_pred is batch_size x point_num x part_cat_num
    # size of seg is batch_size x point_num
    per_instance_seg_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=seg_pred, labels=seg_label), axis=1)
    seg_loss = tf.reduce_mean(per_instance_seg_loss)

    per_instance_seg_pred_res = tf.argmax(seg_pred, 2)

    return seg_loss, per_instance_seg_loss, per_instance_seg_pred_res


if __name__ == '__main__':
    print('This is the Rotaion Invairant Convolution Operator')
