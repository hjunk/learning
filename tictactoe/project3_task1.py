# 20080242 Kim Hyuk Jun
# EE488B Special Topics in EE <Deep Learning and AlphaGo>, Fall 2017
# Project #3 Task #2: Mini AlphaGo Zero
# Date: Dec 18. 2017

import tensorflow as tf
import numpy as np
from types import MethodType
from boardgame import game1, game2, game3, game4, data_augmentation

# Choose game tic-tac-toe
game = game2()

## DEFINE NEW OPPONENT ##
def next_move(self, b, state, game_in_progress, net, rn, p, move, nlevels=1, rw=0):
    # returns next move by using neural networks
    # this is a parallel version, i.e., returns next moves for multiple games
    # Input arguments: b,state,game_in_progress,net,rn,p,move,nlevels,rw
    #   b: current board states for multiple games
    #   state: extra states
    #   game_in_progress: 1 if game is in progress, 0 if ended
    #   net: neural network. can be empty (in that case 'rn' should be 1)
    #   rn: if 0 <= rn <= 1, controls randomness in each move (0: no randomness, 1: pure random)
    #     if rn = -1, -2, ..., then the first |rn| moves are random
    #   p: current player (1: black, 2: white)
    #   move: k-th move (1,2,3,...)
    #   nlevels (optional): tree search depth (1,2, or 3). default=1
    #     if nlevels is even, then 'net' should be the opponent's neural network
    #   rw (optional): randomization in calculating winning probabilities, default=0
    # Return values
    # new_board,new_state,valid_moves,wp_max,wp_all,x,y=next_move(b,game_in_progress,net,rn,p,move)
    #   new_board: updated board states containing new moves
    #   new_state: updated extra states
    #   n_valid_moves: number of valid moves
    #   wp_max: best likelihood of winning
    #   wp_all: likelihood of winning for all possible next moves
    #   x: x coordinates of the next moves in 'new_board'
    #   y: y coordinates of the next moves in 'new_board'

    # board size
    nx = self.nx;
    ny = self.ny;
    nxy = nx * ny
    # randomness for each game & minimum r
    r = rn;
    rmin = np.amin(r)
    # number of games
    if b.ndim >= 3:
        ng = b.shape[2]
    else:
        ng = 1
    # number of valid moves in each game
    n_valid_moves = np.zeros((ng))
    # check whether each of up to 'nxy' moves is valid for each game
    valid_moves = np.zeros((ng, nxy))
    # win probability for each next move
    wp_all = np.zeros((nx, ny, ng))
    # maximum of wp_all over all possible next moves
    wp_max = -np.ones((ng))
    mx = np.zeros((ng))
    my = np.zeros((ng))
    x = -np.ones((ng))
    y = -np.ones((ng))

    # check nlevels
    if nlevels > 3 or nlevels <= 0:
        raise Exception('# of levels not supported. Should be 1, 2, or 3.')
    # total cases to consider in tree search
    ncases = pow(nxy, nlevels)

    # maximum possible board states considering 'ncases'
    d = np.zeros((nx, ny, 3, ng * ncases), dtype=np.int32)

    for p1 in range(nxy):
        vm1, b1, state1 = self.valid(b, state, self.xy(p1), p)
        n_valid_moves += vm1
        if rmin < 1:
            valid_moves[:, p1] = vm1
            if nlevels == 1:
                c = 3 - p  # current player is changed to the next player after placing a stone at 'p1'
                idx = np.arange(ng) + p1 * ng
                d[:, :, 0, idx] = (b1 == c)  # 1 if current player's stone is present, 0 otherwise
                d[:, :, 1, idx] = (b1 == 3 - c)  # 1 if opponent's stone is present, 0 otherwise
                d[:, :, 2, idx] = 2 - c  # 1: current player is black, 0: white
            else:
                for p2 in range(nxy):
                    vm2, b2, state2 = self.valid(b1, state1, self.xy(p2), 3 - p)
                    if nlevels == 2:
                        c = p  # current player is changed again after placing a stone at 'p2'
                        idx = np.arange((ng)) + p1 * ng + p2 * ng * nxy
                        d[:, :, 0, idx] = (b2 == c)
                        d[:, :, 1, idx] = (b2 == 3 - c)
                        d[:, :, 2, idx] = 2 - c
                    else:
                        for p3 in range(nxy):
                            vm3, b3, state3 = self.valid(b2, state2, self.xy(p3), p)
                            c = 3 - p  # current player is changed yet again after placing a stone at 'p3'
                            idx = np.arange(ng) + p1 * ng + p2 * ng * nxy \
                                  + p3 * ng * nxy * nxy
                            d[:, :, 0, idx] = (b3 == c)
                            d[:, :, 1, idx] = (b3 == 3 - c)
                            d[:, :, 2, idx] = 2 - c

    # n_valid_moves is 0 if game is not in progress
    n_valid_moves = n_valid_moves * game_in_progress

    # For operations in TensorFlow, load session and graph
    sess = tf.get_default_session()

    # d(nx, ny, 3, ng * ncases) becomes d(ng * ncases, nx, ny, 3)
    d = np.rollaxis(d, 3)
    if rmin < 1:  # if not fully random, then use the neural network 'net'
        softout = np.zeros((d.shape[0], 3))
        size_minibatch = 1024
        num_batch = np.ceil(d.shape[0] / float(size_minibatch))
        for batch_index in range(int(num_batch)):
            batch_start = batch_index * size_minibatch
            batch_end = \
                min((batch_index + 1) * size_minibatch, d.shape[0])
            indices = range(batch_start, batch_end)
            feed_dict = {'S:0': d[indices, :, :, :]}  # d[indices,:,:,:] goes to 'S' (neural network input)
            softout[indices, :] = sess.run(net, feed_dict=feed_dict)  # get softmax output from 'net'
        if p == 1:  # if the current player is black
            # softout[:,0] is the softmax output for 'tie'
            # softout[:,1] is the softmax output for 'black win'
            # softout[:,2] is the softmax output for 'white win'
            wp = 0.5 * (1 + softout[:, 1] - softout[:, 2])  # estimated win prob. for black
        else:  # if the current player is white
            wp = 0.5 * (1 + softout[:, 2] - softout[:, 1])  # estimated win prob. for white

        if rw != 0:  # this is only for nlevels == 1
            # add randomness so that greedy action selection to be done later is randomized
            wp = wp + np.random.rand((ng, 1)) * rw

        if nlevels >= 3:
            wp = np.reshape(wp, (ng, nxy, nxy, nxy))
            wp = np.amax(wp, axis=3)

        if nlevels >= 2:
            wp = np.reshape(wp, (ng, nxy, nxy))
            wp = np.amin(wp, axis=2)

        wp = np.transpose(np.reshape(wp, (nxy, ng)))
        wp = valid_moves * wp - (1 - valid_moves)
        wp_i = np.argmax(wp, axis=1)  # greedy action selection
        mxy = self.xy(wp_i)  # convert to (x,y) coordinates

        for p1 in range(nxy):
            pxy = self.xy(p1)
            wp_all[int(pxy[:, 0]), int(pxy[:, 1]), :] = wp[:, p1]  # win prob. for each of possible next moves

    new_board = np.zeros(b.shape)
    new_board[:, :, :] = b[:, :, :]
    new_state = np.zeros(state.shape)
    new_state[:, :] = state[:, :]

    for k in range(ng):
        if n_valid_moves[k]:  # if there are valid moves
            # if r[k] == 1, modified method treats this case as suggested opponent
            if (r[k] == 1):
                cg = ng - 1
                oppo = net[cg]
                while True:
                    oxy = self.xy(oppo)
                    isvalid, _, _ = self.valid(b[:, :, [k]], state[:, [k]], oxy, p)
                    if int(isvalid[0]):
                        break
                    else:
                        oppo += 1

            elif (r[k] < 0 and np.ceil(move / 2.) <= -r[k]) \
                    or (r[k] >= 0 and np.random.rand() <= r[k]):
                # if r[k]<0, then randomize the next move if # of moves is <= |r[k]|
                # if 0<r[k]<=1, then randomize the next move with probability r[k]
                # randomization is uniform over all possible valid moves
                while True:
                    # random position selection
                    rj = np.random.randint(nx)
                    rk = np.random.randint(ny)
                    rxy = np.array([[rj, rk]])
                    isvalid, _, _ = self.valid(b[:, :, [k]], state[:, [k]], rxy, p)
                    if int(isvalid[0]):
                        break

                isvalid, bn, sn = self.valid(b[:, :, [k]], state[:, [k]], rxy, p)
                new_board[:, :, [k]] = bn
                new_state[:, [k]] = sn
                x[k] = rj
                y[k] = rk

            else:
                isvalid, bn, sn = self.valid(b[:, :, [k]], state[:, [k]], mxy[[k], :], p)
                new_board[:, :, [k]] = bn
                new_state[:, [k]] = sn
                x[k] = mxy[k, 0]
                y[k] = mxy[k, 1]

        else:  # if there is no more valid move
            isvalid, bn, sn = self.valid(b[:, :, [k]], state[:, [k]], -np.ones((1, 2)), p)
            new_state[:, [k]] = sn

    return new_board, new_state, n_valid_moves, wp_max, wp_all, x, y

game.next_move = MethodType(next_move, game)

### NETWORK ARCHITECTURE ###
def network(state, nx, ny):
    # Set variable initializers
    init_weight = tf.random_normal_initializer(stddev=0.1)
    init_bias = tf.constant_initializer(0.1)

    # Create variables "weights1" and "biases1".
    weights1 = tf.get_variable("weights1", [3, 3, 3, 30], initializer=init_weight)
    biases1 = tf.get_variable("biases1", [30], initializer=init_bias)

    # Create 1st layer
    conv1 = tf.nn.conv2d(state, weights1, strides=[1, 1, 1, 1], padding='SAME')
    out1 = tf.nn.relu(conv1 + biases1)

    # Create variables "weights2" and "biases2".
    weights2 = tf.get_variable("weights2", [3, 3, 30, 50], initializer=init_weight)
    biases2 = tf.get_variable("biases2", [50], initializer=init_bias)

    # Create 2nd layer
    conv2 = tf.nn.conv2d(out1, weights2, strides=[1, 1, 1, 1], padding='SAME')
    out2 = tf.nn.relu(conv2 + biases2)

    # Create variables "weights1fc" and "biases1fc".
    weights1fc = tf.get_variable("weights1fc", [nx * ny * 50, 100], initializer=init_weight)
    biases1fc = tf.get_variable("biases1fc", [100], initializer=init_bias)

    # Create 1st fully connected layer
    fc1 = tf.reshape(out2, [-1, nx * ny * 50])
    out1fc = tf.nn.relu(tf.matmul(fc1, weights1fc) + biases1fc)

    # Create variables "weights2fc" and "biases2fc".
    weights2fc = tf.get_variable("weights2fc", [100, 3], initializer=init_weight)
    biases2fc = tf.get_variable("biases2fc", [3], initializer=init_bias)

    # Create 2nd fully connected layer
    return tf.matmul(out1fc, weights2fc) + biases2fc


# Input (common for all networks)
S = tf.placeholder(tf.float32, shape = [None, game.nx, game.ny, 3], name = "S")

# temporary network for loading from .ckpt
scope = "network"
with tf.variable_scope(scope):
    # Estimation for unnormalized log probability
    Y = network(S, game.nx, game.ny)
    # Estimation for probability
    P = tf.nn.softmax(Y, name = "softmax")

# network0 for black
# network1 for white
for i in range(2):
    scope = "network" + str(i)
    with tf.variable_scope(scope):
        # Estimation for unnormalized log probability
        Y = network(S, game.nx, game.ny)
        # Estimation for probability
        P = tf.nn.softmax(Y, name = "softmax")

N_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = "network/")
N0_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = "network0/")

### SAVER ###
saver = tf.train.Saver(N_variables)

with tf.Session() as sess:
    ### DEFAULT SESSION ###
    sess.as_default()

    ### VARIABLE INITIALIZATION ###
    sess.run(tf.global_variables_initializer())
    n_test = 1
    r_none = np.zeros((n_test))
    r_one = np.ones((n_test))

    saver.restore(sess, "./project3_task1.ckpt")
    for i in range(len(N_variables)):
        sess.run(tf.assign(N0_variables[i], N_variables[i]))

    N0 = tf.get_default_graph().get_tensor_by_name("network0/softmax:0")

    win=0; loss=0; tie=0
    for i in range(8):
        for j in range(6):
            for k in range(4):
                for l in range(2):
                    opponent = [i, j, k, l]
                    s = game.play_games(N0, r_none, opponent, r_one, n_test, nargout=1)
                    win += s[0][0]; loss += s[0][1]; tie += s[0][2]
    print('net1 (black) against opponent (white): win %d times, loss %d times, tie %d times' % (win, loss, tie))

    win=0; loss=0; tie=0
    for i in range(9):
        for j in range(7):
            for k in range(5):
                for l in range(3):
                    opponent = [i, j, k, l]
                    s = game.play_games(opponent, r_one, N0, r_none, n_test, nargout=1)
                    win += s[0][1]; loss += s[0][0]; tie += s[0][2]
    print('net1 (white) against opponent (black): win %d times, loss %d times, tie %d times' % (win, loss, tie))