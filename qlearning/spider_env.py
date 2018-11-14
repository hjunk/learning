# 20080242 Kim Hyuk Jun
# EE488B Special Topics in EE <Deep Learning and AlphaGo>, Fall 2017
# Project #2 Task #2: 4-legged spider
# Date: Nov 29. 2017

import numpy as np

# 4-legged spider environment
class spider_environment:
    def __init__(self):
        self.n_states = 256         # number of states
        self.n_actions = 256        # number of actions
        self.reward = np.zeros([self.n_states, self.n_actions])
        self.terminal = np.zeros(self.n_states, dtype=np.int)          # 1 if terminal state, 0 otherwise
        self.next_state = np.zeros([self.n_states, self.n_actions], dtype=np.int)        # next_state
        self.init_state = 0b00001010    # initial state
        transition = [[1,0,2,0],[1,0,3,1],[3,2,2,0],[3,2,3,1]]

        for s in range(256):
            lf_up = s & 1
            lf_fw = (s >> 1) & 1
            rf_up = (s >> 2) & 1
            rf_fw = (s >> 3) & 1
            lb_up = (s >> 4) & 1
            lb_fw = (s >> 5) & 1
            rb_up = (s >> 6) & 1
            rb_fw = (s >> 7) & 1

            for a in range(256):
                lf_action_up = (a & 3) == 0
                lf_action_dn = (a & 3) == 1
                lf_action_fw = (a & 3) == 2
                lf_action_bw = (a & 3) == 3
                rf_action_up = ((a >> 2) & 3) == 0
                rf_action_dn = ((a >> 2) & 3) == 1
                rf_action_fw = ((a >> 2) & 3) == 2
                rf_action_bw = ((a >> 2) & 3) == 3
                lb_action_up = ((a >> 4) & 3) == 0
                lb_action_dn = ((a >> 4) & 3) == 1
                lb_action_fw = ((a >> 4) & 3) == 2
                lb_action_bw = ((a >> 4) & 3) == 3
                rb_action_up = ((a >> 6) & 3) == 0
                rb_action_dn = ((a >> 6) & 3) == 1
                rb_action_fw = ((a >> 6) & 3) == 2
                rb_action_bw = ((a >> 6) & 3) == 3

                lf_s = s & 3
                rf_s = (s >> 2) & 3
                lb_s = (s >> 4) & 3
                rb_s = (s >> 6) & 3

                lf_a = a & 3
                rf_a = (a >> 2) & 3
                lb_a = (a >> 4) & 3
                rb_a = (a >> 6) & 3

                next_state = (transition[rb_s][rb_a] << 6) + (transition[lb_s][lb_a] << 4) + (transition[rf_s][rf_a] << 2) + (transition[lf_s][lf_a])
                self.next_state[s,a] = next_state

                total_down = (lf_up == 0 and lf_action_up == 0) + (rf_up == 0 and rf_action_up == 0) + (lb_up == 0 and lb_action_up == 0) + (rb_up == 0 and lb_action_up == 0)

                lf_force = (lf_up == 0 and lf_fw == 1 and lf_action_bw == 1) - (lf_up == 0 and lf_fw == 0 and lf_action_fw == 1)
                rf_force = (rf_up == 0 and rf_fw == 1 and rf_action_bw == 1) - (rf_up == 0 and rf_fw == 0 and rf_action_fw == 1)
                lb_force = (lb_up == 0 and lb_fw == 1 and lb_action_bw == 1) - (lb_up == 0 and lb_fw == 0 and lb_action_fw == 1)
                rb_force = (rb_up == 0 and rb_fw == 1 and rb_action_bw == 1) - (rb_up == 0 and rb_fw == 0 and rb_action_fw == 1)
                total_force = (lf_force + rf_force + lb_force + rb_force)

                diag_cond = (lf_up == 0 and lf_action_up == 0 and rb_up == 0 and lb_action_up == 0) or (rf_up == 0 and rf_action_up == 0 and lb_up == 0 and lb_action_up == 0)

                reward = 0
                if (total_down == 0): pass
                elif (total_down >= 3):
                    reward = (total_force / float(total_down))
                elif (diag_cond and total_down == 2):
                    reward = (total_force / float(total_down))
                else:
                    reward = 0.25 * (total_force / float(total_down))

                self.reward[s, a] = reward