# Copyright 2019 Nathan Jay and Noga Rotman
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.registration import register
import numpy as np
import heapq
import time
import random
import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from common import sender_obs
from common.simple_arg_parse import arg_or_default

MAX_CWND = 5000
MIN_CWND = 4

MAX_RATE = 1000
MIN_RATE = 40

REWARD_SCALE = 0.001

MAX_STEPS = 400

EVENT_TYPE_SEND = 'S'
EVENT_TYPE_ACK = 'A'

BYTES_PER_PACKET = 1500

LATENCY_PENALTY = 1.0
LOSS_PENALTY = 1.0

USE_LATENCY_NOISE = False
MAX_LATENCY_NOISE = 1.1

DELTA_SCALE = arg_or_default("--delta-scale", 0.025) # default = 0.025
HISTORY_LEN = arg_or_default("--history-len", 10)  # default = 10

USE_CWND = False    # default = False

class Link():

    #sim.links = [Link(bw, lat, queue, loss), Link(bw, lat, queue, loss)]  
    def __init__(self, bandwidth, delay, queue_size, loss_rate):
        self.bw = float(bandwidth)     # bandwidth
        self.dl = delay                # delay ( latency )
        self.lr = loss_rate            # random loss rate
        self.queue_delay = 0.0 
        self.queue_delay_update_time = 0.0
        self.max_queue_delay = queue_size / self.bw # queue size * sec / bit 

    def get_cur_queue_delay(self, event_time):
        #print("qdelay %f, event_time %f, qdelaytime %f" %(self.queue_delay, event_time, self.queue_delay_update_time))
        return max(0.0, self.queue_delay - (event_time - self.queue_delay_update_time))

    def get_cur_latency(self, event_time):
        return self.dl + self.get_cur_queue_delay(event_time)

    # random loss and loss by delay 발생
    def packet_enters_link(self, event_time):
        if (random.random() < self.lr):     # random loss rate
            #print("\t Random Drop!")
            return False
        self.queue_delay = self.get_cur_queue_delay(event_time)
        self.queue_delay_update_time = event_time
        extra_delay = 1.0 / self.bw
        #print("Extra delay: %f, Current delay: %f, Max delay: %f" % (extra_delay, self.queue_delay, self.max_queue_delay))
        if extra_delay + self.queue_delay > self.max_queue_delay:  # queue_delay > queue_size - 1 / bandwidth
            #print("\t Congestion Drop!")
            return False
        self.queue_delay += extra_delay
        #print("\tNew delay = %f" % self.queue_delay)
        return True

    def print_debug(self):
        print("Link:")
        print("Bandwidth: %f" % self.bw)
        print("Delay: %f" % self.dl)
        print("Queue Delay: %f" % self.queue_delay)
        print("Max Queue Delay: %f" % self.max_queue_delay)
        print("One Packet Queue Delay: %f" % (1.0 / self.bw))

    def reset(self):
        self.queue_delay = 0.0
        self.queue_delay_update_time = 0.0

class Network():
    
    #sim.net = Network(self.senders, self.links)
    def __init__(self, senders, links):
        self.q = []
        self.cur_time = 0.0
        self.senders = senders
        self.links = links
        self.queue_initial_packets()

    # packet = ( event_time, sender, event_type, next_hop, cur_latency, dropped )
    def queue_initial_packets(self):
        for sender in self.senders:
            sender.register_network(self)
            sender.reset_obs()
            heapq.heappush(self.q, (1.0 / sender.rate, sender, EVENT_TYPE_SEND, 0, 0.0, False)) 
            # event_time = 1 / sender.rate = packet 하나? 에 걸리는 시간

    def reset(self):
        self.cur_time = 0.0
        self.q = []
        [link.reset() for link in self.links]
        [sender.reset() for sender in self.senders]
        self.queue_initial_packets()

    def get_cur_time(self):
        #print("[time step ",self.cur_time, "]")
        return self.cur_time
    
    # network run for dur ( 3 * lat ) = T_m 
    def run_for_dur(self, dur):
        end_time = self.cur_time + dur
        for sender in self.senders:
            sender.reset_obs()

        while self.cur_time < end_time:
            event_time, sender, event_type, next_hop, cur_latency, dropped = heapq.heappop(self.q)
            #print("######### cur_time %f, end time ########", self.cur_time, end_time)
            #print("##Got event %s from sender to link %d, latency %f at time %f q size = %d" % (event_type, next_hop, cur_latency, event_time, len(self.q)))
            self.cur_time = event_time
            # new event for next packet
            new_event_time = event_time
            new_event_type = event_type
            new_next_hop = next_hop
            new_latency = cur_latency
            new_dropped = dropped
            push_new_event = False # packet 손실시 push 하지 않음

            # event type == SEND
            # 1. send packet to dest through link0
            if event_type == EVENT_TYPE_SEND:
                if next_hop == 0:
                    #print("next_hop = 0")
                    #print("Packet sent at time %f" % self.cur_time)
                    if sender.can_send_packet():
                        sender.on_packet_sent()
                        push_new_event = True
                    heapq.heappush(self.q, (self.cur_time + (1.0 / sender.rate), sender, EVENT_TYPE_SEND, 0, 0.0, False))
                else:
                    # this situation not happen
                    push_new_event = True

                if next_hop == sender.dest:
                    #print("next_hop = seder.dest, event type S -> A")
                    new_event_type = EVENT_TYPE_ACK
                new_next_hop = next_hop + 1
                
                link_latency = sender.path[next_hop].get_cur_latency(self.cur_time)
                if USE_LATENCY_NOISE:
                    link_latency *= random.uniform(1.0, MAX_LATENCY_NOISE)
                new_latency += link_latency
                new_event_time += link_latency
                new_dropped = not sender.path[next_hop].packet_enters_link(self.cur_time) 
                                    # if dropeed = True
            
            # event type == ACK
            if event_type == EVENT_TYPE_ACK:
                # 3. packet ack or drop
                if next_hop == len(sender.path): # len(sender,path) = 2
                    if dropped:
                        sender.on_packet_lost()
                        #print("Packet lost at time %f" % self.cur_time)
                    else:
                        sender.on_packet_acked(cur_latency)
                        #print("Packet acked at time %f" % self.cur_time)
                # 2. send ack to sender through link[1]
                else:
                    new_next_hop = next_hop + 1
                    link_latency = sender.path[next_hop].get_cur_latency(self.cur_time)
                    if USE_LATENCY_NOISE:
                        link_latency *= random.uniform(1.0, MAX_LATENCY_NOISE)
                    new_latency += link_latency
                    new_event_time += link_latency
                    push_new_event = True
                   
            if push_new_event:
                #print("##Push event %s from sender to link %d, latency %f at time %f" % (new_event_type, new_next_hop, new_latency, new_event_time))
                heapq.heappush(self.q, (new_event_time, sender, new_event_type, new_next_hop, new_latency, new_dropped))

        sender_mi = self.senders[0].get_run_data()
        throughput = sender_mi.get("recv rate")  # sender_obs
        latency = sender_mi.get("avg latency")  
        loss = sender_mi.get("loss ratio")
        bw_cutoff = self.links[0].bw * 0.8
        lat_cutoff = 2.0 * self.links[0].dl * 1.5
        loss_cutoff = 2.0 * self.links[0].lr * 1.5
        #print("thpt %f, bw %f" % (throughput, bw_cutoff))
        #reward = 0 if (loss > 0.1 or throughput < bw_cutoff or latency > lat_cutoff or loss > loss_cutoff) else 1 #
        
        # Super high throughput
        #reward = REWARD_SCALE * (20.0 * throughput / RATE_OBS_SCALE - 1e3 * latency / LAT_OBS_SCALE - 2e3 * loss)
        
        # Very high thpt = defualut = model_A
        reward = (10.0 * throughput / (8 * BYTES_PER_PACKET) - 1e3 * latency - 2e3 * loss)
        # High thpt
        #reward = REWARD_SCALE * (5.0 * throughput / RATE_OBS_SCALE - 1e3 * latency / LAT_OBS_SCALE - 2e3 * loss)
        
        # Low latency = model_B
        #reward = REWARD_SCALE * (2.0 * throughput / RATE_OBS_SCALE - 1e3 * latency / LAT_OBS_SCALE - 2e3 * loss)
        #reward = (2.0 * throughput / (8 * BYTES_PER_PACKET) - 10e3 * latency - 2e3 * loss)
        #if reward > 857:
        #print("Reward = %f, thpt = %f, lat = %f, loss = %f" % (reward, throughput, latency, loss))

        #Bad Model testing = model_bad
        #reward = - (10.0 * throughput / (8 * BYTES_PER_PACKET) - 1e3 * latency - 2e3 * loss)

        #reward = (throughput / RATE_OBS_SCALE) * np.exp(-1 * (LATENCY_PENALTY * latency / LAT_OBS_SCALE + LOSS_PENALTY * loss))
        return reward * REWARD_SCALE

class Sender():
    
    #self.senders = [Sender(random.uniform(0.3, 1.5) * bw, [self.links[0], self.links[1]], 
    #                       0, self.features, history_len=self.history_len)]
    def __init__(self, rate, path, dest, features, cwnd=25, history_len=HISTORY_LEN):
        self.id = Sender._get_next_id()
        self.starting_rate = rate
        self.rate = rate                # random.uniform(0.3, 1.5) * bw
        self.sent = 0
        self.acked = 0
        self.lost = 0
        self.bytes_in_flight = 0
        self.min_latency = None
        self.rtt_samples = []
        self.sample_time = []
        self.net = None
        self.path = path                # [self.links[0], self.links[1]] LIink
        self.dest = dest                # 0
        self.history_len = history_len  
        self.features = features        #['sent latency inflation', 'latency ratio', 'send ratio']
        self.history = sender_obs.SenderHistory(self.history_len,
                                                self.features, self.id)
        self.cwnd = cwnd
        
    _next_id = 1
    def _get_next_id():
        result = Sender._next_id
        Sender._next_id += 1
        return result

    # change sending rate
    def apply_rate_delta(self, delta):
        delta *= DELTA_SCALE  # 0.025 * action
        #print("Applying delta %f" % delta)
        
        if delta >= 0.0:
            self.set_rate(self.rate * (1.0 + delta))
        else:
            self.set_rate(self.rate / (1.0 - delta))
        '''
        #bad model
        if delta >= 0.0:
            self.set_rate(self.rate * (1.0 + delta))
        else:
            self.set_rate(self.rate * (1.0 - delta))
        '''

    def apply_cwnd_delta(self, delta):
        delta *= DELTA_SCALE
        #print("Applying delta %f" % delta)
        if delta >= 0.0:
            self.set_cwnd(self.cwnd * (1.0 + delta))
        else:
            self.set_cwnd(self.cwnd / (1.0 - delta))

    def set_rate(self, new_rate):
        self.rate = new_rate
        #print("Attempt to set new rate to %f (min %f, max %f)" % (new_rate, MIN_RATE, MAX_RATE))
        if self.rate > MAX_RATE:
            self.rate = MAX_RATE
        if self.rate < MIN_RATE:
            self.rate = MIN_RATE

    def set_cwnd(self, new_cwnd):
        self.cwnd = int(new_cwnd)
        #print("Attempt to set new rate to %f (min %f, max %f)" % (new_rate, MIN_RATE, MAX_RATE))
        if self.cwnd > MAX_CWND:
            self.cwnd = MAX_CWND
        if self.cwnd < MIN_CWND:
            self.cwnd = MIN_CWND

    def can_send_packet(self):
        if USE_CWND:
            return int(self.bytes_in_flight) / BYTES_PER_PACKET < self.cwnd
        else:
            return True

    def register_network(self, net):
        self.net = net

    # sender observation
    def on_packet_sent(self):
        self.sent += 1
        self.bytes_in_flight += BYTES_PER_PACKET

    def on_packet_acked(self, rtt):
        self.acked += 1
        self.rtt_samples.append(rtt)
        if (self.min_latency is None) or (rtt < self.min_latency):
            self.min_latency = rtt
        self.bytes_in_flight -= BYTES_PER_PACKET

    def on_packet_lost(self):
        self.lost += 1
        self.bytes_in_flight -= BYTES_PER_PACKET

    # state 계산 후 return
    def get_obs(self):
        return self.history.as_array()

    def record_run(self):
        smi = self.get_run_data()
        self.history.step(smi)  # pop old history and push new history

    # MI observation data 가져오기
    def get_run_data(self):
        obs_end_time = self.net.get_cur_time()
        
        #obs_dur = obs_end_time - self.obs_start_time
        #print("Got %d acks in %f seconds" % (self.acked, obs_dur))
        #print("Sent %d packets in %f seconds" % (self.sent, obs_dur))
        #print("self.rate = %f" % self.rate)

        return sender_obs.SenderMonitorInterval(
            self.id,
            bytes_sent=self.sent * BYTES_PER_PACKET,
            bytes_acked=self.acked * BYTES_PER_PACKET,
            bytes_lost=self.lost * BYTES_PER_PACKET,
            send_start=self.obs_start_time,
            send_end=obs_end_time,
            recv_start=self.obs_start_time,
            recv_end=obs_end_time,
            rtt_samples=self.rtt_samples,
            packet_size=BYTES_PER_PACKET
        )

    def reset_obs(self):
        self.sent = 0
        self.acked = 0
        self.lost = 0
        self.rtt_samples = []
        self.obs_start_time = self.net.get_cur_time()
        Sender._next_id = 1

    def print_debug(self):
        print("Sender:")
        print("Obs: %s" % str(self.get_obs()))
        print("Rate: %f" % self.rate)
        print("Sent: %d" % self.sent)
        print("Acked: %d" % self.acked)
        print("Lost: %d" % self.lost)
        print("Min Latency: %s" % str(self.min_latency))

    def reset(self):
        #print("Resetting sender!")
        self.rate = self.starting_rate
        self.bytes_in_flight = 0
        self.min_latency = None
        self.reset_obs()
        self.history = sender_obs.SenderHistory(self.history_len,
                                                self.features, self.id)

class SimulatedNetworkEnv(gym.Env):
    
    def __init__(self,
                 history_len=HISTORY_LEN,
                 features=arg_or_default("--input-features",
                    default="sent latency inflation,"
                          + "latency ratio,"
                          + "send ratio")):
        self.viewer = None
        self.rand = None

        # Link Parameter
        self.min_bw, self.max_bw = (100, 500)       # Bandwidth
        self.min_lat, self.max_lat = (0.05, 0.5)    # Latency
        self.min_queue, self.max_queue = (0, 8)     # Queue size
        self.min_loss, self.max_loss = (0.0, 0.05)  # Random Loss rate
        self.history_len = history_len
        #Features: ['sent latency inflation', 'latency ratio', 'send ratio']
        self.features = features.split(",") 
        #print("Features: %s" % str(self.features)) 

        self.links = None
        self.senders = None
        self.create_new_links_and_senders()         # initialize links and sender
        self.net = Network(self.senders, self.links)
        self.run_dur = None
        self.run_period = 0.1
        self.steps_taken = 0
        self.max_steps = MAX_STEPS                  # default = 400
        self.debug_thpt_changes = False
        self.last_thpt = None
        self.last_rate = None

        ##### action space
        if USE_CWND:
            self.action_space = spaces.Box(np.array([-1e12, -1e12]), np.array([1e12, 1e12]), dtype=np.float32)
        else:
            self.action_space = spaces.Box(np.array([-1e12]), np.array([1e12]), dtype=np.float32)
        print("---action_space----")
        print(self.action_space)
                   
        #### observation space
        self.observation_space = None
        use_only_scale_free = True
        single_obs_min_vec = sender_obs.get_min_obs_vector(self.features) # [-1, 1, 0]
        single_obs_max_vec = sender_obs.get_max_obs_vector(self.features) # [10, 10,000, 1,000]
        self.observation_space = spaces.Box(np.tile(single_obs_min_vec, self.history_len),
                                            np.tile(single_obs_max_vec, self.history_len),
                                            dtype=np.float32)
        print("---observation_space----")
        print(self.observation_space)

        self.reward_sum = 0.0
        self.reward_ewma = 0.0

        self.event_record = {"Events":[]}
        self.episodes_run = -1

    def create_new_links_and_senders(self):
        bw    = random.uniform(self.min_bw, self.max_bw) # 100 ~ 500
        lat   = random.uniform(self.min_lat, self.max_lat) # 0.05 ~ 0.5
        queue = 1 + int(np.exp(random.uniform(self.min_queue, self.max_queue))) # 2 ~ 1 + e^8 (2981)
        loss  = random.uniform(self.min_loss, self.max_loss) # 0.0 ~ 0.05
        self.links = [Link(bw, lat, queue, loss), Link(bw, lat, queue, loss)]
        self.senders = [Sender(random.uniform(0.3, 1.5) * bw, [self.links[0], self.links[1]], 0, self.features, history_len=self.history_len)]
        self.run_dur = 3 * lat
        

        #self.senders = [Sender(0.3 * bw, [self.links[0], self.links[1]], 0, self.history_len)]
        #self.senders = [Sender(random.uniform(0.2, 0.7) * bw, [self.links[0], self.links[1]], 0, self.history_len)]
        #Sender(self, rate, path, dest, features, cwnd=25, history_len=10):

    def seed(self, seed=None):
        self.rand, seed = seeding.np_random(seed)
        return [seed]

    def _get_all_sender_obs(self):
        sender_obs = self.senders[0].get_obs()
        sender_obs = np.array(sender_obs).reshape(-1,)
        #print("ALl sender obs: ", sender_obs)
        return sender_obs

    # step
    def step(self, actions):
        #print("Actions: %s" % str(actions))     # acitons = Rate Change Factor #print(actions)
        #self.print_debug()
        #print("Step: %d" %self.steps_taken )

        # change transmission rate
        for i in range(0, 1): #len(actions)):
            #print("Updating rate for sender %d" % i)
            action = actions    
            self.senders[i].apply_rate_delta(action[0])     # action[0] = rate 조절
            if USE_CWND:
                self.senders[i].apply_cwnd_delta(action[1]) # action[1] = cwnd 조절

        #print("Running for %fs" % self.run_dur)    # run_dur = 초기값 3 * lat
        reward = self.net.run_for_dur(self.run_dur)  # reward
        for sender in self.senders:
            sender.record_run()
        self.steps_taken += 1
        sender_obs = self._get_all_sender_obs()  # state

        sender_mi = self.senders[0].get_run_data()
        event = {}
        event["Name"] = "Step"
        event["Time"] = self.steps_taken
        event["Reward"] = reward
        #event["Target Rate"] = sender_mi.target_rate
        event["Send Rate"] = sender_mi.get("send rate")
        event["Throughput"] = sender_mi.get("recv rate")
        event["Latency"] = sender_mi.get("avg latency")
        event["Loss Rate"] = sender_mi.get("loss ratio")
        event["Latency Inflation"] = sender_mi.get("sent latency inflation")
        event["Latency Ratio"] = sender_mi.get("latency ratio")
        event["Send Ratio"] = sender_mi.get("send ratio")
        #event["Cwnd"] = sender_mi.cwnd
        #event["Cwnd Used"] = sender_mi.cwnd_used
        self.event_record["Events"].append(event)
        if event["Latency"] > 0.0:
            self.run_dur = 0.5 * sender_mi.get("avg latency")
            #print("run_dur control : %f" %self.run_dur)
        #print("Sender obs: %s" % sender_obs)
        should_stop = False

        self.reward_sum += reward
        return sender_obs, reward, (self.steps_taken >= self.max_steps or should_stop), {}

    def print_debug(self):
        print("---Link Debug---")
        for link in self.links:
            link.print_debug()
        print("---Sender Debug---")
        for sender in self.senders:
            sender.print_debug()

    def reset(self):
        self.steps_taken = 0
        self.net.reset()
        self.create_new_links_and_senders()
        self.net = Network(self.senders, self.links)
        self.episodes_run += 1
        if self.episodes_run > 0 and self.episodes_run % 100 == 0:
            self.dump_events_to_file("pcc_env_log_run_%d.json" % self.episodes_run)
            #self.dump_events_to_txt("pcc_env_log_run_%d.txt" % self.episodes_run)
        self.event_record = {"Events":[]}
        self.net.run_for_dur(self.run_dur)   # 이건 왜있징
        self.net.run_for_dur(self.run_dur)
        self.reward_ewma *= 0.99
        self.reward_ewma += 0.01 * self.reward_sum
        print("Reward: %0.2f, Ewma Reward: %0.2f" % (self.reward_sum, self.reward_ewma))
        self.reward_sum = 0.0
        return self._get_all_sender_obs()

    def render(self, mode='human'):
        pass

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def dump_events_to_file(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.event_record, f, indent=4)

    def dump_events_to_txt(self, filename):
        with open(filename, 'w') as f:
            f.write(self.event_record)

register(id='PccNs-v0', entry_point='network_sim:SimulatedNetworkEnv')
#env = SimulatedNetworkEnv()
#env.step([1.0])
