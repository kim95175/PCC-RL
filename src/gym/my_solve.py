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
import network_sim
import os
import tensorflow.compat.v1 as tf
os.environ['TF_CPP_MIN_LOG_LEVE'] = '2'
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
#from stable_baselines.common.policies import MlpPolicy
#from stable_baselines.common.policies import FeedForwardPolicy
#from stable_baselines import PPO1
from stable_baselines.sac.policies import MlpPolicy
from stable_baselines.sac.policies import FeedForwardPolicy
from stable_baselines import SAC
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from common.simple_arg_parse import arg_or_default

arch_str = arg_or_default("--arch", default="32,16")
if arch_str == "":
    arch = []
else:
    arch = [int(layer_width) for layer_width in arch_str.split(",")]
print("Architecture is: %s" % str(arch))

training_sess = None


class MyMlpPolicy(FeedForwardPolicy):

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **_kwargs):
        super(MyMlpPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse, net_arch=[{"pi":arch, "vf":arch}],
                                        feature_extraction="mlp", **_kwargs)
        global training_sess
        training_sess = sess

class CustomSACPolicy(FeedForwardPolicy):

    def __init__(self, *args, **_kwargs):
        super(CustomSACPolicy, self).__init__(*args, **_kwargs, layers=[arch[0], arch[1]],
                                                layer_norm=False, feature_extraction="mlp")
        

def stable_solve(model_name, step):
    env = gym.make('PccNs-v0')

    gamma = arg_or_default("--gamma", default=0.99)
    print("gamma = %f" % gamma)
    #model = PPO1(MyMlpPolicy, env, verbose=1, schedule='constant', 
    #            timesteps_per_actorbatch=8192, optim_batchsize=2048, gamma=gamma)
    model = SAC(CustomSACPolicy, env, gamma=gamma, verbose=1, batch_size=2048)
    #global training_sess
    #training_sess = CustomSACPolicy.sess

    ####for i in range(0, 6):
    for i in range(0, 1):
        #with model.graph.as_default():                                                                   
        #    saver = tf.train.Saver()                                                                     
        #    saver.save(training_sess, "./sac_model_%d.ckpt" % i)
        #model.learn(total_timesteps=(1600 * 410))
        #model.learn(total_timesteps=(1600 * 1600))
        model.learn(total_timesteps=(step*1600))

    #Save the model to the location specified below.
    ##
    default_export_dir = "/home/airman/Github/PCC-RL/src/pcc_saved_models/" + model_name + "/"
    export_dir = arg_or_default("--model-dir", default=default_export_dir)
    with model.graph.as_default():

        pol = model.policy_tf #act_model

        obs_ph = pol.obs_ph
        act = pol.deterministic_action
        sampled_act = pol.action

        obs_input = tf.saved_model.utils.build_tensor_info(obs_ph)
        outputs_tensor_info = tf.saved_model.utils.build_tensor_info(act)
        stochastic_act_tensor_info = tf.saved_model.utils.build_tensor_info(sampled_act)
        signature = tf.saved_model.signature_def_utils.build_signature_def(
            inputs={"ob":obs_input},
            outputs={"act":outputs_tensor_info, "stochastic_act":stochastic_act_tensor_info},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

        #"""
        signature_map = {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                        signature}

        model_builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
        model_builder.add_meta_graph_and_variables(model.sess,
            tags=[tf.saved_model.tag_constants.SERVING],
            signature_def_map=signature_map,
            clear_devices=True)
        model_builder.save(as_text=True)


def testing_with_model():
    pass

while True:
    print()
    print("1. Training model")
    print("2. Testing with model")
    print("3. Exit")
    menu = int(input("select: "))

    if menu == 1:
        model_name = str(input("model name: "))
        time_step = int(input("step (*1600): "))
        stable_solve(model_name, time_step)
    elif menu == 2:
        pass
    elif menu == 3:
        break
    else:
        print("wrong input!")