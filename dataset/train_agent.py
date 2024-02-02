import sys
sys.path.append("../src")

import os
from matgfn.gflow.environments.sequence import SequenceEnvironment
from matgfn.gflow.agent import TrajectoryBalanceGFlowNet
from matgfn.gflow.flow_models.lstm import LSTM
from matgfn.reticular import PormakeStructureBuilder

import subprocess

import math
import torch

def property_to_reward(prop, cutoff):

    if prop < cutoff:
        return 0
    else:
        return math.exp((prop - cutoff) / cutoff)

def reward_to_property(reward, cutoff):
    if reward == 0:
        return 0
    else: 
        return math.log(reward) * cutoff + cutoff
    
def surface_area_reward(sequence, builder):
    assert sequence[-1] == "[TER]"

    mof=builder.make_pormake_mof(sequence)

    name_root='temp'
    cif_name=name_root+'.cif'
    sa_name=name_root+'.sa'
    
    mof.write_cif(cif_name)

    if os.path.exists(cif_name)==False:
        return 0

    command=(['./network']  + ['-sa'] + ['1.525'] + ['1.525'] + ['2000'] + [cif_name])
    subprocess.run(command,stdout=subprocess.DEVNULL)

    if os.path.exists(sa_name) == False:
        return 0

    lines=[]
    with open(sa_name) as result_file:
        for line in result_file:
            lines.append(line.rstrip())
    result_file.close()

    if len(lines)==0:
        return 0

    frags=lines[0].split()
    NASA=float(frags[17])
    ASA=float(frags[11])

    command=(['rm'] + [sa_name])
    subprocess.run(command)

    area = ASA + NASA
    
    return area

def build_agent(builder,cutoff):

    token_vocabulary=builder.token_vocabulary
    n_slots=builder.n_slots
    mask=builder.mask

    env = SequenceEnvironment(
        token_vocabulary=token_vocabulary,
        termination_token="[TER]", 
        reward_function=lambda s: property_to_reward(surface_area_reward(s, builder),cutoff),
        mask=mask,
        render_function=None,
        max_sequence_length=n_slots, min_sequence_length=n_slots
        )
    
    flow_model =  LSTM(token_vocabulary=token_vocabulary, n_actions=env.action_space.n)
    agent = TrajectoryBalanceGFlowNet(env, flow_model)

    return agent

def train_agent(builder, loss_threshold, run_name,cutoff):

    agent=build_agent(builder,cutoff)
    agent.train(True)

    current_loss=loss_threshold+99999

    all_observations=[]
    all_infos=[]
    all_rewards=[]
    all_losses=[]
    all_logZs=[]

    last_mean_loss=9999999

    continue_training=True

    while continue_training==True:

        observations, infos, rewards, losses, logZs = agent.fit(learning_rate=5e-3, num_episodes=5000, minibatch_size=5)

        all_observations+=observations
        all_infos+=infos
        all_rewards+=rewards
        all_losses+=losses
        all_logZs+=logZs

        test_losses=[]

        mean_loss_all_points=sum(losses)/len(losses)

        for loss in losses:
            if loss < (mean_loss_all_points*10):
                test_losses.append(loss)
                
        current_mean_loss=sum(test_losses)/len(test_losses)
        print('current loss =',current_mean_loss)

        if current_mean_loss > last_mean_loss*0.95 and current_mean_loss < last_mean_loss:
            if current_mean_loss < 50:
                continue_training=False

        if current_mean_loss < loss_threshold:
            continue_training=False

        if len(all_losses) > 79999:
            continue_training=False

        last_mean_loss=current_mean_loss

    # extra 5000 episodes just to be sure of convergence
    observations, infos, rewards, losses, logZs = agent.fit(learning_rate=5e-3, num_episodes=5000, minibatch_size=5)

    all_observations+=observations
    all_infos+=infos
    all_rewards+=rewards
    all_losses+=losses
    all_logZs+=logZs

    agent_name=run_name+'_agent.pkl'
    agent_path=os.path.join('trained_agents',agent_name)
    torch.save(agent.state_dict(), agent_path)

    training_log_name=run_name+'_training_log.txt'
    training_log_path=os.path.join('training_logs',training_log_name)

    with open(training_log_path,'w') as f:
        
        for i in range(len(all_observations)):

            line='observation --- ' + str(all_observations[i]) + ' info --- ' + str(all_infos[i]) + ' reward --- ' + str(all_rewards[i]) + ' loss --- ' + str(all_losses[i]) + ' logZ --- ' + str(all_logZs[i])
            f.write(f"{line}\n")

    f.close()

    return 

topology_names=['tsg','cdl-e','cdz-e', 'eft', 'ffc', 'tff', 'asc', 'dmg', 'dnq', 'fso', 'urj']
cutoff=5000
loss_cutoff=1.8

for name in topology_names:

    include_edges=False
    builder=PormakeStructureBuilder(topology_string=name,include_edges=include_edges)
    run_name=name+'_no_edges'
    train_agent(builder=builder,loss_threshold=loss_cutoff,run_name=run_name,cutoff=cutoff)
    print(name, 'no edges done')
        
    include_edges=True
    builder=PormakeStructureBuilder(topology_string=name,include_edges=include_edges)
    run_name=name+'_edges'
    train_agent(builder=builder,loss_threshold=loss_cutoff,run_name=run_name,cutoff=cutoff)
    print(name, 'edges done')
