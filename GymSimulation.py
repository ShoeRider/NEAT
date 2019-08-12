import gym
import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean, median

LR = 1e-3

goal_steps = 1000
score_requirement = -100


def some_random_games_first(Simulation):
    env = gym.make(Simulation)
    env.reset()
    uniqueActions = {}
    for episode in range(5):
        env.reset()
        for t in range(goal_steps):
            env.render()
            action = env.action_space.sample()
            if not action in uniqueActions:
                uniqueActions[action] = 1
            observation, reward, done, info = env.step(action)

            if done:
                break
    print(len(observation))
    print(observation)
    print(uniqueActions)

if(False):
    some_random_games_first('Pong-ram-v0')
    #some_random_games_first('Assault-ram-v0')




def Initialized_Simulations(Tuple):
    Simulation,Simulations,Range = Tuple
    env = gym.make(Simulation)
    env.reset()

    training_data = []
    scores = []
    accepted_scores = []
    for _ in range(Simulations):
        score = 0
        Temp_Game = []
        prev_observation = []
        for _ in range(goal_steps):
            #env.render()
            action = random.randrange(0,Range)
            observation, reward, done, info = env.step(action)

            if len(prev_observation) > 0:
                Temp_Game.append([prev_observation, action])

            prev_observation = observation
            score += reward
            if done:
                break

        if score >= score_requirement:
            accepted_scores.append(score)

            for data in Temp_Game:
                if(False):
                    print(data)
                    if data[1] == 1:
                        output = [0,1]
                    elif data[1] == 0:
                        output = [1,0]

                training_data.append([data[0], data[1]])
        env.reset()
        scores.append(score)

        training_data_save = np.array(training_data)
        np.save('saved.npy',training_data_save)

    if(True):
        print('average accepted scores:', np.mean(accepted_scores))
        print('median accepted scores:',np.median(accepted_scores))
        #print('counter;',accepted_scores)
    return training_data_save

import os.path
import json
import numpy


# TrainingData: Dictionary with associated values
# contains:
# "Iteration"
def Save_TrainingData(TrainingData):
    if not os.path.exists(TrainingData["Game"]+"/"):
        os.makedirs(TrainingData["Game"]+"/")

    Location = TrainingData["Game"]+"/"
    JSON_File = Location+TrainingData["Game"]+"_"+str(TrainingData["Iteration"])+".json"
    CSV_File  = Location+TrainingData["Game"]+"_"+str(TrainingData["Iteration"])+".csv"

    for Observation,Instance in zip(TrainingData["Observations"],range(len(TrainingData["Observations"]))):
        CSV_File  = Location+TrainingData["Game"]+"_"+str(TrainingData["Iteration"])+"_"+str(Instance)+".csv"
        Observation.tofile(CSV_File,sep=',')

    TrainingData["Observations"] = CSV_File
    with open(JSON_File, 'w') as outfile:
        json.dump(TrainingData, outfile)




def Read_Simulations():
    return 0



if(True):
    TrainingData = {}
    Observations = Initialized_Simulations(('Pong-ram-v0',10,5))
    TrainingData["Game"] = 'Pong-ram-v0'
    TrainingData["Observations"] = Observations
    TrainingData["Iteration"] = 0
    Save_TrainingData(TrainingData)



import multiprocessing as mp
def Parallel_Initial_Population(Simulation_Name,Simulations):

    print("Simulations:",Simulations)
    #CoreCount = mp.cpu_count()
    CoreCount = 10
    SimulationsPerCore = int(Simulations/CoreCount)

    print("Number of Threads: ", CoreCount, " with ",SimulationsPerCore," simulations/core")
    pool = mp.Pool(CoreCount)
    Thread_results = pool.map(Initialized_Simulations,[(Simulation_Name,SimulationsPerCore,5) for x in range(CoreCount)])
    pool.close()

    Results = []
    #condense different games into one list
    for Result_Set in Thread_results:
        for Item in Result_Set:
            Results.append(Item)

    return Result_Set


if(False):
    print(Parallel_Initial_Population('CartPole-v0',60000))
    print(initial_population(('CartPole-v0',1000)))

if(False):
    print(Parallel_Initial_Population('Pong-ram-v0',10))
    print(initial_population(('Pong-ram-v0',10,5)))




#Create different Models:
def neural_network_model(input_size):
    network = input_data(shape = [None, input_size,1], name ='input')

    network = fully_connected(network,128,activation='relu')
    network = dropout(network,0.8)

    network = fully_connected(network,256,activation='relu')
    network = dropout(network,0.8)

    network = fully_connected(network,512,activation='relu')
    network = dropout(network,0.8)

    network = fully_connected(network,256,activation='relu')
    network = dropout(network,0.8)

    network = fully_connected(network,128,activation='relu')
    network = dropout(network,0.8)

    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=LR,
                         loss='categorical_crossentropy',name='targets')

    model = tflearn.DNN(network, tensorboard_dir='log')

    return model



def train_model(training_data, model= False):
    print("(",len(training_data),",",")")
    print(training_data)
    X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0]),1)
    Y = [i[1] for i in training_data]

    if not model:
        model = neural_network_model(input_size = len(X[0]))

    model.fit({'input': X}, {'targets': Y}, n_epoch=5, snapshot_step=500, show_metric=True, run_id='openai_learning')
    return model


#train_model(initial_population(Simulation_Name))



def Simulate_Model(model,Simulation):
    env = gym.make(Simulation)
    env.reset()

    scores = []
    choices = []

    for each_game in range(10):
        score = 0
        game_memory = []
        prev_obs = []
        env.reset()
        for _ in range(goal_steps):
            env.render()
            if len(prev_obs) ==0:
                action = random.randrange(0,2)
            else:
                action = np.argmax(model.predict(prev_obs.reshape(-1,len(prev_obs),1))[0])
            choices.append(action)

            new_observation, reward, done, info = env.step(action)
            prev_obs = new_observation
            game_memory.append([new_observation, action])
            score += reward
            if done:
                break
        scores.append(score)
    print('Average Score:', sum(scores)/len(scores))
    return sum(scores)/len(scores)



def Test_Simulation(Simulation):
    Simulation_Name = Simulation
    TrainedModel = train_model(Parallel_Initial_Population(Simulation,10))

    #TODO Save and load Models
    #TrainedModel.save(Simulation_Name+".model")
    #TrainedModel.load(Simulation_Name+".model")
    average = Simulate_Model(TrainedModel,Simulation)

    TrainedModel.save(Simulation+average+".model")

#Test_Simulation('CartPole-v1')
#Test_Simulation('Pong-v0')
