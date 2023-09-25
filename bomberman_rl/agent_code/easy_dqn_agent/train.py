from collections import namedtuple, deque
import math
import numpy as np

from typing import List
import random

import settings
import events as e
from .callbacks import state_to_features
from .utils import ACTIONS, add_events

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

# Hyper parameters -- DO modify
BATCH_SIZE= 512 #一次载入迭代的个数
MEMORY_SIZE = BATCH_SIZE*4 #回访池得个数

#全局变量
KILL_OTHER=[0,0,0]
MODEL_NUM=[0,0,0]
TAG=-1#记录所有金币被吃完得step
BEST_REWARD_AVE=-10000#记录全局最优回报
REWARD_AVERAGE=0#计算这一轮游戏得平均回报
BEST_SCORE=0#最好的分数，金币一分，杀人五分
CUR_SCORE=0#当前分数
ROUND=0#第几轮游戏
STEP=0#本轮游戏第几步
ACTIONS_NUM={"UP":0,"DOWN":0,"LEFT":0,"RIGHT":0,"WAIT":0,"BOMB":0}#记录操作得字典
REWARD_NUM={'COIN_COLLECTED':0,#记录reward的字典
            'KILLED_OPPONENT':0,
            'BOMB_DROPPED':0,
            'CRATE_DESTROYED':0,
            'COIN_FOUND':0,
            'SURVIVED_ROUND':0,
            'KILLED_SELF':0,
            'WAITED':0,
            'INVALID_ACTION':0,
            'TOWARD_COIN':0,
            'AWAY_COIN':0,
            'NEW_EXPLORATION':0,
            'REPEAT_MOVE':0,
            'TRY_TO_SAVE_SELF':0,
            'SUICIDE':0,
            'NICE_TRY_BOMB':0,
            'SUICIDE_BOMB':0,
            'NOT_USEFUL_BOMB':0,
            'DESTORY_CRATE_LEVEL1':0,
            'DESTORY_CRATE_LEVEL2':0,
            'TRY_KILL_OTHER':0,
            'SURVIVED_STEP':0,
            'SUCCESS_USE_BOMB':0,
            'MOVE_HIGH_POTENTIAL_FIELD':0,
            'MOVE_LOW_POTENTIAL_FIELD':0,
            'NOT_EAT_COIN_QUICK':0,
            'INVALID_BOMB':0,
            'GOT_KILLED':0,
            'NO_MORE_COIN':0,
            }

class ReplayMemory:#回放池
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append([*args])

    def sample(self, batch_size):
        return random.sample(self.memory, min(len(self.memory), batch_size))

    def __len__(self):
        return len(self.memory)


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.memory = ReplayMemory(MEMORY_SIZE)
    self.batch_size = BATCH_SIZE


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    #全局状态
    global CUR_SCORE,STEP

    #事件更新
    add_events(self,old_game_state,new_game_state,events,self_action,0)

    #游戏信息更新
    ACTIONS_NUM[self_action]+=1
    if e.COIN_COLLECTED in events:
        CUR_SCORE += 1
    if e.KILLED_OPPONENT in events:
        CUR_SCORE += 5
    STEP+=1
    self.memory.push(state_to_features(old_game_state),
                     ACTIONS.index(self_action),
                     state_to_features(new_game_state),
                     reward_from_events(self, events),
                     0)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    #全局状态
    global ROUND,KILL_OTHER, TAG,CUR_SCORE,REWARD_AVERAGE,STEP,ACTIONS_NUM,BEST_REWARD_AVE,REWARD_NUM,BEST_SCORE,MODEL_NUM
    
    # 事件更新
    add_events(self,last_game_state,last_game_state,events,None,1)

    #游戏和训练信息更新
    self.memory.push(state_to_features(last_game_state),
                     ACTIONS.index(last_action),
                     state_to_features(last_game_state),
                     reward_from_events(self, events),
                     1)

    self.model.count = len(self.memory)

    for i in range(len(self.memory) - 2, -1, -1):
        if self.memory.memory[i][-1] == 1:
            break
    eps_i = i + 1
    while eps_i < len(self.memory):
        next_10_reward = 0
        discount = 0.9
        # 未来5步奖励
        for i in range(min(eps_i + 10, len(self.memory) - 1), eps_i - 1, -1):
            next_10_reward *= discount
            next_10_reward += self.memory.memory[i][3]
        self.memory.memory[eps_i][3] = next_10_reward
        eps_i += 1

    steps = 2 * (len(self.memory) // self.batch_size + 1)
    for _ in range(steps):
        memorys = self.memory.sample(self.batch_size)
        obs, action, next_obs, reward, done = [], [], [], [], []
        for transition in memorys:
            obs.append(transition[0])
            action.append(transition[1])
            next_obs.append(transition[2])
            reward.append(transition[3])
            done.append(transition[4])

        obs = np.stack(obs, axis=0)
        action = np.array(action)
        next_obs = np.stack(next_obs, axis=0)
        reward = np.array(reward)
        done = np.array(done)

    #日志信息更新 
    
    if e.COIN_COLLECTED in events:
        CUR_SCORE += 1
    if e.KILLED_OPPONENT in events:
        CUR_SCORE += 5

    self.logger.debug("{} round is end,totaly {} steps,gain {} scores,average reward is {}.".format(ROUND,STEP,CUR_SCORE,REWARD_AVERAGE/400))
    self.logger.debug("{:^10}{:^10}{:^10}{:^10}{:^10}{:^10}".format("UP","DOWN","LEFT","RIGHT","WAIT","BOMB"))
    self.logger.debug("{:^10}{:^10}{:^10}{:^10}{:^10}{:^10}".format(ACTIONS_NUM["UP"],ACTIONS_NUM["DOWN"],ACTIONS_NUM["LEFT"],ACTIONS_NUM["RIGHT"],ACTIONS_NUM["WAIT"],ACTIONS_NUM["BOMB"]))

    self.logger.debug("{:^20}{:^20}{:^20}{:^20}{:^20}{:^20}{:^20}{:^20}{:^20}{:^20}{:^20}{:^20}{:^20}{:^20}{:^20}{:^20}{:^20}{:^20}{:^20}{:^20}{:^20}{:^20}{:^20}{:^20}{:^20}{:^20}{:^20}{:^20}".format('COIN_COLLECTED','KILLED_OPPONENT','BOMB_DROPPED','CRATE_DESTROYED', 'COIN_FOUND','SURVIVED_ROUND','KILLED_SELF','WAITED','INVALID_ACTION','TOWARD_COIN','AWAY_COIN','NEW_EXPLORATION','REPEAT_MOVE','TRY_TO_SAVE_SELF','SUICIDE','NICE_TRY_BOMB','SUICIDE_BOMB','NOT_USEFUL_BOMB','DESTORY_CRATE_LEVEL1','DESTORY_CRATE_LEVEL2','TRY_KILL_OTHER','SURVIVED_STEP','SUCCESS_USE_BOMB','MOVE_HIGH_POTENTIAL_FIELD','MOVE_LOW_POTENTIAL_FIELD','NOT_EAT_COIN_QUICK','INVALID_BOMB','NO_MORE_COIN'))
    self.logger.debug("{:^20}{:^20}{:^20}{:^20}{:^20}{:^20}{:^20}{:^20}{:^20}{:^20}{:^20}{:^20}{:^20}{:^20}{:^20}{:^20}{:^20}{:^20}{:^20}{:^20}{:^20}{:^20}{:^20}{:^20}{:^20}{:^20}{:^20}{:^20}".format(REWARD_NUM['COIN_COLLECTED'],REWARD_NUM['KILLED_OPPONENT'],REWARD_NUM['BOMB_DROPPED'],REWARD_NUM['CRATE_DESTROYED'], REWARD_NUM['COIN_FOUND'],REWARD_NUM['SURVIVED_ROUND'],REWARD_NUM['KILLED_SELF'],REWARD_NUM['WAITED'],REWARD_NUM['INVALID_ACTION'],REWARD_NUM['TOWARD_COIN'],REWARD_NUM['AWAY_COIN'],REWARD_NUM['NEW_EXPLORATION'],REWARD_NUM['REPEAT_MOVE'],REWARD_NUM['TRY_TO_SAVE_SELF'],REWARD_NUM['SUICIDE'],REWARD_NUM['NICE_TRY_BOMB'],REWARD_NUM['SUICIDE_BOMB'],REWARD_NUM['NOT_USEFUL_BOMB'],REWARD_NUM['DESTORY_CRATE_LEVEL1'],REWARD_NUM['DESTORY_CRATE_LEVEL2'],REWARD_NUM['TRY_KILL_OTHER'],REWARD_NUM['SURVIVED_STEP'],REWARD_NUM['SUCCESS_USE_BOMB'],REWARD_NUM['MOVE_HIGH_POTENTIAL_FIELD'],REWARD_NUM['MOVE_LOW_POTENTIAL_FIELD'],REWARD_NUM['NOT_EAT_COIN_QUICK'],REWARD_NUM['INVALID_BOMB'],REWARD_NUM['NO_MORE_COIN']))
    

    #model save
    if CUR_SCORE>=BEST_SCORE and REWARD_AVERAGE/400>=BEST_REWARD_AVE:
        BEST_SCORE=CUR_SCORE
        BEST_REWARD_AVE=REWARD_AVERAGE/400
        self.logger.debug("{} round is the current best model.".format(ROUND))
        self.model.save_model("model_best.pt".format(MODEL_NUM[0]))
        MODEL_NUM[0]+=1
    else:
        if REWARD_AVERAGE/400>=BEST_REWARD_AVE:
            self.model.save_model("model_highest_reward.pt".format(MODEL_NUM[1]))
            MODEL_NUM[1]+=1
            BEST_REWARD_AVE=REWARD_AVERAGE/400
            self.logger.debug("{} round gain current highest reward.".format(ROUND))

        if CUR_SCORE>=BEST_SCORE:
            BEST_SCORE=CUR_SCORE
            self.logger.debug("{} round gain the most score".format(ROUND))
            self.model.save_model("model_highest_score.pt".format(MODEL_NUM[2]))
            MODEL_NUM[2]+=1
    
    

    self.model.train(obs, action, next_obs, reward, done)
    self.model.save_model("model.pt")

    #重置全局变量
    REWARD_AVERAGE=0
    ACTIONS_NUM = dict.fromkeys(ACTIONS_NUM, 0)
    REWARD_NUM=dict.fromkeys(REWARD_NUM, 0)
    CUR_SCORE=0
    ROUND+=1
    STEP=0
    TAG=-1
    KILL_OTHER=[0,0,0]


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    global REWARD_AVERAGE,TAG,KILL_OTHER

    if 'NO_MORE_COIN' not in events:
        game_rewards = {
            #每一步都一定会有的reward
            e.TOWARD_COIN:0.2,
            e.AWAY_COIN:-0.5,

            e.MOVE_HIGH_POTENTIAL_FIELD:0.1,
            e.MOVE_LOW_POTENTIAL_FIELD:-0.3,

            e.SURVIVED_STEP:0.001,

            #判断触发
            e.BOMB_DROPPED: 0.8, #放下炸弹
            e.COIN_COLLECTED: 15, #收集到金币
            e.KILLED_OPPONENT: 40,  #杀死对方
            e.GOT_KILLED:-1*STEP, 
            e.CRATE_DESTROYED: 0.4+0.45*math.exp((250-STEP)/200)/2, #摧毁墙壁
            e.COIN_FOUND: 0.7, #找到金币
            e.SURVIVED_ROUND:40, #活着的奖励
            #即在120步之内炸死自己的损失会非常高，反之越低 
            e.KILLED_SELF: -1*STEP, #杀死自己
            e.WAITED: -0.52, #原地等待
            e.INVALID_ACTION:-0.85, #无效操作

            e.NEW_EXPLORATION:0.3+0.2*math.exp((400-STEP)/200)/2,#探索新的区域
            e.REPEAT_MOVE:-0.5-0.5*math.exp((400-STEP)/200)/2,#重复踏入区域3次

            e.TRY_TO_SAVE_SELF:0.85+0.8*math.exp(STEP/200),#在炸弹区域内都会判断
            e.SUICIDE:-1-1.3*math.exp(STEP/200),#在炸弹区域内是否远离炸弹

            e.NICE_TRY_BOMB:0.3,#放下炸弹的动作判断一次
            e.SUICIDE_BOMB:-1.2*STEP,#放下的炸弹无法逃生

            e.NOT_USEFUL_BOMB:-1-0.5*math.exp(STEP/200),#放下的炸弹区域内没有任何可以炸毁的东西
            e.DESTORY_CRATE_LEVEL1:4,#可以摧毁3个墙壁
            e.DESTORY_CRATE_LEVEL2:10,#可以摧毁6个墙壁
            e.TRY_KILL_OTHER:10,

            e.SUCCESS_USE_BOMB:2 if 'NOT_USEFUL_BOMB' in events else 8,#使用炸弹后没有死

            e.NOT_EAT_COIN_QUICK:-0.5-0.5/((STEP+10)/80),#没有及时的去吃金币

            e.INVALID_BOMB:-1.0,
        }
    else:#如果没有金币了，那么存活和杀人的优先级需要提高
        game_rewards = {
            #0-200的区间是0.5-1.3
            e.SURVIVED_STEP:0.05+REWARD_NUM["COIN_COLLECTED"]/9*math.exp((STEP-TAG)/200)/2,

            #判断触发
            e.BOMB_DROPPED: 0.33, #放下炸弹
            e.KILLED_OPPONENT: 80,  #杀死对方
            e.GOT_KILLED:-1.5*STEP,
            e.CRATE_DESTROYED: 0.5, #摧毁墙壁
            e.SURVIVED_ROUND:100+REWARD_NUM["COIN_COLLECTED"]/9*100*math.exp((STEP-TAG)/200)/2, #活着的奖励
            #即在120步之内炸死自己的损失会非常高，反之越低 
            e.KILLED_SELF: -2*STEP, #杀死自己
            e.INVALID_ACTION:-0.85, #无效操作

            e.NEW_EXPLORATION:0.5+REWARD_NUM["COIN_COLLECTED"]/9*0.2*math.exp((400-STEP)/200)/2,#探索新的区域
            e.REPEAT_MOVE:-0.55-0.8*math.exp((400-STEP)/200)/2,#重复踏入区域3次

            e.TRY_TO_SAVE_SELF:1,#在炸弹区域内都会判断
            e.SUICIDE:-1-1.5*math.exp(STEP/200),

            e.NICE_TRY_BOMB:0.7,#放下炸弹的动作判断一次
            e.SUICIDE_BOMB:-1*STEP,#放下的炸弹无法逃生

            e.NOT_USEFUL_BOMB:-1-0.5*math.exp(STEP/200),#放下的炸弹区域内没有任何可以炸毁的东西
            e.DESTORY_CRATE_LEVEL1:1,#可以摧毁3个墙壁
            e.DESTORY_CRATE_LEVEL2:2,#可以摧毁6个墙壁
            e.TRY_KILL_OTHER:15+REWARD_NUM["COIN_COLLECTED"]/9*15*math.exp((250-STEP)/200)/2,

            e.SUCCESS_USE_BOMB:5,#使用炸弹后没有死
            e.INVALID_BOMB:-1.0,
        }
        if TAG==-1:
            self.logger.info("{} gain all the coins".format(STEP))
            TAG=STEP
    if 'KILLED_OPPONENT' in events:
        if KILL_OTHER[0]==0:
            KILL_OTHER[0]=1
        else:
            if KILL_OTHER[1]==0:
                KILL_OTHER[1]=1
            else:
                if KILL_OTHER[2]==0:
                    KILL_OTHER[2]=1
                    
#     game_rewards = {
#             #每一步都一定会有的reward
#             e.TOWARD_COIN:0.3,
#             e.AWAY_COIN:-0.6,

#             e.MOVE_HIGH_POTENTIAL_FIELD:0.4,
#             e.MOVE_LOW_POTENTIAL_FIELD:-0.2,

#             e.SURVIVED_STEP:-0.003,

#             #判断触发
#             e.BOMB_DROPPED:1.8, #放下炸弹
#             e.COIN_COLLECTED: 18, #收集到金币
#             e.KILLED_OPPONENT: 20,  #杀死对方
#             e.GOT_KILLED:-1, 
#             e.CRATE_DESTROYED: 0.75, #摧毁墙壁
#             e.COIN_FOUND: 0.7, #找到金币
#             e.SURVIVED_ROUND:5, #活着的奖励
#             #即在120步之内炸死自己的损失会非常高，反之越低 
#             e.KILLED_SELF: -1*STEP, #杀死自己
#             e.WAITED: -0.32, #原地等待
#             e.INVALID_ACTION:-0.45, #无效操作

#             e.NEW_EXPLORATION:0.5,#探索新的区域
#             e.REPEAT_MOVE:-0.35,#重复踏入区域3次

#             e.TRY_TO_SAVE_SELF:1.2,#在炸弹区域内都会判断
#             e.SUICIDE:-1,#在炸弹区域内是否远离炸弹

#             e.NICE_TRY_BOMB:0.5,#放下炸弹的动作判断一次
#             e.SUICIDE_BOMB:-1*STEP,#放下的炸弹无法逃生

#             e.NOT_USEFUL_BOMB:-1,#放下的炸弹区域内没有任何可以炸毁的东西
#             e.DESTORY_CRATE_LEVEL1:4,#可以摧毁3个墙壁
#             e.DESTORY_CRATE_LEVEL2:10,#可以摧毁6个墙壁
#             e.TRY_KILL_OTHER:30,

#             e.SUCCESS_USE_BOMB:2,#使用炸弹后没有死

#             e.NOT_EAT_COIN_QUICK:-1,#没有及时的去吃金币

#             e.INVALID_BOMB:-1.0,
#         }
    reward_sum =0
    reward_sum+= REWARD_NUM["COIN_COLLECTED"]/4
    if sum(KILL_OTHER)==1:
        reward_sum+=1
    if sum(KILL_OTHER)==2:
        reward_sum+=1.5
    if sum(KILL_OTHER)==3:
        reward_sum+=2
    if sum(KILL_OTHER)==3 and "NO_MORE_COIN" in events:
        reward_sum+=STEP
    for event in events:
        if event in REWARD_NUM.keys():
            REWARD_NUM[event]+=1
        if event in game_rewards:
            reward_sum += game_rewards[event]
    REWARD_AVERAGE+=reward_sum
    return reward_sum
