import os
import copy
import settings
import random
from collections import deque
import numpy as np
from .model import DQN
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

#进行最初始化的设置
def setup(self):
    #载入的模型名称
    model_name="model.pt"
    #如果是train模式并且存在模型则load，没有则重新创建
    if self.train or not os.path.isfile(model_name):
        self.logger.info("Setting up model from scratch.")
        self.model = DQN()
        if os.path.exists(model_name):
            self.model.load_model(model_name)
        self.model.switch_train_mode()
    else:
        #测试模式直接载入模型
        self.logger.info("Loading model from saved state.")
        self.model = DQN()
        self.model.load_model(model_name)
        self.model.switch_eval_mode()


def act(self, game_state: dict):
    #根据自己设定的state获取特征，传入网络得到
    features = state_to_features(game_state)
    #再传入模型得到行动
    action = self.model.action(features, game_state)
    #由于我们在单个场景模型收敛的时候，其他的选择会变得特别少，所以需要进行真随机的扰动
    #从而跳出某个场景的局部最优
    # if random.uniform(0, 10)<=1:
    #     action=ACTIONS[ random.randint(0, 5)]
    #返回action，游戏下一步行动
    return action


def state_to_features(game_state: dict) -> np.array:
    #如果没有game state则出错
    if game_state is None:
        return None
    #6*17*17的网络，值得范围在0-1
    channels = []
    #获取矩阵形状
    H, W = game_state['field'].shape
    def is_valid(x,y):
        return 0<=x<H and 0<=y<W

    #1,H,W -> postion matrix,close means can kill other agent
    #将所有人的位置形成一个1*17*17的矩阵
    agent_postion=np.zeros((H,W))
    agent_postion[game_state['self'][3][0]][game_state['self'][3][1]]=1
    for other_agent in game_state['others']:
        agent_postion[other_agent[3][0]][other_agent[3][1]]= (abs(other_agent[3][0]-game_state['self'][3][0])+abs(other_agent[3][1]-game_state['self'][3][1]))/(H+W)
    channels.append(agent_postion)
   
    #1,H,W -> bomb matrix, remain time is higher the value is higher
    #将炸弹倒计时和爆炸时间进行累加
    #如果炸弹倒计时是3，爆炸持续时间是5，那么这一格子的值为(3+5)/(爆炸等待最大值和爆炸持续最大值)
    field_copy=copy.deepcopy(game_state['field'])
    bomb_explosion_matrix=np.zeros((H,W))
    bomb_max_value=settings.BOMB_TIMER+settings.EXPLOSION_TIMER
    left_power_list=[[0,i] for i in range(-1,-settings.BOMB_POWER-1,-1)]
    right_power_list=[[0,i] for i in range(1,settings.BOMB_POWER+1)]
    up_power_list=[[i,0] for i in range(-1,-settings.BOMB_POWER-1,-1)]
    down_power_list=[[i,0] for i in range(1,settings.BOMB_POWER+1)]
    for bomb in game_state['bombs']:
        bomb_x,bomb_y=bomb[0][0],bomb[0][1]
        bomb_remain_time=(bomb[1]+settings.EXPLOSION_TIMER)/bomb_max_value   
        #多个炸弹的倒计时取最小值
        for [x,y] in left_power_list:
            if is_valid(bomb_x+x,bomb_y+y):
                if field_copy[bomb_x+x][bomb_y+y]==-1:
                    break
                bomb_explosion_matrix[bomb_x+x][bomb_y+y]=bomb_remain_time
        for [x,y] in right_power_list:
            if is_valid(bomb_x+x,bomb_y+y):
                if field_copy[bomb_x+x][bomb_y+y]==-1:
                    break
                bomb_explosion_matrix[bomb_x+x][bomb_y+y]=bomb_remain_time
        for [x,y] in up_power_list:
            if is_valid(bomb_x+x,bomb_y+y):
                if field_copy[bomb_x+x][bomb_y+y]==-1:
                    break
                bomb_explosion_matrix[bomb_x+x][bomb_y+y]=bomb_remain_time
        for [x,y] in down_power_list:
            if is_valid(bomb_x+x,bomb_y+y):
                if field_copy[bomb_x+x][bomb_y+y]==-1:
                    break
                bomb_explosion_matrix[bomb_x+x][bomb_y+y]=bomb_remain_time
        bomb_explosion_matrix[bomb_x][bomb_y]=bomb_remain_time
    explosion_position = np.where(game_state['explosion_map'] != 0)
    for coord in zip(explosion_position[0], explosion_position[1]):
        bomb_explosion_matrix[coord[0]][coord[1]]=game_state['explosion_map'][coord[0]][coord[1]]/bomb_max_value
    channels.append(bomb_explosion_matrix)

    #1,H,W -> bomb distance , faraway and safer.
    #炸弹和爆炸区域里玩家得距离，越远值越高
    bomb_explosion_distance=np.zeros((H,W))
    for i in range(H):
        for j in range(W):
            if bomb_explosion_matrix[i][j]!=0:
                bomb_explosion_distance[i][j]=1- (abs(i-game_state['self'][3][0])+abs(j-game_state['self'][3][1]))/(H+W)

    channels.append(bomb_explosion_distance)

    # field -> crate wall
    #墙和箱子得，墙为0，箱子为0.5
    field=np.array(copy.deepcopy( game_state['field']),dtype='float')
    field[field==0]=0.5
    field[field==-1]=0
    channels.append(field)

    #coin 
    #金币得位置
    coins=game_state['coins']
    coins_value=np.zeros((H,W))
    for coin_position in coins:
        coins_value[coin_position[0]][coin_position[1]]= (abs(coin_position[0]-game_state['self'][3][0])+abs(coin_position[1]-game_state['self'][3][1]))/(H+W)
    
    channels.append(coins_value)
    
    #可行区域,可以达到的区域为0.5,有金币的为1
    can_arrive_area=np.zeros((H,W))
    field_map=np.array(copy.deepcopy(game_state['field']),dtype='float')
    for i in range(H):
        for j in range(W):
            if bomb_explosion_matrix[i][j]!=0 and field_map[i][j]==0:
                field_map[i][j] = bomb_explosion_matrix[i][j]
    for coord in zip(explosion_position[0], explosion_position[1]):
        field_map[coord[0]][coord[1]]=explosion_position[0][1]/(settings.BOMB_TIMER+settings.EXPLOSION_TIMER)
    current_postion=game_state['self'][3]

    can_arrive_area=dfs(current_postion,field_map,can_arrive_area,coins)

    channels.append(can_arrive_area)

    stacked_channels = np.stack(channels)

    return stacked_channels

def is_valid(field_matrix,x, y):
    # 检查坐标(x, y)是否在矩阵范围内，以及是否是障碍格子
    H,W=field_matrix.shape
    return 0 <= x < H and 0 <= y < W and field_matrix[x][y] != 1 and field_matrix[x][y]!=-1

def dfs(start_postion,field_matrix,can_arrive_area,coin_postion):
    stack = [((start_postion[0], start_postion[1]),0)]
    can_arrive_area[start_postion[0]][start_postion[1]]=0.5
    while stack:
        (x, y),dis = stack.pop()  # 出栈
        # 检查当前格子(x, y)周围的格子
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            new_x, new_y = x + dx, y + dy
            if is_valid(field_matrix,new_x, new_y) and can_arrive_area[new_x][new_y]==0:
                if field_matrix[new_x][new_y]!=0:
                    time=field_matrix[new_x][new_y]*(settings.BOMB_TIMER+settings.EXPLOSION_TIMER)
                    time=time-settings.EXPLOSION_TIMER
                    if time<=0:
                        continue
                    if time<=dis+1:
                        continue
                stack.append(((new_x, new_y),dis+1))  # 将未访问的邻居格子入栈
                if len(coin_postion)!=0 and  (new_x,new_y) in coin_postion:
                    can_arrive_area[new_x][new_y]=1
                else:
                    can_arrive_area[new_x][new_y]=0.5
    return can_arrive_area

"""
old state function,suit for current best model
"""
# def state_to_features(game_state: dict) -> np.array:
#     #如果没有game state则出错
#     if game_state is None:
#         return None

#     #6*17*17的网络，值得范围在0-1
#     channels = []
#     #获取矩阵形状
#     H, W = game_state['field'].shape

#     #1,H,W -> postion matrix,close means can kill other agent
#     #将所有人的位置形成一个1*17*17的矩阵
#     agent_postion=np.zeros((H,W))
#     agent_postion[game_state['self'][3][0]][game_state['self'][3][1]]=1
#     for other_agent in game_state['others']:
#         agent_postion[other_agent[3][0]][other_agent[3][1]]= (abs(other_agent[3][0]-game_state['self'][3][0])+abs(other_agent[3][1]-game_state['self'][3][1]))/(H+W)
#     channels.append(agent_postion)

#     #1,H,W -> bomb matrix, remain time is higher the value is higher
#     #将炸弹倒计时和爆炸时间进行累加
#     #如果炸弹倒计时是3，爆炸持续时间是5，那么这一格子的值为(3+5)/(爆炸等待最大值和爆炸持续最大值)
#     bomb_explosion_matrix=np.zeros((H,W))
#     bomb_max_value=settings.BOMB_TIMER+settings.EXPLOSION_TIMER
#     bomb_power=settings.BOMB_POWER
#     for bomb in game_state['bombs']:
#         bomb_x,bomb_y=bomb[0][0],bomb[0][1]
#         bomb_remain_time=(bomb[1]+settings.EXPLOSION_TIMER)/bomb_max_value
#         power_points = [[i, j] for i in range(-bomb_power, bomb_power + 1) for j in range(-bomb_power, bomb_power + 1) if (i == 0 or j == 0) and (i != 0 or j != 0)]
#         power_points.append([0,0])
#         for (x,y) in power_points:
#             if bomb_x+x<0 or bomb_x+x>=H or bomb_y+y<0 or bomb_y+y>=W:
#                 continue
#             bomb_explosion_matrix[bomb_x+x][bomb_y+y]=bomb_remain_time
#     explosion_position = np.where(game_state['explosion_map'] != 0)
#     for coord in zip(explosion_position[0], explosion_position[1]):
#         bomb_explosion_matrix[coord[0]][coord[1]]=game_state['explosion_map'][coord[0]][coord[1]]/bomb_max_value
#     channels.append(bomb_explosion_matrix)

#     #1,H,W -> bomb distance , faraway and safer.
#     #炸弹和爆炸区域里玩家得距离，越远值越高
#     bomb_explosion_distance=np.zeros((H,W))
#     for bomb in game_state['bombs']:
#         bomb_x,bomb_y=bomb[0][0],bomb[0][1]
#         bomb_remain_time=(bomb[1]+settings.EXPLOSION_TIMER)/bomb_max_value
#         power_points = [[i, j] for i in range(-bomb_power, bomb_power + 1) for j in range(-bomb_power, bomb_power + 1) if (i == 0 or j == 0) and (i != 0 or j != 0)]
#         power_points.append([0,0])
#         for (x,y) in power_points:
#             if bomb_x+x<0 or bomb_x+x>=H or bomb_y+y<0 or bomb_y+y>=W:
#                 continue
#             bomb_explosion_distance[bomb_x+x][bomb_y+y]=1- (abs(bomb_x+x-game_state['self'][3][0])+abs(bomb_y+y-game_state['self'][3][1]))/(H+W)
#     for coord in zip(explosion_position[0], explosion_position[1]):
#         bomb_explosion_distance[coord[0]][coord[1]]=1- (abs(coord[0]-game_state['self'][3][0])+abs(coord[1]-game_state['self'][3][1]))/(H+W)
#     channels.append(bomb_explosion_distance)

#     # field -> crate wall
#     #墙和箱子得，墙为0，箱子为0.5
#     field=copy.deepcopy( game_state['field'])
#     field[field==0]=0.5
#     field[field==-1]=0
#     channels.append(field)

#     #coin 
#     #金币得位置
#     coins=game_state['coins']
#     coins_value=np.zeros((H,W))
#     for coin_position in coins:
#         coins_value[coin_position[0]][coin_position[1]]= (abs(coin_position[0]-game_state['self'][3][0])+abs(coin_position[1]-game_state['self'][3][1]))/(H+W)
    
#     channels.append(coins_value)
    
#     #可行区域,可以达到的区域为0.5,有金币的为1
#     can_arrive_area=np.zeros((H,W))
#     field_map=copy.deepcopy(game_state['field'])
#     bombs=game_state['bombs']
#     for bomb in bombs:
#         power_points = [[i, j] for i in range(-bomb_power, bomb_power + 1) for j in range(-bomb_power, bomb_power + 1) if (i == 0 or j == 0) and (i != 0 or j != 0)]
#         power_points.append([0,0])
#         for (x,y) in power_points:
#             if bomb[0][0]+x<0 or bomb[0][0]+x>=H or bomb[0][1]+y<0 or bomb[0][1]+y>=W:
#                 continue
#             field_map[bomb[0][0]+x][bomb[0][1]+y]=1
#     for coord in zip(explosion_position[0], explosion_position[1]):
#         field_map[coord[0]][coord[1]]=1
#     current_postion=game_state['self'][3]

#     can_arrive_area=dfs(current_postion,field_map,can_arrive_area,coins)

#     channels.append(can_arrive_area)

#     stacked_channels = np.stack(channels)

#     return stacked_channels

# def is_valid(field_matrix,x, y):
#     # 检查坐标(x, y)是否在矩阵范围内，以及是否是障碍格子
#     H,W=field_matrix.shape
#     return 0 <= x < H and 0 <= y < W and field_matrix[x][y] != 1 and field_matrix[x][y]!=-1
# def dfs(start_postion,field_matrix,can_arrive_area,coin_postion):
#     stack = [(start_postion[0], start_postion[1])]
#     can_arrive_area[start_postion[0]][start_postion[1]]=0.5
#     while stack:
#         x, y = stack.pop()  # 出栈
#         # 检查当前格子(x, y)周围的格子
#         for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
#             new_x, new_y = x + dx, y + dy
#             if is_valid(field_matrix,new_x, new_y) and can_arrive_area[new_x][new_y]==0:
#                 stack.append((new_x, new_y))  # 将未访问的邻居格子入栈
#                 if len(coin_postion)!=0 and  (new_x,new_y) in coin_postion:
#                     can_arrive_area[new_x][new_y]=1
#                 else:
#                     can_arrive_area[new_x][new_y]=0.5
#     return can_arrive_area