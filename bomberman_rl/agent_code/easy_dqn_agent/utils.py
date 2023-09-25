import copy
from collections import deque
import numpy as np

import settings
import events

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

#会将field和炸弹和爆炸区域重组
#组成的新的矩阵，wall是-80，crate是-40，爆炸是负的倒计时，炸弹是正的倒计时
WALL=-80
CRATE=-40
TRAIL_MATRIX=np.zeros([settings.ROWS,settings.COLS]) #记录重复踏入的次数
BOMB=-90 # 用来在bfs中进行障碍物的定义
LIVE_STEP=[False,settings.BOMB_TIMER] #在炸弹爆炸后是否存活
EAT_COIN_STEP=0 #发现金币到吃金币的步数
STEPS=0 #当前步数
COIN_POSITION=set() #记录已经发现了多少个金币


def add_events(self,old_game_state,new_game_state,events,action,reset_tag):
    global TRAIL_MATRIX,LIVE_STEP,EAT_COIN_STEP,STEPS,COIN_POSITION_RECORD,COIN_POSITION,TAG

    

    #新旧所需相关变量定义
    field=old_game_state['field']
    old_position=old_game_state['self'][3]
    new_position=new_game_state['self'][3]
    coins_position=[] #[[],[],[]]
    for coin in old_game_state['coins']:
        coins_position.append([coin[0],coin[1]])
        
        coin_positon_set=set(tuple(item) for item in coins_position)
        #全局记录没有而现在有的
        set_a=set(coin_positon_set).difference(COIN_POSITION)
        COIN_POSITION=COIN_POSITION.union(set_a)
    

    coins_position_new=[] #[[],[],[]]
    for coin in new_game_state['coins']:
        coins_position_new.append([coin[0],coin[1]])

    #如果发现了9个金币，并且记录的游戏状态里没有金币了则代表金币被吃完了
    if len(COIN_POSITION)==9 and len(coins_position)==0 and len(coins_position_new)==0:
        events.append("NO_MORE_COIN")

    bombs=[]
    bomb_position=[]
    for bomb in old_game_state['bombs']:
        bombs.append([[bomb[0][0],bomb[0][1]],bomb[1]])
        bomb_position.append([bomb[0][0],bomb[0][1]])
    explosition_map=old_game_state['explosion_map']
    others=old_game_state['others']
    others_position=[]
    for other in others:
        others_position.append([other[3][0],other[3][1]])

    field_combined=combine_field_expolistion_bomb(field,bombs,explosition_map)

    field_new=new_game_state['field']
    bombs_new=[]
    bomb_position_new=[]
    for bomb in new_game_state['bombs']:
        bombs_new.append([[bomb[0][0],bomb[0][1]],bomb[1]])
        bomb_position_new.append([bomb[0][0],bomb[0][1]])
    explosition_map_new=new_game_state['explosion_map']
    others_new=new_game_state['others']

    field_combined_new=combine_field_expolistion_bomb(field_new,bombs_new,explosition_map_new)



    #1. 行动趋势的好坏
    #1.1 是否靠近金币
    if len(coins_position)!=0 or len(coins_position_new)!=0:
        if old_position!=new_position :
            #新位置与原来位置不一样，计算原来位置最近距离的硬币
            nearest_coin_postion,nearest_coin_distance=bfs(field_combined,old_position,coins_position,[WALL,CRATE,BOMB])
            check=check_object_in_explosion_bomb(nearest_coin_postion,explosition_map,bombs)
            if check[0]:
                #如果金币附近有炸弹，并且炸弹倒计时小于要到金币的距离，则到金币的距离需要更新
                if check[1]>nearest_coin_distance:
                    #爆炸持续时间大于走到的最短时间
                    nearest_coin_distance=check[1]
                if check[2]<nearest_coin_distance and check[2]+settings.EXPLOSION_TIMER>=nearest_coin_distance:
                    #等待爆炸时间要小于走到的最短时间，并且等待爆炸时间和等待时间要大于走到的最短时间
                    nearest_coin_distance=check[2]+settings.EXPLOSION_TIMER
            #新的位置到该目标硬币的最短时间
            target_coin_position_new,target_coin_distance_new=bfs(field_combined_new,new_position,[nearest_coin_postion],[WALL,CRATE,BOMB])
            check=check_object_in_explosion_bomb(target_coin_position_new,explosition_map_new,bombs_new)
            if check[0]:
                #如果金币附近有炸弹，并且炸弹倒计时小于要到金币的距离，则到金币的距离需要更新
                if check[1]>target_coin_distance_new:
                    #爆炸持续时间大于走到的最短时间
                    target_coin_distance_new=check[1]
                if check[2]<target_coin_distance_new and check[2]+settings.EXPLOSION_TIMER>=target_coin_distance_new:
                    #等待爆炸时间要小于走到的最短时间，并且等待爆炸时间和等待时间要大于走到的最短时间
                    target_coin_distance_new=check[2]+settings.EXPLOSION_TIMER
            #新的位置选取所有硬币最短的时间
            nearest_coin_postion_new,nearest_coin_distance_new=bfs(field_combined_new,new_position,coins_position_new,[WALL,CRATE,BOMB])
            check=check_object_in_explosion_bomb(nearest_coin_postion_new,explosition_map_new,bombs_new)
            if check[0]:
                #如果金币附近有炸弹，并且炸弹倒计时小于要到金币的距离，则到金币的距离需要更新
                if check[1]>nearest_coin_distance_new:
                    #爆炸持续时间大于走到的最短时间
                    nearest_coin_distance_new=check[1]
                if check[2]<nearest_coin_distance_new and check[2]+settings.EXPLOSION_TIMER>=nearest_coin_distance_new:
                    #等待爆炸时间要小于走到的最短时间，并且等待爆炸时间和等待时间要大于走到的最短时间
                    nearest_coin_distance_new=check[2]+settings.EXPLOSION_TIMER
            #会根据前一步的金币和本步最近的金币取一个最小值
            nearest_coin_distance_new=min(nearest_coin_distance_new,target_coin_distance_new)
            if nearest_coin_distance>nearest_coin_distance_new:
                events.append("TOWARD_COIN")
            if nearest_coin_distance<nearest_coin_distance_new:
                events.append("AWAY_COIN")
            #如果发现金币，那么将吃金币的阈值设定为这个距离
            if nearest_coin_distance!=float('inf') and nearest_coin_distance<5 and EAT_COIN_STEP==0:
                EAT_COIN_STEP=nearest_coin_distance
        #如果经过的step超过阈值的2呗，则判定为没有很快速的去吃金币
        if EAT_COIN_STEP!=0:
            STEPS+=1
            if [new_position[0],new_position[1]] in coins_position:
                EAT_COIN_STEP=0
                STEPS=0
            else:
                if STEPS>=EAT_COIN_STEP*2:
                    EAT_COIN_STEP=0
                    STEPS=0
                    events.append("NOT_EAT_COIN_QUICK")

    
    #2. 放置炸弹的好坏判断
    #2.1 炸弹是否会将自己炸死
    if action=="BOMB":
        safe_position_new=[]
        none_zero=np.where(field_combined_new==0)
        for coord in zip(none_zero[0],none_zero[1]):
            safe_position_new.append([coord[0],coord[1]])
        nearest_safe_position_new,nearest_safe_distance_new=bfs(field_combined_new,old_position,safe_position_new,[WALL,CRATE,BOMB])
        safe_position=[]
        none_zero=np.where(field_combined==0)
        for coord in zip(none_zero[0],none_zero[1]):
            safe_position.append([coord[0],coord[1]])
        nearest_safe_position,nearest_safe_distance=bfs(field_combined,old_position,safe_position,[WALL,CRATE,BOMB])
        distance=max(nearest_safe_distance_new,nearest_safe_distance)
        if distance<=settings.BOMB_TIMER:
            events.append("SUICIDE_BOMB")
        else:
            events.append("NICE_TRY_BOMB")
    #2.2 炸弹炸掉墙的数目
    if action=="BOMB":
        destory_crate_num,try_kill_other=check_destory_object_num(field_combined,old_position,others_position)
        if destory_crate_num==0 and try_kill_other==0:
            events.append("NOT_USEFUL_BOMB")
        if 3<=destory_crate_num<6:
            events.append("DESTORY_CRATE_LEVEL1")
        if 6<=destory_crate_num<10:
            events.append("DESTORY_CRATE_LEVEL2")
        if try_kill_other>0:
            events.append("TRY_KILL_OTHER")

    #这一步没啥用，发现不会触发
    none_zero_crate=np.where(field_combined==CRATE)
    if len(others)==0 and len(coins_position)==0 and len(none_zero_crate)==0 and action=='BOMB':
        events.append("INVALID_BOMB")


    #2.3 炸弹是否会阻挡敌人获取金币的效率，暂未实现

    #2.4 炸弹放置后还活着
    if action=="BOMB":
        LIVE_STEP[0]=True
    if LIVE_STEP[0]:
        LIVE_STEP[1]-=1
    if LIVE_STEP[1]==-1 and action!=None:
        LIVE_STEP=[False,settings.BOMB_TIMER]
        events.append("SUCCESS_USE_BOMB")
    if LIVE_STEP[1]==-1:
        LIVE_STEP=[False,settings.BOMB_TIMER]

    
    #3. 自身位置的好坏
    #3.1 在炸弹区域内，是否远离炸弹
    check_self_position=check_object_in_explosion_bomb(old_position,explosition_map,bombs)
    if check_self_position[0]:
        nearest_bomb_position=[check_self_position[3],check_self_position[4]]
        distance_old=abs(old_position[0]-nearest_bomb_position[0])+abs(old_position[1]-nearest_bomb_position[1])
        distance_new=abs(new_position[0]-nearest_bomb_position[0])+abs(new_position[1]-nearest_bomb_position[1])
        if distance_old<distance_new:
            events.append("TRY_TO_SAVE_SELF")
        else:
            events.append("SUICIDE")
    if -20<field_combined[new_position[0]][new_position[1]]<0 or field_combined[new_position[0]][new_position[1]]==1:
        events.append("SUICIDE")
    #3.2 是否重复踏入一片区域内
    if len(others)!=0 or len(coins_position)!=0 or len(none_zero_crate)!=0:
    #场上如果没有别人，也没有墙，也没有金币
        TRAIL_MATRIX[new_position[0]][new_position[1]]+=1
        if TRAIL_MATRIX[new_position[0]][new_position[1]]%5==0:
            events.append("REPEAT_MOVE")
        #3.3 开拓新的区域
        if TRAIL_MATRIX[new_position[0]][new_position[1]]==1:
            events.append("NEW_EXPLORATION")
    #3.4 活了一步
    events.append("SURVIVED_STEP")
    
    #3.5 能量判断
    old_potential=potential_field(old_game_state['field'],old_game_state['self'][3],coins_position,bomb_position,old_game_state['explosion_map'])
    new_potential=potential_field(old_game_state['field'],new_game_state['self'][3],coins_position,bomb_position,old_game_state['explosion_map'])

    if len(others)!=0 or len(coins_position)!=0 or len(none_zero_crate)!=0:
        if old_potential<new_potential:
            events.append("MOVE_HIGH_POTENTIAL_FIELD")
        elif old_potential>new_potential:
            events.append('MOVE_LOW_POTENTIAL_FIELD')
    
    #如果本轮游戏结束，存入tag是1，重置所有全局变量
    if reset_tag==1:
        EAT_COIN_STEP=0
        TRAIL_MATRIX=np.zeros([settings.ROWS,settings.COLS])
        LIVE_STEP=[False,settings.BOMB_TIMER]
        STEPS=0
        COIN_POSITION=set()
        COIN_POSITION_RECORD=set()

    
def check_destory_object_num(field,bomb,others):
    H,W=field.shape
    def is_valid(x,y):
        return 0<=x<H and 0<=y<W
    cratenum,other_num=0,0
    left_power_list=[[0,i] for i in range(-1,-settings.BOMB_POWER-1,-1)]
    right_power_list=[[0,i] for i in range(1,settings.BOMB_POWER+1)]
    up_power_list=[[i,0] for i in range(-1,-settings.BOMB_POWER-1,-1)]
    down_power_list=[[i,0] for i in range(1,settings.BOMB_POWER+1)]
    for pos in left_power_list:
        if is_valid(bomb[0]+pos[0],bomb[1]+pos[1]):
            if field[bomb[0]+pos[0]][bomb[1]+pos[1]]==WALL:
                break
            if field[bomb[0]+pos[0]][bomb[1]+pos[1]]==CRATE:
                cratenum+=1
    for pos in right_power_list:
        if is_valid(bomb[0]+pos[0],bomb[1]+pos[1]):
            if field[bomb[0]+pos[0]][bomb[1]+pos[1]]==WALL:
                break
            if field[bomb[0]+pos[0]][bomb[1]+pos[1]]==CRATE:
                cratenum+=1
    for pos in up_power_list:
        if is_valid(bomb[0]+pos[0],bomb[1]+pos[1]):
            if field[bomb[0]+pos[0]][bomb[1]+pos[1]]==WALL:
                break
            if field[bomb[0]+pos[0]][bomb[1]+pos[1]]==CRATE:
                cratenum+=1
    for pos in down_power_list:
        if is_valid(bomb[0]+pos[0],bomb[1]+pos[1]):
            if field[bomb[0]+pos[0]][bomb[1]+pos[1]]==WALL:
                break
            if field[bomb[0]+pos[0]][bomb[1]+pos[1]]==CRATE:
                cratenum+=1
    for pos in left_power_list:
        if is_valid(bomb[0]+pos[0],bomb[1]+pos[1]):
            if len(others)!=0 and  [bomb[0]+pos[0],bomb[1]+pos[1]] in others:
                other_num+=1
    for pos in right_power_list:
        if is_valid(bomb[0]+pos[0],bomb[1]+pos[1]):
            if len(others)!=0 and  [bomb[0]+pos[0],bomb[1]+pos[1]] in others:
                other_num+=1
    for pos in up_power_list:
        if is_valid(bomb[0]+pos[0],bomb[1]+pos[1]):
            if len(others)!=0 and  [bomb[0]+pos[0],bomb[1]+pos[1]] in others:
                other_num+=1
    for pos in down_power_list:
        if is_valid(bomb[0]+pos[0],bomb[1]+pos[1]):
            if len(others)!=0 and  [bomb[0]+pos[0],bomb[1]+pos[1]] in others:
                other_num+=1
    return cratenum,other_num

def combine_field_expolistion_bomb(field_matrix,bombs,explosition_map):
    """
    field_matrix是最原始的矩阵
    bombs是[(x,y),time]
    explosition_map是与field_matrix是最原始的矩阵大小一致的矩阵
    #箱子修改为-40
    #墙壁修改为-80
    #爆炸持续时间为负的倒计时
    #爆炸等待时间为正的倒计时
    """
    H,W=field_matrix.shape
    def is_valid(x,y):
        return 0<=x<H and 0<=y<W
    field=copy.deepcopy(field_matrix)
    field[field==1]=CRATE
    field[field==-1]=WALL
    if len(bombs)!=0:
        left_power_list=[[0,i] for i in range(-1,-settings.BOMB_POWER-1,-1)]
        right_power_list=[[0,i] for i in range(1,settings.BOMB_POWER+1)]
        up_power_list=[[i,0] for i in range(-1,-settings.BOMB_POWER-1,-1)]
        down_power_list=[[i,0] for i in range(1,settings.BOMB_POWER+1)]
        #多个炸弹的倒计时取最小值
        for bomb in bombs:
            bomb_x,bomb_y,bomb_time=bomb[0][0],bomb[0][1],bomb[1]
            field[bomb_x][bomb_y]=bomb_time
            for [x,y] in left_power_list:
                if is_valid(bomb_x+x,bomb_y+y):
                    if field[bomb_x+x][bomb_y+y]==WALL:
                        break
                    if field[bomb_x+x][bomb_y+y]==CRATE:
                        continue
                    if  field[bomb_x+x][bomb_y+y]>0:
                        field[bomb_x+x][bomb_y+y]=min(field[bomb_x+x][bomb_y+y],bomb_time)
                    else:
                        field[bomb_x+x][bomb_y+y]=bomb_time
            for [x,y] in right_power_list:
                if is_valid(bomb_x+x,bomb_y+y):
                    if  field[bomb_x+x][bomb_y+y]==WALL:
                        break
                    if field[bomb_x+x][bomb_y+y]==CRATE:
                        continue
                    if field[bomb_x+x][bomb_y+y]>0:
                        field[bomb_x+x][bomb_y+y]=min(field[bomb_x+x][bomb_y+y],bomb_time)
                    else:
                        field[bomb_x+x][bomb_y+y]=bomb_time
            for [x,y] in up_power_list:
                if is_valid(bomb_x+x,bomb_y+y):
                    if field[bomb_x+x][bomb_y+y]==WALL:
                        break
                    if field[bomb_x+x][bomb_y+y]==CRATE:
                        continue
                    if field[bomb_x+x][bomb_y+y]>0:
                        field[bomb_x+x][bomb_y+y]=min(field[bomb_x+x][bomb_y+y],bomb_time)
                    else:
                        field[bomb_x+x][bomb_y+y]=bomb_time
            for [x,y] in down_power_list:
                if is_valid(bomb_x+x,bomb_y+y):
                    if field[bomb_x+x][bomb_y+y]==WALL:
                        break
                    if field[bomb_x+x][bomb_y+y]==CRATE:
                        continue
                    if field[bomb_x+x][bomb_y+y]>0:
                        field[bomb_x+x][bomb_y+y]=min(field[bomb_x+x][bomb_y+y],bomb_time)
                    else:
                        field[bomb_x+x][bomb_y+y]=bomb_time
            field[bomb_x][bomb_y]=bomb_time
    none_zero_value=np.where(explosition_map!=0)
    for coord in zip(none_zero_value[0],none_zero_value[1]):
        field[coord[0]][coord[1]]=-explosition_map[coord[0]][coord[1]]
    return field

def check_object_in_explosion_bomb(object_position,explosition_map,bombs):
    """
    检查该硬币是否在爆炸范围内，如果在则返回 (True,Time,0)
    检查该硬币是否在炸弹边,如果在则返回(True,0,Time)
    如果同时满足上述，则返回(True,Time,Time)
    如果都不满足，则返回(False,0,0)
    """
    return_value=[False,0,10000,-1,-1]
    if explosition_map[object_position[0]][object_position[1]]!=0:
        return_value[0]=True  
        return_value[1]=explosition_map[object_position[0]][object_position[1]]
    for bomb in bombs:
        if (object_position[0]==bomb[0][0] and object_position[1]-bomb[0][1]<=settings.BOMB_POWER) or (object_position[1]==bomb[0][1] and object_position[0]-bomb[0][0]<=settings.BOMB_POWER):
            return_value[0]=True  
            #多个炸弹的倒计时，需要取最小值作为安全值  
            return_value[2]=min(return_value[2],bomb[1])
            return_value[3]=bomb[0][0] 
            return_value[4]=bomb[0][1] 
    return return_value

def bfs(field,start_position, target_position,obstacle_type):
    """
    寻找与start_position最近的一个点的位置和距离
    field是经过组合后的矩阵
    start_postion为tuple类型
    target_position为[[],[],[]]
    obstacle_type为list类型,其中在的话，就需要考虑影响，比如bomb在的话，则需要考虑bomb倒计时结束前能否出去
    """
    H,W=field.shape
    def is_valid(x,y):
        return 0<=x<H and 0<=y<W
    
    if len(start_position)==0 or len(target_position)==0 or len(obstacle_type)==0 or start_position[0]<0 or start_position[0]>=H or  start_position[1]<0 or start_position[1]>=H:
        return (-1,-1),float('inf')
    
    queue = deque([(start_position, 0)])
    visited = set()
    while queue:
        (x, y), distance = queue.popleft()

        if [x,y] in target_position:
            return [x,y],distance

        visited.add((x, y))

        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy

            if is_valid(nx, ny) and field[nx][ny] not in obstacle_type  and (nx, ny) not in visited:
                if BOMB in obstacle_type:
                    if field[nx][ny]<0 and field[nx][ny]>distance+1:
                        queue.append(((nx, ny), distance + 1))
                        visited.add((nx, ny))
                if field[nx][ny]>0 and field[nx][ny]<distance+1:
                    queue.append(((nx, ny), distance + 1))
                    visited.add((nx, ny))
                if field[nx][ny]==0:
                    queue.append(((nx, ny), distance + 1))
                    visited.add((nx, ny))

    return [-1,-1],float('inf')

def potential_field(field_matrix,start_postion,coin_postion,bomb_postion,explostion_posision):
    def attractive_potential(x, y, goal):
        k_att = 1.0  # 吸引力系数
        return 0.5 * k_att * np.linalg.norm(np.array([x, y]) - np.array(goal)) ** 2

    def repulsive_potential(x, y, obstacle, obstacle_type):
        k_rep = 1.0  # 斥力系数
        min_distance = 1.0  # 避免除以零的最小距离
        distance = max(min_distance, np.linalg.norm(np.array([x, y]) - np.array(obstacle[0],obstacle[1])))
        if obstacle_type == 1:
            return 0.5 * k_rep * (1.0 / distance - 1.0 / min_distance) ** 2
        elif obstacle_type == 2:
            return 0.25 * k_rep * (1.0 / distance - 1.0 / min_distance) ** 2
        elif obstacle_type == 3:
            return 0.1 * k_rep * (1.0 / distance - 1.0 / min_distance) ** 2
        
    def compute_total_potential(x,y, goals, obstacles,obstacle_type):
        total_potential = 0.0
        for goal in goals:
            total_potential += attractive_potential(x, y, goal)
        for obstacle in obstacles:
            total_potential += repulsive_potential(x, y, obstacle, obstacle_type)
        return total_potential

    nonzero_coords = np.where(field_matrix!= 0)
    crate_postion=[]
    wall_postion=[]
    for coord in zip(nonzero_coords[0], nonzero_coords[1]):
        if field_matrix[coord[0]][coord[1]]==-1:
            wall_postion.append([coord[0],coord[0]])
        if field_matrix[coord[0]][coord[1]]==1:
            crate_postion.append([coord[0],coord[0]])
    crate_postion=np.array(crate_postion)
    wall_postion=np.array(wall_postion)

    explosition_pos=[]
    nonzero_coords = np.where(explostion_posision!= 0)
    for coord in zip(nonzero_coords[0], nonzero_coords[1]):
        explosition_pos.append([coord[0],coord[1]])

    potential=0
    potential+=compute_total_potential(start_postion[0],start_postion[1],coin_postion,[],0)
    potential+=compute_total_potential(start_postion[0],start_postion[1],[],bomb_postion,1)
    potential+=compute_total_potential(start_postion[0],start_postion[1],[],explosition_pos,1)
    potential+=compute_total_potential(start_postion[0],start_postion[1],[],wall_postion,2)
    potential+=compute_total_potential(start_postion[0],start_postion[1],[],crate_postion,3)

    return potential