
import numpy as np
import events as e

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def vaild_action_filter(game_state):
    """
    filter vaild action
    :game_state The dictionary that describes everything on the board.
    :valid_actions valid action list according to game_state
    """
    x = game_state['self'][3][0]
    y = game_state['self'][3][1]
    field = np.array(game_state["field"]).copy()

    for bomb in game_state["bombs"]:
        (_x, _y), timer = bomb
        field[_x, _y] = 2  # 炸弹
    for other in game_state["others"]:
        name, score, bombs_left, (_x, _y) = other
        field[_x, _y] = 3  # 其他人

    valid_actions = []
    if field[x, y] == 0:
        valid_actions.append(4)

    # 能添加炸弹
    if game_state['self'][2]:
        valid_actions.append(5)

    # 不能撞墙
    if x >= 1 and field[x - 1, y] == 0:
        valid_actions.append(3)
    if y <= 15 and field[x, y + 1] == 0:
        valid_actions.append(2)
    if x <= 15 and field[x + 1, y] == 0:
        valid_actions.append(1)
    if y >= 1 and field[x, y - 1] == 0:
        valid_actions.append(0)

    if len(valid_actions) == 0:
        valid_actions.append(4)

    return valid_actions


def DFS(game_state, step):
    my_pos = game_state['self'][-1]
    field = game_state['field']
    node = my_pos
    point_canbe_reached = []

    def dfs(node, point_canbe_reached, left_step):
        if left_step == 0:
            point_canbe_reached.append(node)
            return
        if node in point_canbe_reached:
            return
        point_canbe_reached.append(node)
        next_node = []

        x = node[0]
        y = node[1]
        if x - 1 >= 0 and field[x - 1][y] == 0 and (x - 1, y) not in point_canbe_reached:
            next_node.append((x - 1, y))
        if y + 1 <= 16 and field[x][y + 1] == 0 and (x, y + 1) not in point_canbe_reached:
            next_node.append((x, y + 1))
        if x + 1 <= 16 and field[x + 1][y] == 0 and (x + 1, y) not in point_canbe_reached:
            next_node.append((x + 1, y))
        if y - 1 >= 0 and field[x][y - 1] == 0 and (x, y - 1) not in point_canbe_reached:
            next_node.append((x, y - 1))

        for node in next_node:
            if not node in point_canbe_reached:
                dfs(node, point_canbe_reached, left_step - 1)

    dfs(node, point_canbe_reached, step)
    return point_canbe_reached


# [eat coins, kill opponent, bomb crates, not kill self]
def add_high_events(self, old_game_state, new_game_state, events, mission, all_old_bomb_state):
    target_mission = None
    # if will be exploded -> not kill self
    my_pos = old_game_state['self'][-1]
    if all_old_bomb_state:
        for bomb in all_old_bomb_state:
            if bomb[-1] == 0:
                old_game_state['bombs'].append(bomb)

    for bomb in old_game_state['bombs']:
        bomb_pos = bomb[0]
        if (bomb_pos[0] == my_pos[0] and abs(bomb_pos[1] - my_pos[1]) <= 3) or (
                bomb_pos[1] == my_pos[1] and abs(bomb_pos[0] - my_pos[0]) <= 3):
            target_mission = 3
            break

    # if a opponent nearby -> kill opponent
    if not target_mission:
        # find nodes that can be reached in n step
        pos = DFS(old_game_state, step=5)
        for opp in old_game_state['others']:
            if opp[-1] in pos:
                target_mission = 1
                break

    # if a coin in 10 step -> eat coins
    if not target_mission:
        # find nodes that can be reached in n step
        pos = DFS(old_game_state, step=10)
        for coin in old_game_state['coins']:
            if coin in pos:
                target_mission = 0
                break
    # else bomb crates
    if not target_mission:
        target_mission = 2


    if target_mission == mission.item():
        events.append("MATCH")
    else:
        events.append("UNMATCH")

    return events


def do_escape(self, old_game_state, new_game_state, events, action):
    old_self_x, old_self_y = old_game_state['self'][-1][0], old_game_state['self'][-1][1]
    new_self_x, new_self_y = new_game_state['self'][-1][0], new_game_state['self'][-1][1]
    minum_bomb = None
    minum_dis = 100000
    if old_game_state['bombs'] and old_game_state['bombs'][-1] != 0:
        for bomb in old_game_state['bombs']:
            if minum_dis > abs(old_self_x - bomb[0][0]) + abs(old_self_y - bomb[0][1]):
                minum_dis = abs(old_self_x - bomb[0][0]) + abs(old_self_y - bomb[0][1])
                minum_bomb = bomb[0]
        old_state_in_line_small_dis = (old_self_x == minum_bomb[0] and abs(old_self_y - minum_bomb[1]) < 4) \
                                      or (old_self_y == minum_bomb[1] and abs(old_self_x - minum_bomb[0]))
        new_state_not_in_line_or_big_dis = (new_self_x != minum_bomb[0] and new_self_y != minum_bomb[1]) \
                                           or ((new_self_x == minum_bomb[0] and abs(new_self_y - minum_bomb[1]) >= 4)
                                               or (new_self_y == minum_bomb[1] and abs(
                    new_self_x - minum_bomb[0]) >= 4))
        old_state_not_in_line_or_big_dis = (old_self_x != minum_bomb[0] and old_self_y != minum_bomb[1]) \
                                           or ((old_self_x == minum_bomb[0] and abs(old_self_y - minum_bomb[1]) >= 4)
                                               or (old_self_y == minum_bomb[1] and abs(
                    old_self_x - minum_bomb[0]) >= 4))
        new_state_in_line_small_dis = (new_self_x == minum_bomb[0] and abs(new_self_y - minum_bomb[1]) < 4) \
                                      or (new_self_y == minum_bomb[1] and abs(new_self_x - minum_bomb[0]))

        # old state in line and small distance -> new state not in line or big distance
        if old_state_in_line_small_dis and new_state_not_in_line_or_big_dis:
            events.append("ESCAPE")  # good action
        if old_state_not_in_line_or_big_dis and new_state_not_in_line_or_big_dis:
            events.append("ESCAPE")  # good action
        # old state not in line or big distance -> new state in line and small distance
        if old_state_not_in_line_or_big_dis and new_state_in_line_small_dis:
            # print("SUICIDE")
            events.append("SUICIDE")  # bad action
        if old_state_in_line_small_dis and new_state_in_line_small_dis:
            events.append("SUICIDE")  # bad action
    return events


# e.FINSIH_TASK: 1,
# e.CLOSER_TO_FINSIH_TASK: 0.5,
# e.FAR_FROM_TAKS: -0.5
def find_nearest_target(game_state):
    my_pos = game_state['self'][-1]
    bombs = game_state['bombs']
    others = game_state['others']
    coins = game_state['coins']
    min_bomb_dis = 17 * 17 * 17
    min_opp_dis = 17 * 17 * 17
    min_coin_dis = 17 * 17 * 17

    for bomb in bombs:
        dis_bomb = abs(my_pos[0] - bomb[0][0]) + abs(my_pos[1] - bomb[0][1])
        if dis_bomb < min_bomb_dis:
            min_bomb_dis = dis_bomb
    for opp in others:
        dis_opp = abs(my_pos[0] - opp[-1][0]) + abs(my_pos[1] - opp[-1][1])
        if dis_opp < min_bomb_dis:
            min_bomb_dis = dis_opp
    for coin in coins:
        dis_coin = abs(my_pos[0] - coin[0]) + abs(my_pos[1] - coin[1])
        if dis_coin < min_coin_dis:
            min_coin_dis = dis_coin
    return min_bomb_dis, min_opp_dis, min_coin_dis

def drop_bomb(self, old_game_state, new_game_state, events, action):
    field = old_game_state['field']
    if action == "BOMB":
        bomb_pos = old_game_state['self'][-1]
        x_min = max(0, bomb_pos[0] - 3)
        x_max = min(16, bomb_pos[0] + 3)
        y_min = max(0, bomb_pos[1] - 3)
        y_max = min(16, bomb_pos[1] + 3)
        crate = 0
        for i in range(x_min, x_max + 1):
            if field[i][bomb_pos[1]] == 1:
                crate += 1
        for j in range(y_min, y_max + 1):
            if field[bomb_pos[0]][j] == 1:
                crate += 1
        if crate >= 5:
            events.append("BOMB_DROPPED_GOOD")
        elif crate <= 2:
            events.append("BOMB_DROPPED_BAD")
    return events

# [eat coins, kill opponent, bomb crates, not kill self]
def add_low_events(self, old_game_state, new_game_state, events, action, temp_mission, all_old_bomb_state):
    events = do_escape(self, old_game_state, new_game_state, events, action)
    events = drop_bomb(self, old_game_state, new_game_state, events, action)
    if find_nearest_target(old_game_state)[2] > find_nearest_target(new_game_state)[2]:
        events.append("CLOSER_TO_COIN")
    elif find_nearest_target(old_game_state)[2] < find_nearest_target(new_game_state)[2]:
        events.append('FURTHER_FROM_COIN')
    return events
