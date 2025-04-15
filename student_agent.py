import math
import struct
import numpy as np
import random
import gym
from gym import spaces
import matplotlib.pyplot as plt
import copy
from collections import defaultdict
import gc
import helper

COLOR_MAP = {
    0: "#cdc1b4", 2: "#eee4da", 4: "#ede0c8", 8: "#f2b179",
    16: "#f59563", 32: "#f67c5f", 64: "#f65e3b", 128: "#edcf72",
    256: "#edcc61", 512: "#edc850", 1024: "#edc53f", 2048: "#edc22e",
    4096: "#3c3a32", 8192: "#3c3a32", 16384: "#3c3a32", 32768: "#3c3a32"
}
TEXT_COLOR = {
    2: "#776e65", 4: "#776e65", 8: "#f9f6f2", 16: "#f9f6f2",
    32: "#f9f6f2", 64: "#f9f6f2", 128: "#f9f6f2", 256: "#f9f6f2",
    512: "#f9f6f2", 1024: "#f9f6f2", 2048: "#f9f6f2", 4096: "#f9f6f2"
}

class Game2048Env(gym.Env):
    def __init__(self):
        super(Game2048Env, self).__init__()

        self.size = 4
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0

        # Action space: 0: up, 1: down, 2: left, 3: right
        self.action_space = spaces.Discrete(4)
        self.actions = ["up", "right", "down", "left"]

        self.last_move_valid = True

        self.reset()

    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.add_random_tile()
        self.add_random_tile()
        return self.board

    def add_random_tile(self):
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            x, y = random.choice(empty_cells)
            self.board[x, y] = 2 if random.random() < 0.9 else 4

    def compress(self, row):
        new_row = row[row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        return new_row

    def merge(self, row):
        for i in range(len(row) - 1):
            if row[i] == row[i + 1] and row[i] != 0:
                row[i] *= 2
                row[i + 1] = 0
                self.score += row[i]
        return row

    def move_left(self):
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            new_row = self.compress(self.board[i])
            new_row = self.merge(new_row)
            new_row = self.compress(new_row)
            self.board[i] = new_row
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_right(self):
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            reversed_row = self.board[i][::-1]
            reversed_row = self.compress(reversed_row)
            reversed_row = self.merge(reversed_row)
            reversed_row = self.compress(reversed_row)
            self.board[i] = reversed_row[::-1]
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_up(self):
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            col = self.compress(self.board[:, j])
            col = self.merge(col)
            col = self.compress(col)
            self.board[:, j] = col
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def move_down(self):
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            reversed_col = self.board[:, j][::-1]
            reversed_col = self.compress(reversed_col)
            reversed_col = self.merge(reversed_col)
            reversed_col = self.compress(reversed_col)
            self.board[:, j] = reversed_col[::-1]
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def is_game_over(self):
        if np.any(self.board == 0):
            return False
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.board[i, j] == self.board[i, j+1]:
                    return False
        for j in range(self.size):
            for i in range(self.size - 1):
                if self.board[i, j] == self.board[i+1, j]:
                    return False

        return True

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action"

        if action == 0:
            moved = self.move_up()
        elif action == 1:
            moved = self.move_down()
        elif action == 2:
            moved = self.move_left()
        elif action == 3:
            moved = self.move_right()
        else:
            moved = False

        self.last_move_valid = moved

        if moved:
            self.add_random_tile()

        done = self.is_game_over()

        return self.board, self.score, done, {}

    def render(self, mode="human", action=None):
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, self.size - 0.5)
        ax.set_ylim(-0.5, self.size - 0.5)

        for i in range(self.size):
            for j in range(self.size):
                value = self.board[i, j]
                color = COLOR_MAP.get(value, "#3c3a32")
                text_color = TEXT_COLOR.get(value, "white")
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=color, edgecolor="black")
                ax.add_patch(rect)

                if value != 0:
                    ax.text(j, i, str(value), ha='center', va='center',
                            fontsize=16, fontweight='bold', color=text_color)
        title = f"score: {self.score}"
        if action is not None:
            title += f" | action: {self.actions[action]}"
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.show()

def tile_to_number(tile):
    return 0 if tile == 0 else int(math.log(tile, 2))

def convert_to_flat(env_board, i=0, j=0, flat=None):
    if flat is None:
        flat = []
    rows, cols = env_board.shape
    if i >= rows:
        return flat
    if j >= cols:
        return convert_to_flat(env_board, i + 1, 0, flat)
    flat.append(tile_to_number(env_board[i, j]))
    return convert_to_flat(env_board, i, j + 1, flat)

def merge_with_reward_recursive(row, index, reward):
    if index >= len(row) - 1:
        return row, reward
    if row[index] != 0 and row[index] == row[index + 1]:
        row[index] *= 2
        reward += row[index]
        row[index + 1] = 0
    return merge_with_reward_recursive(row, index + 1, reward)

def merge_with_reward(row):
    return merge_with_reward_recursive(row.copy(), 0, 0)

def compress_and_merge(row):
    comp = compress(row)
    merged, reward = merge_with_reward(comp)
    final = compress(merged)
    return final, reward

def compress(row):
    new_row = row[row != 0]
    new_row = np.pad(new_row, (0, 4 - len(new_row)), mode='constant')
    return new_row

def afterstate_move_left(board, i=0, total_reward=0):
    if i >= 4:
        return board, total_reward
    row = board[i, :].copy()
    new_row, reward = compress_and_merge(row)
    board[i, :] = new_row
    return afterstate_move_left(board, i + 1, total_reward + reward)

def afterstate_move_right(board, i=0, total_reward=0):
    if i >= 4:
        return board, total_reward
    row = board[i, ::-1].copy()
    new_row, reward = compress_and_merge(row)
    board[i, :] = new_row[::-1]
    return afterstate_move_right(board, i + 1, total_reward + reward)

def afterstate_move_up(board, j=0, total_reward=0):
    if j >= 4:
        return board, total_reward
    col = board[:, j].copy()
    comp_col = compress(col)
    merged, reward = merge_with_reward(comp_col)
    final_col = compress(merged)
    board[:, j] = final_col
    return afterstate_move_up(board, j + 1, total_reward + reward)

def afterstate_move_down(board, j=0, total_reward=0):
    if j >= 4:
        return board, total_reward
    col = board[::-1, j].copy()
    comp_col = compress(col)
    merged, reward = merge_with_reward(comp_col)
    final_col = compress(merged)
    board[:, j] = final_col[::-1]
    return afterstate_move_down(board, j + 1, total_reward + reward)

def get_afterstate_and_reward(board, action):
    if action == 0:
        return afterstate_move_up(board)
    elif action == 1:
        return afterstate_move_down(board)
    elif action == 2:
        return afterstate_move_left(board)
    elif action == 3:
        return afterstate_move_right(board)

def is_move_legal(board, action):
    after_board, _ = get_afterstate_and_reward(board, action)
    return not np.array_equal(board, after_board)

def add_random_tile(board):
    empty = list(zip(*np.where(board == 0)))
    if not empty:
        return board.copy()
    board_new = board.copy()
    i, j = random.choice(empty)
    board_new[i, j] = 2 if random.random() < 0.9 else 4
    return board_new

def best_move_selection(env_board, approximator):
    # Recursively examine actions 0..3 to choose the best.
    def rec_best(action, best_pair):
        if action >= 4:
            return best_pair
        after, _ = get_afterstate_and_reward(env_board, action)
        if np.array_equal(env_board, after):
            return rec_best(action + 1, best_pair)
        flat_after = convert_to_flat(after)
        current_val = approximator.value(flat_after)
        if current_val > best_pair[1]:
            best_pair = (action, current_val)
        return rec_best(action + 1, best_pair)
    return rec_best(0, (None, -float('inf')))

class NTupleApproximator:
    def __init__(self, board_size, patterns, iso=8):
        self.board_size = board_size
        self.features = []
        for pat in patterns:
            self.features.append(helper.PatternFeature(pat, iso))

    def _recursive_value(self, feats, flat_board):
        if not feats:
            return 0
        return feats[0].estimate(flat_board) + self._recursive_value(feats[1:], flat_board)

    def value(self, flat_board):
        return self._recursive_value(self.features, flat_board)

    def _recursive_update(self, feats, flat_board, delta, alpha):
        if not feats:
            return
        feats[0].update(flat_board, alpha * delta)
        self._recursive_update(feats[1:], flat_board, delta, alpha)

    def update(self, flat_board, delta, alpha):
        self._recursive_update(self.features, flat_board, delta, alpha)

    def load(self, file_path, size_t_fmt='Q'):
        size_t_size = struct.calcsize(size_t_fmt)
        with open(file_path, 'rb') as f:
            count_bytes = f.read(size_t_size)
            struct.unpack(size_t_fmt, count_bytes)[0]
            for feat in self.features:
                feat.load_from_stream(f, size_t_fmt)

# ----------------------------------------------------------------------
# MCTS node classes using recursive versions for internal loops

class NodeForState:
    def __init__(self, board, parent=None):
        self.node_type = 0  # 0 for state node, 1 for afterstate node
        self.board = board.copy()
        self.parent = parent
        self.children = {}  # action -> NodeForAfterstate
        self.visits = 0
        self.value = 0.0

    def is_fully_expanded(self, legal_actions):
        # Recursive check for each legal action in the children.
        def rec_check(actions):
            if not actions:
                return True
            return (actions[0] in self.children) and rec_check(actions[1:])
        return rec_check(legal_actions)

    def select_child(self, legal_actions, exploration_weight=1.0):
        def rec_select(actions, best_act, best_score):
            if not actions:
                return best_act
            a = actions[0]
            if a not in self.children:
                return a  # if not expanded, return immediately
            child = self.children[a]
            # Use UCT formula
            score = child.value + exploration_weight * math.sqrt(math.log(self.visits) / child.visits)
            if score > best_score:
                best_score, best_act = score, a
            return rec_select(actions[1:], best_act, best_score)
        return rec_select(legal_actions, None, -float('inf'))

class NodeForAfterstate:
    def __init__(self, board, parent=None, reward=0):
        self.node_type = 1
        self.board = board.copy()
        self.parent = parent
        self.children = {}  # (row, col, value) -> NodeForState
        self.visits = 0
        self.value = 0.0
        self.reward = reward

    def is_fully_expanded(self, empty_locations):
        return len(self.children) == len(empty_locations) * 2

# ----------------------------------------------------------------------
# TD-MCTS using recursion instead of loops in its core methods

class td_mcts:
    def __init__(self, approximator, num_iterations=1000, exploration_weight=1.0, scaling=4096):
        self.approximator = approximator
        self.num_iterations = num_iterations
        self.exploration_weight = exploration_weight
        self.scaling = scaling

    def tmp_env(self, board):
        env = Game2048Env()
        env.board = board.copy()
        return env

    def recursive_select(self, current_node, available_moves, path_history):
        if current_node.node_type != 0:
            return current_node, path_history
        if available_moves is None:
            temp_game = self.tmp_env(current_node.board)
            available_moves = [move for move in range(4) if temp_game.is_move_legal(move)]
        if not current_node.is_fully_expanded(available_moves):
            chosen_move = current_node.select_child(available_moves, self.exploration_weight)
            if chosen_move not in current_node.children:
                return current_node, path_history
            path_history.append((current_node, chosen_move))
            return self.recursive_select(current_node.children[chosen_move], None, path_history)
        else:
            chosen_move = current_node.select_child(available_moves, self.exploration_weight)
            path_history.append((current_node, chosen_move))
            return self.recursive_select(current_node.children[chosen_move], None, path_history)

    def select(self, current_node, env, available_moves=None):
        return self.recursive_select(current_node, available_moves, [])

    def expand(self, node, env, previous_action=None):
        if node.node_type == 0:
            legal_actions = [a for a in range(4) if is_move_legal(node.board, a)]
            def rec_expand(actions):
                if not actions:
                    return None
                a = actions[0]
                if a not in node.children:
                    afterstate, reward = get_afterstate_and_reward(node.board, a)
                    new_node = NodeForAfterstate(afterstate, node, reward)
                    node.children[a] = new_node
                    return new_node
                return rec_expand(actions[1:])
            result = rec_expand(legal_actions)
            if result is not None:
                return result
        elif node.node_type == 1:
            empty_locations = list(zip(*np.where(node.board == 0)))
            def rec_expand_after(empties):
                if not empties:
                    return None
                row, col = empties[0]
                key1, key2 = (row, col, 2), (row, col, 4)
                if key1 not in node.children:
                    new_board = node.board.copy()
                    new_board[row, col] = 2
                    state_node = NodeForState(new_board, node)
                    node.children[key1] = state_node
                    return state_node
                elif key2 not in node.children:
                    new_board = node.board.copy()
                    new_board[row, col] = 4
                    state_node = NodeForState(new_board, node)
                    node.children[key2] = state_node
                    return state_node
                return rec_expand_after(empties[1:])
            result = rec_expand_after(empty_locations)
            if result is not None:
                return result
        return node

    def simulate(self, node):
        board = node.board
        afterstate_value = self.approximator.value(convert_to_flat(board))
        if node.node_type == 1:
            return (node.reward + afterstate_value) / self.scaling
        else:
            legal_actions = [a for a in range(4) if is_move_legal(board, a)]
            if not legal_actions:
                return 0
            def rec_simulate(actions, best_val=-float('inf')):
                if not actions:
                    return best_val
                a = actions[0]
                new_board, reward = get_afterstate_and_reward(board, a)
                value = self.approximator.value(convert_to_flat(new_board))
                action_value = (reward + value) / self.scaling
                best_val = max(best_val, action_value)
                return rec_simulate(actions[1:], best_val)
            return rec_simulate(legal_actions)

    def backpropagate(self, node, value, trajectory):
        node.visits += 1
        diff = (value - node.value) / node.visits
        node.value += diff
        rev_traj = list(reversed(trajectory))
        def rec_backprop(traj):
            if not traj:
                return
            parent, action = traj[0]
            child = parent.children[action]
            parent.visits += 1
            d = (child.value - parent.value) / parent.visits
            parent.value += d
            rec_backprop(traj[1:])
        rec_backprop(rev_traj)

    def best_action(self, root, legal_actions):
        def rec_best(actions, best_act=None, best_val=-float('inf')):
            if not actions:
                return best_act
            a = actions[0]
            if a not in root.children:
                val = 0
            else:
                val = root.children[a].visits
            if val > best_val:
                best_act, best_val = a, val
            return rec_best(actions[1:], best_act, best_val)
        return rec_best(legal_actions)

# ----------------------------------------------------------------------
# Model initialization and action selection

patterns = [
    [0, 1, 2, 3, 4, 5],
    [4, 5, 6, 7, 8, 9],
    [0, 1, 2, 4, 5, 6],
    [4, 5, 6, 8, 9, 10]
]
approximator = NTupleApproximator(board_size=4, patterns=patterns)
approximator.load("./my2048-1.bin", size_t_fmt='Q')

mcts_agent = td_mcts(approximator, num_iterations=100, exploration_weight=1.0, scaling=4096)

def init_model():
    global approximator
    global mcts_agent
    if approximator is None:
        gc.collect()
        approximator = NTupleApproximator(board_size=4, patterns=patterns)
        approximator.load("./my2048-1.bin", size_t_fmt='Q')
        mcts_agent = td_mcts(approximator, num_iterations=100, exploration_weight=1.0, scaling=4096)

def iterate_mcts(n, root, score, legal_actions):
    if n <= 0:
        return
    env = Game2048Env()
    env.board = root.board.copy()
    env.score = score
    leaf, path = mcts_agent.select(root, env, legal_actions)
    if leaf.visits > 0:
        leaf = mcts_agent.expand(leaf, env, path[-1][0] if path else None)
    val = mcts_agent.simulate(leaf)
    mcts_agent.backpropagate(leaf, val, path)
    iterate_mcts(n - 1, root, score, legal_actions)

def get_action(state, score):
    init_model()
    root = NodeForState(state)
    temp_env = mcts_agent.tmp_env(state)
    legal_actions = [a for a in range(4) if temp_env.is_move_legal(a)]
    if not legal_actions:
        action = 0
    else:
        iterate_mcts(mcts_agent.num_iterations, root, score, legal_actions)
        action = mcts_agent.best_action(root, legal_actions)
    return action