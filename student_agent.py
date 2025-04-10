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

    def simulate_row_move(self, row):
        new_row = row[row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        for i in range(len(new_row) - 1):
            if new_row[i] == new_row[i + 1] and new_row[i] != 0:
                new_row[i] *= 2
                new_row[i + 1] = 0
        new_row = new_row[new_row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        return new_row

    def is_move_legal(self, action):
        temp_board = self.board.copy()

        if action == 0:  # Move up
            for j in range(self.size):
                col = temp_board[:, j]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col
        elif action == 1:  # Move down
            for j in range(self.size):
                col = temp_board[:, j][::-1]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col[::-1]
        elif action == 2:  # Move left
            for i in range(self.size):
                row = temp_board[i]
                temp_board[i] = self.simulate_row_move(row)
        elif action == 3:  # Move right
            for i in range(self.size):
                row = temp_board[i][::-1]
                new_row = self.simulate_row_move(row)
                temp_board[i] = new_row[::-1]
        else:
            raise ValueError("Invalid action")
        return not np.array_equal(self.board, temp_board)

def mirror_flat(flat_board):
    square = [flat_board[(i * 4):((i + 1) * 4)] for i in range(4)]
    mirrored_square = [row[::-1] for row in square]
    return [idx for row in mirrored_square for idx in row]

def rotate_clockwise_flat(flat_board):
    square = [flat_board[(i * 4):((i + 1) * 4)] for i in range(4)]
    rotated_square = [[square[3 - j][i] for j in range(4)] for i in range(4)]
    return [idx for row in rotated_square for idx in row]

def rotate_flat(flat_board, cnt):
    ret = flat_board[:]
    for _ in range(cnt):
        ret = rotate_clockwise_flat(ret)
    return ret

class PatternFeature:
    def __init__(self, base_pattern, iso=8):
        self.base_pattern = base_pattern
        self.iso = iso
        self.table_size = 1 << (len(base_pattern) * 4)
        self.weights = [0.0] * self.table_size  # initialize weights to zero
        self.isom = self.generate_symmetries()

    def generate_symmetries(self):
        const_board = list(range(16))
        iso_patterns = []
        for i in range(self.iso):
            temp = const_board[:]
            if i >= 4:
                temp = mirror_flat(temp)
            temp = rotate_flat(temp, i % 4)
            transformed = [temp[t] for t in self.base_pattern]
            iso_patterns.append(transformed)
        return iso_patterns

    def feature_index(self, flat_board):
        indices = []
        for pattern in self.isom:
            idx = 0
            for i, t in enumerate(pattern):
                idx |= flat_board[t] << (4 * i)
            indices.append(idx)
        return indices

    def estimate(self, flat_board):
        indices = self.feature_index(flat_board)
        total = 0.0
        for idx in indices:
            total += self.weights[idx]
        return total

    def update(self, flat_board, adjust):
        indices = self.feature_index(flat_board)
        delta = adjust / len(indices)
        for idx in indices:
            self.weights[idx] += delta
        ret = 0.0
        for idx in indices:
            ret += self.weights[idx]
        return ret

    def name(self):
        base_hex = "".join(f"{x:x}" for x in self.base_pattern)
        return f"{len(self.base_pattern)}-tuple pattern {base_hex}"

    def load_from_stream(self, stream, size_t_fmt='I'):
        name_length_bytes = stream.read(4)
        name_length = struct.unpack('i', name_length_bytes)[0]
        name_bytes = stream.read(name_length)
        name_str = name_bytes.decode('utf-8')
        size_t_size = struct.calcsize(size_t_fmt)
        weight_count_bytes = stream.read(size_t_size)
        weight_count = struct.unpack(size_t_fmt, weight_count_bytes)[0]
        float_size = struct.calcsize('f')
        data_bytes = stream.read(float_size * weight_count)
        self.weights = list(struct.unpack(f'{weight_count}f', data_bytes))

class NTupleApproximator:
    def __init__(self, board_size, patterns, iso=8):
        self.board_size = board_size
        self.features = []
        for pattern in patterns:
            self.features.append(PatternFeature(pattern, iso))

    def value(self, flat_board):
        return sum(feat.estimate(flat_board) for feat in self.features)

    def update(self, flat_board, delta, alpha):
        for feat in self.features:
            feat.update(flat_board, alpha * delta)

    def load(self, file_path, size_t_fmt='Q'):
        size_t_size = struct.calcsize(size_t_fmt)
        with open(file_path, 'rb') as f:
            count_bytes = f.read(size_t_size)
            struct.unpack(size_t_fmt, count_bytes)[0]
            for feat in self.features:
                feat.load_from_stream(f, size_t_fmt)

def tile_to_number(tile):
    if tile == 0:
        return 0
    else:
        return int(math.log(tile, 2))

def convert_to_flat(env_board):
    flat = []
    for i in range(env_board.shape[0]):
        for j in range(env_board.shape[1]):
            tile = env_board[i, j]
            flat.append(tile_to_number(tile))
    return flat

def compress(row):
    new_row = row[row != 0]
    new_row = np.pad(new_row, (0, 4 - len(new_row)), mode='constant')
    return new_row

def merge_with_reward(row):
    reward = 0
    row = row.copy()
    for i in range(3):
        if row[i] != 0 and row[i] == row[i + 1]:
            row[i] *= 2
            reward += row[i]
            row[i + 1] = 0
    return row, reward

def compress_and_merge(row):
    compressed = compress(row)
    merged, reward = merge_with_reward(compressed)
    final = compress(merged)
    return final, reward

def afterstate_move_left(board):
    new_board = board.copy()
    total_reward = 0
    for i in range(4):
        row = new_board[i, :].copy()
        new_row, reward = compress_and_merge(row)
        total_reward += reward
        new_board[i, :] = new_row
    return new_board, total_reward

def afterstate_move_right(board):
    new_board = board.copy()
    total_reward = 0
    for i in range(4):
        row = new_board[i, ::-1].copy()
        new_row, reward = compress_and_merge(row)
        total_reward += reward
        new_board[i, :] = new_row[::-1]
    return new_board, total_reward

def afterstate_move_up(board):
    new_board = board.copy()
    total_reward = 0
    for j in range(4):
        col = new_board[:, j].copy()
        col = compress(col)
        merged, reward = merge_with_reward(col)
        final_col = compress(merged)
        total_reward += reward
        new_board[:, j] = final_col
    return new_board, total_reward

def afterstate_move_down(board):
    new_board = board.copy()
    total_reward = 0
    for j in range(4):
        col = new_board[::-1, j].copy()
        col = compress(col)
        merged, reward = merge_with_reward(col)
        final_col = compress(merged)
        total_reward += reward
        new_board[:, j] = final_col[::-1]
    return new_board, total_reward

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
    best_action = None
    best_value = -float('inf')
    current = env_board
    for action in range(4):
        after, _ = get_afterstate_and_reward(current, action)
        if np.array_equal(current, after):
            continue
        flat_after = convert_to_flat(after)
        val = approximator.value(flat_after)
        if val > best_value:
            best_value = val
            best_action = action
    return best_action, best_value


class NodeForState:
    def __init__(self, board, parent=None):
        self.node_type = 0 # 0 for state node, 1 for afterstate node
        self.board = board.copy()
        self.parent = parent
        self.children = {}  # action -> NodeForAfterstate
        self.visits = 0
        self.value = 0.0

    def is_fully_expanded(self, legal_actions):
        return all(action in self.children for action in legal_actions)

    def select_child(self, legal_actions, exploration_weight=1.0):
        best_score = -float('inf')
        best_action = None
        for action in legal_actions:
            if action not in self.children:
                return action
            else:
              child = self.children[action]
              score = child.value + exploration_weight * math.sqrt(math.log(self.visits) / child.visits)
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

class NodeForAfterstate:
    def __init__(self, board, parent=None, reward=0):
        self.node_type = 1
        self.board = board.copy()
        self.parent = parent
        self.children = {}  # (row, col, value) -> NodeForState
        self.visits = 0
        self.value = 0.0
        self.reward = reward

    def is_fully_expanded(self, empty_location):
        return len(self.children) == len(empty_location) * 2

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
    def select(self, current_node, env, available_moves=None):
        path_history = []

        # Traverse through state nodes.
        while current_node.node_type == 0:
            if available_moves is None:
                temp_game = self.tmp_env(current_node.board)
                available_moves = [move for move in range(4) if temp_game.is_move_legal(move)]
            if not current_node.is_fully_expanded(available_moves):
                chosen_move = current_node.select_child(available_moves, self.exploration_weight)
                if chosen_move not in current_node.children:
                    return current_node, path_history
                path_history.append((current_node, chosen_move))
                current_node = current_node.children[chosen_move]
            else:
                chosen_move = current_node.select_child(available_moves, self.exploration_weight)
                path_history.append((current_node, chosen_move))
                current_node = current_node.children[chosen_move]
                available_moves = None  # Reset for the next state node

        return current_node, path_history


    def expand(self, node, env, previous_action=None):
        if node.node_type == 0:
            legal_actions = [a for a in range(4) if is_move_legal(node.board, a)]
            for action in legal_actions:
                if action not in node.children:
                    afterstate, reward = get_afterstate_and_reward(node.board, action)
                    afterstate_node = NodeForAfterstate(afterstate, node, reward)
                    node.children[action] = afterstate_node
                    return afterstate_node

                elif node.node_type == 1:
                    empty_location = list(zip(*np.where(node.board == 0)))
            for row, col in empty_location:
                key1 = (row, col, 2)
                key2 = (row, col, 4)
                if key1 not in node.children:
                    new_board = node.board.copy()
                    new_board[row, col] = 2
                    state_node = NodeForState(new_board, node)
                    self.children[(row, col, 2)] = state_node
                    return state_node
                elif key2 not in node.children:
                    new_board = node.board.copy()
                    new_board[row, col] = 4
                    state_node = NodeForState(new_board, node)
                    self.children[(row, col, 4)] = state_node
                    return state_node
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
            max_value = -float('inf')
            for action in legal_actions:
                new_board, reward = get_afterstate_and_reward(board, action)
                value = self.approximator.value(convert_to_flat(new_board))
                action_value = (reward + value) / self.scaling
                max_value = max(max_value, action_value)
            return max_value


    def backpropagate(self, node, value, trajectory):
        node.visits += 1
        x = (value - node.value) / node.visits
        node.value += x

        for parent, action in reversed(trajectory):
            child = parent.children[action]
            value = child.value
            parent.visits += 1
            x = (value - parent.value) / parent.visits
            parent.value += x

    def best_action(self, root, legal_actions):
        best_action = None
        best_value = -float('inf')
        for action in legal_actions:
            if action not in root.children:
                value = 0
                if value > best_value:
                    best_value = value
                    best_action = action
            else:
                child = root.children[action]
                value = child.visits
                if value > best_value:
                    best_value = child.visits
                    best_action = action
        return best_action

# patterns = [
#     [0, 1, 2, 4, 5, 6], 
#     [1, 2, 5, 6, 9, 13],
#     [0, 1, 2, 3, 4, 5],
#     [0, 1, 5, 6, 7, 10],
#     [0, 1, 2, 5, 9, 10],
#     [0, 1, 5, 9, 13, 14],
#     [0, 1, 5, 8, 9, 13],
#     [0, 1, 2, 4, 6, 10]
# ]

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

def get_action(state, score):
    init_model()
    root = NodeForState(state)
    temp_env = mcts_agent.tmp_env(state)
    legal_actions = [a for a in range(4) if temp_env.is_move_legal(a)]
    if not legal_actions:
        action = 0
    else:
        for _ in range(mcts_agent.num_iterations):
            env = Game2048Env()
            env.board = root.board.copy()
            env.score = score
            leaf, path = mcts_agent.select(root, env, legal_actions)
            if leaf.visits > 0:
                leaf = mcts_agent.expand(leaf, env, path[-1][0] if path else None)
            value = mcts_agent.simulate(leaf)
            mcts_agent.backpropagate(leaf, value, path)
        action =  mcts_agent.best_action(root, legal_actions)
    # print("score:", score)
    # action, _ = best_move_selection(state, approximator)
    # print("action:", action)
    return action