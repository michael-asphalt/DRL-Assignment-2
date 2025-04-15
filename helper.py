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