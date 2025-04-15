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
    def rec_rows(i, board, rows):
        if i >= 4:
            return rows
        row = board[i*4:(i+1)*4]
        rows.append(row)
        return rec_rows(i+1, board, rows)
    
    square = rec_rows(0, flat_board, [])
    def rec_reverse(row):
        if len(row) <= 1:
            return row
        return [row[-1]] + rec_reverse(row[:-1])
    
    def rec_reverse_rows(rows, out):
        if not rows:
            return out
        out.append(rec_reverse(rows[0]))
        return rec_reverse_rows(rows[1:], out)
    
    mirrored_square = rec_reverse_rows(square, [])
    def rec_flatten(lst):
        if not lst:
            return []
        return lst[0] + rec_flatten(lst[1:])
    
    return rec_flatten(mirrored_square)


def rotate_clockwise_flat(flat_board):
    def rec_make_square(i, board, sq):
        if i >= 4:
            return sq
        sq.append(board[i*4:(i+1)*4])
        return rec_make_square(i+1, board, sq)
    
    square = rec_make_square(0, flat_board, [])
    def rec_build_row(i, square, acc):
        if i >= 4:
            return acc
        def rec_build_element(j, i, row_acc):
            if j >= 4:
                return row_acc
            return rec_build_element(j+1, i, row_acc + [square[3-j][i]])
        new_row = rec_build_element(0, i, [])
        return rec_build_row(i+1, square, acc + [new_row])
    
    rotated_square = rec_build_row(0, square, [])
    def rec_flatten(lst):
        if not lst:
            return []
        return lst[0] + rec_flatten(lst[1:])
    
    return rec_flatten(rotated_square)


def rotate_flat(flat_board, cnt):
    if cnt <= 0:
        return flat_board[:]
    return rotate_flat(rotate_clockwise_flat(flat_board), cnt - 1)


class PatternFeature:
    def __init__(self, base_pattern, iso=8):
        self.base_pattern = base_pattern
        self.iso = iso
        self.table_size = 1 << (len(base_pattern) * 4)
        self.weights = [0.0] * self.table_size  
        self.isom = self.generate_symmetries()

    def generate_symmetries(self):
        const_board = list(range(16))
        iso_patterns = []

        def rec(i, patterns):
            if i >= self.iso:
                return patterns
            temp = const_board[:]
            if i >= 4:
                temp = mirror_flat(temp)
            temp = rotate_flat(temp, i % 4)
            def rec_transform(j, acc):
                if j >= len(self.base_pattern):
                    return acc
                return rec_transform(j+1, acc + [temp[self.base_pattern[j]]])
            transformed = rec_transform(0, [])
            patterns.append(transformed)
            return rec(i+1, patterns)
        
        return rec(0, iso_patterns)

    def feature_index(self, flat_board):
        indices = []
        
        def rec_features(patterns, acc):
            if not patterns:
                return acc
            pattern = patterns[0]
            def rec_inner(i, val):
                if i >= len(pattern):
                    return val
                return rec_inner(i+1, val | (flat_board[pattern[i]] << (4 * i)))
            idx = rec_inner(0, 0)
            acc.append(idx)
            return rec_features(patterns[1:], acc)
        
        return rec_features(self.isom, [])

    def estimate(self, flat_board):
        indices = self.feature_index(flat_board)
        
        def rec_sum(idx_list, total):
            if not idx_list:
                return total
            return rec_sum(idx_list[1:], total + self.weights[idx_list[0]])
        
        return rec_sum(indices, 0.0)

    def update(self, flat_board, adjust):
        indices = self.feature_index(flat_board)
        delta = adjust / len(indices) if indices else 0
        
        def rec_update(idx_list):
            if not idx_list:
                return
            self.weights[idx_list[0]] += delta
            rec_update(idx_list[1:])
        
        rec_update(indices)
        
        def rec_sum(idx_list, total):
            if not idx_list:
                return total
            return rec_sum(idx_list[1:], total + self.weights[idx_list[0]])
        
        return rec_sum(indices, 0.0)

    def load_from_stream(self, stream, size_t_fmt='I'):
        def compute_checksum(data):
            checksum_val = 0
            for byte in data:
                checksum_val = (checksum_val * 31 + byte) & 0xFFFFFFFF
            return checksum_val

        name_length_bytes = stream.read(4)
        name_length_integrity = compute_checksum(name_length_bytes)
        name_length = struct.unpack('i', name_length_bytes)[0]
        name_bytes = stream.read(name_length)
        name_integrity = compute_checksum(name_bytes)
        name_str = name_bytes.decode('utf-8')
        name_section_integrity = (name_length_integrity + name_integrity) & 0xFFFFFFFF
        size_t_size = struct.calcsize(size_t_fmt)
        weight_count_bytes = stream.read(size_t_size)
        weight_count_integrity = compute_checksum(weight_count_bytes)
        weight_count = struct.unpack(size_t_fmt, weight_count_bytes)[0]
        float_size = struct.calcsize('f')
        data_bytes = stream.read(float_size * weight_count)
        data_integrity = compute_checksum(data_bytes)
        overall_integrity = (name_section_integrity + weight_count_integrity + data_integrity) & 0xFFFFFFFF
        self.weights = list(struct.unpack(f'{weight_count}f', data_bytes))