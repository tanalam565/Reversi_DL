from typing import Type
import copy
import numpy as np
import math
import socket, pickle
from reversi import reversi

EMPTY = 0   # Empty Square
BLACK = -1  # Black Disc
WHITE = 1   # White Disc

DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1),
              ( 0, -1),          ( 0, 1),
              ( 1, -1), ( 1, 0), ( 1, 1)]

BLACK_PLAYER = -1
WHITE_PLAYER = 1

STRATEGY_V1 = 'Evaluate_Cost'     # Typical Approach
STRATEGY_V2 = 'Evaluate_Cost_r1'  # Add Stability and Game Phase
SELECTED = STRATEGY_V2

class Cost_Estimate:

    def Estimator(self, Estimator_Name, bord, turn):
        My_Estimator = getattr(self, Estimator_Name)
        return My_Estimator(bord, turn)
    ###=======================================================================###
    ###        Evaluation Cost Init: Basic and Typical Approach
    ###=======================================================================###
    def Evaluate_Cost(self, board, turn):
        My_Player = BLACK_PLAYER if turn == -1 else WHITE_PLAYER
        OP_Player = WHITE_PLAYER if turn == -1 else BLACK_PLAYER

        # Check the Num of Disc Diff
        my_discs = self.count_discs(board, My_Player)
        opp_discs = self.count_discs(board, OP_Player)
        disc_diff = my_discs - opp_discs

        # Mobility Score
        my_moves = len(self.get_Candi_moves(board, My_Player))
        opp_moves = len(self.get_Candi_moves(board, OP_Player))
        mobility = my_moves - opp_moves

        # Corner Position Score
        my_corners = self.count_corners(board, My_Player)
        opp_corners = self.count_corners(board, OP_Player)
        corner_occupancy = my_corners - opp_corners

        # Position based Score
        position_score = self.positional_score(board, My_Player)

        # Weight => Could be updated by
        weight_disc_diff = 10    # Range of diff: -64 ~ 64
        weight_mobility  = 78    # Range of Mobility: -30 ~ 30
        weight_corner    = 800   # Range of Corner: -4 ~ 4
        weight_position  = 100   # Range of Weight Position -1200 ~ 1200

        print("weight_disc_diff: {:<4}   disc_diff        :{}".format(weight_disc_diff, disc_diff))
        print("weight_mobility : {:<4}   mobility         :{}".format(weight_mobility , mobility))
        print("weight_corner   : {:<4}   corner_occupancy :{}".format(weight_corner   , corner_occupancy))
        print("weight_position : {:<4}   position_score   :{}".format(weight_position , position_score))

        # Total Cost
        score = (
                weight_disc_diff * disc_diff +
                weight_mobility * mobility +
                weight_corner * corner_occupancy +
                weight_position * position_score
        )

        return score, [my_discs, opp_discs]

    ###=======================================================================###
    ###        Evaluation Cost Rev1: Adding Game Phase and Stability
    ###=======================================================================###
    def Evaluate_Cost_r1(self, board, turn):
        My_Player = BLACK_PLAYER if turn == -1 else WHITE_PLAYER
        OP_Player = WHITE_PLAYER if turn == -1 else BLACK_PLAYER

        # Check Game Phase
        Game_Phase = self.get_game_phase(board)

        # Stability: cannot be flipped by the opponent
        stable = self.calculate_stability(board)

        # The Num of diff of Disc
        my_discs = self.count_discs(board, My_Player)
        opp_discs = self.count_discs(board, OP_Player)
        disc_diff = my_discs - opp_discs  # Value Range: -64 ~ +64

        # Mobility Socre
        my_moves = len(self.get_Candi_moves(board, My_Player))
        opp_moves = len(self.get_Candi_moves(board, OP_Player))
        mobility = my_moves - opp_moves  # Value Range: 대략 -30 ~ +30

        # Corner Score
        my_corners = self.count_corners(board, My_Player)
        opp_corners = self.count_corners(board, OP_Player)
        corner_occupancy = my_corners - opp_corners  # Value Range: -4 ~ +4

        # Calc Position based Weighted Score with Stable Disc Map
        # position_score = positional_score(board, My_Player)  # Value Range: -1200 ~ +1200
        position_score = self.positional_score_Unstable(board, My_Player, stable)  # Value Range: -1200 ~ +1200

        # Calc Stability Score
        stable_score = self.positional_score_Stable(board, My_Player, stable)

        # Weight Change by the num of empty Disc
        if Game_Phase == 'opening':
            weight_disc_diff = 10
            weight_mobility  = 150
            weight_corner    = 600
            weight_position  = 400
            weight_stable    = 0
        elif Game_Phase == 'midgame':
            weight_disc_diff = 10
            weight_mobility  = 78
            weight_corner    = 800
            weight_position  = 100
            weight_stable    = 100
        else:  # 'endgame'
            weight_disc_diff = 20
            weight_mobility  = 40
            weight_corner    = 1000
            weight_position  = 100
            weight_stable    = 100
        """"
        print("weight_disc_diff: {:<4}   disc_diff        :{}".format(weight_disc_diff, disc_diff))
        print("weight_mobility : {:<4}   mobility         :{}".format(weight_mobility , mobility))
        print("weight_corner   : {:<4}   corner_occupancy :{}".format(weight_corner   , corner_occupancy))
        print("weight_position : {:<4}   position_score   :{}".format(weight_position , position_score))
        print("weight_stable   : {:<4}   stable_score     :{}".format(weight_stable   , stable_score))
        print("Game Phase: ", Game_Phase)
        """
        # Total Score
        score = (
                weight_disc_diff * disc_diff +
                weight_mobility * mobility +
                weight_corner * corner_occupancy +
                weight_position * position_score +
                weight_stable * stable_score
        )

        return score, [my_discs, opp_discs]



    def get_opponent(self, My_Player):
        return BLACK_PLAYER if My_Player == WHITE_PLAYER else WHITE_PLAYER

    def count_corners(self, board: np.array, player)->int:
        corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
        corner_count = 0

        for x, y in corners:
            if board[x][y] == player:
                corner_count += 1

        return corner_count

    def positional_score(self, board:np.array, player):
        # Position based Weight
        weights = [
            [ 100, -20, 10,  5,  5, 10, -20, 100],
            [ -20, -50, -2, -2, -2, -2, -50, -20],
            [  10,  -2,  3,  2,  2,  3,  -2,  10],
            [   5,  -2,  2,  1,  1,  2,  -2,   5],
            [   5,  -2,  2,  1,  1,  2,  -2,   5],
            [  10,  -2,  3,  2,  2,  3,  -2,  10],
            [ -20, -50, -2, -2, -2, -2, -50, -20],
            [ 100, -20, 10,  5,  5, 10, -20, 100]
        ]

        score = 0
        for x in range(8):
            for y in range(8):
                if board[x][y] == player:
                    score += weights[x][y]               # Score of My Disc
                elif board[x][y] == self.get_opponent(player):
                    score -= weights[x][y]               # Score of Opponent's Disc

        return score

    def count_discs(self, board, player):
        #print(board == player)
        return np.sum(board == player)

    def get_Candi_moves(self, board, player):
        opponent = self.get_opponent(player)
        Candi_moves = []

        for x in range(8):
            for y in range(8):
                if board[x][y] != EMPTY:
                    continue

                for dx, dy in DIRECTIONS:
                    nx, ny = x + dx, y + dy
                    has_opponent_between = False

                    while 0 <= nx < 8 and 0 <= ny < 8 and board[nx][ny] == opponent:
                        nx += dx
                        ny += dy
                        has_opponent_between = True

                    if has_opponent_between and 0 <= nx < 8 and 0 <= ny < 8 and board[nx][ny] == player:
                        Candi_moves.append((x, y))
                        break

        return Candi_moves

    def get_game_phase(self, board):
        empty_count = np.sum(board == EMPTY)
        total_squares = 64

        if empty_count > 44:
            return 'opening'  # Num of Empty > 44
        elif empty_count > 20:
            return 'midgame'  # Num of Empty > 20
        else:
            return 'endgame'  # Else

    def calculate_stability(self, board):
        stable = [[False for _ in range(8)] for _ in range(8)]
        directions = DIRECTIONS

        # Find the stable disc
        corners = [(0, 0), (0, 7), (7, 0), (7, 7)]  # Corner Disc does not have any chance
        for x, y in corners:
            if board[x][y] != EMPTY:
                self.mark_stable_from_corner(board, stable, x, y)

        return stable

    def mark_stable_from_corner(self, board, stable, x, y):
        player = board[x][y]
        queue = [(x, y)]
        stable[x][y] = True

        while queue:
            cx, cy = queue.pop(0)
            for dx, dy in DIRECTIONS:
                nx, ny = cx + dx, cy + dy
                while 0 <= nx < 8 and 0 <= ny < 8:
                    if board[nx][ny] != player or stable[nx][ny]:
                        break
                    stable[nx][ny] = True
                    queue.append((nx, ny))
                    nx += dx
                    ny += dy

    # Give score at stable positions
    def positional_score_Stable(self, board, player, stable):
        # Position based Weight
        weights = [
            [ 20,  10,  10,  10,  10,  10,  10, 20],
            [ 10,  10,  10,  10,  10,  10,  10, 10],
            [ 10,  10,  10,  10,  10,  10,  10, 10],
            [ 10,  10,  10,  10,  10,  10,  10, 10],
            [ 10,  10,  10,  10,  10,  10,  10, 10],
            [ 10,  10,  10,  10,  10,  10,  10, 10],
            [ 10,  10,  10,  10,  10,  10,  10, 10],
            [ 20,  10,  10,  10,  10,  10,  10, 20]
        ]


        score = 0
        for x in range(8):
            for y in range(8):
                if not stable[x][y]:
                    continue  # Only consider the stable Disc
                if board[x][y] == player:
                    score += weights[x][y]
                    ##score += 10

                elif board[x][y] == self.get_opponent(player):
                    score -= weights[x][y]
                    ##score -= 10

        return score

    def positional_score_Unstable(self, board, player, stable):
        # Position based Weight
        weights = [
            [100, -20, 10,  5,  5, 10, -20, 100],
            [-20, -50, -2, -2, -2, -2, -50, -20],
            [ 10,  -2,  3,  2,  2,  3,  -2,  10],
            [  5,  -2,  2,  1,  1,  2,  -2,   5],
            [  5,  -2,  2,  1,  1,  2,  -2,   5],
            [ 10,  -2,  3,  2,  2,  3,  -2,  10],
            [-20, -50, -2, -2, -2, -2, -50, -20],
            [100, -20, 10,  5,  5, 10, -20, 100]
        ]

        score = 0
        for x in range(8):
            for y in range(8):
                if board[x][y] == player and not stable[x][y]:
                    score += weights[x][y]
                elif board[x][y] == self.get_opponent(player) and not stable[x][y]:
                    score -= weights[x][y]

        return score



def get_Candi_moves(board, player):
    opponent = BLACK_PLAYER if player == WHITE_PLAYER else WHITE_PLAYER
    Candi_moves = []

    for row in range(8):
        for col in range(8):
            if board[row][col] != EMPTY:
                continue

            for dx, dy in DIRECTIONS:
                x, y = row + dx, col + dy
                has_opponent_between = False

                while 0 <= x < 8 and 0 <= y < 8 and board[x][y] == opponent:
                    x += dx
                    y += dy
                    has_opponent_between = True

                if has_opponent_between and 0 <= x < 8 and 0 <= y < 8 and board[x][y] == player:
                    Candi_moves.append((row, col))
                    break

    return Candi_moves


def make_move(board, player, move):
    opponent = BLACK_PLAYER if player == WHITE_PLAYER else WHITE_PLAYER
    x, y = move
    board[x][y] = player

    for dx, dy in DIRECTIONS:
        nx, ny = x + dx, y + dy
        stones_to_flip = []

        while 0 <= nx < 8 and 0 <= ny < 8 and board[nx][ny] == opponent:
            stones_to_flip.append((nx, ny))
            nx += dx
            ny += dy

        if 0 <= nx < 8 and 0 <= ny < 8 and board[nx][ny] == player:
            for fx, fy in stones_to_flip:
                board[fx][fy] = player

def minimax(board, depth, alpha, beta, maximizing_player, turn):
    opponent = BLACK_PLAYER if turn == WHITE_PLAYER else WHITE_PLAYER

    Candi_moves = get_Candi_moves(board, turn if maximizing_player else opponent)

    Calc_Cost: Type[Cost_Estimate] = Cost_Estimate()

    if not Candi_moves:
        score, prediction = Calc_Cost.Estimator(SELECTED, board, turn)
        return score, prediction, [-1, -1]

    if depth == 0:
        score, prediction = Calc_Cost.Estimator(SELECTED, board, turn)
        return score, prediction, None

    best_move = None
    best_prediction = []

    if maximizing_player:
        max_eval = -math.inf
        for move in Candi_moves:
            #new_board = [row[:] for row in board]
            new_board = copy.deepcopy(board)
            make_move(new_board, turn, move)
            eval, prediction, _ = minimax(new_board, depth - 1, alpha, beta, False, turn)

            #if(depth == 4):
            #    print("Move: ", move[0], move[1])
            if eval > max_eval:
                max_eval = eval
                best_move = move
                best_prediction = prediction
            alpha = max(alpha, eval)
            if beta <= alpha:
                break  # Cut Beta
        return max_eval, best_prediction,best_move
    else:
        min_eval = math.inf
        for move in Candi_moves:
            #new_board = [row[:] for row in board]
            new_board = copy.deepcopy(board)
            make_move(new_board, opponent, move)
            eval, prediction, _ = minimax(new_board, depth - 1, alpha, beta, True, turn)
            if eval < min_eval:
                min_eval = eval
                best_move = move
                best_prediction = prediction
            beta = min(beta, eval)
            if beta <= alpha:
                break  # Cut Alpha
        return min_eval, best_prediction, best_move


def display_disc_gauge(My_discs, Op_discs):
    max_discs = 64  # 8x8
    length = 50     # Lenght of gauge

    # My_Player
    My_percentage = (My_discs / max_discs) * 100
    My_filled_length = int(length * My_percentage // 100)
    My_bar = '■' * My_filled_length + ' ' * (length - My_filled_length)

    # Opp_Player
    Opp_percentage = (Op_discs / max_discs) * 100
    Opp_filled_length = int(length * Opp_percentage // 100)
    Opp_bar = '□' * Opp_filled_length + ' ' * (length - Opp_filled_length)

    # Opp_Player
    Pro_percentage = ((My_discs + Op_discs) / max_discs) * 100
    Pro_filled_length = int(length * Pro_percentage // 100)
    Pro_bar = '□' * Pro_filled_length + ' ' * (length - Pro_filled_length)


    print("#===== Predicted Result =====#")
    print(f"Progress ({My_discs + Op_discs:2}/64): |{Pro_bar}| {Pro_percentage:.2f}%")
    print(f"My Discs ({My_discs:2}/64): |{My_bar}| {My_percentage:.2f}%")
    print(f"OP Discs ({Op_discs:2}/64): |{Opp_bar}| {Opp_percentage:.2f}%\n")


def main():
    game_socket = socket.socket()
    game_socket.connect(('127.0.0.1', 33333))
    game = reversi()

    while True:

        # Receive play request from the server
        # turn : 1 --> you are playing as white | -1 --> you are playing as black
        # board : 8*8 numpy array
        data = game_socket.recv(4096)
        turn, board = pickle.loads(data)

        # Turn = 0 indicates game ended
        if turn == 0:
            game_socket.close()
            return

        # Debug info
        print(turn)
        print(board)

        Calc_Cost, prediction, move = minimax(board, depth=4, alpha=-math.inf, beta=math.inf, maximizing_player=True, turn=turn)

        #print("Move => ", move[0], move[1])
        #print("Prediction => My_Player: {:<3}    Opp_Player: {:<3}".format(prediction[0], prediction[1]))

        # Display the predicted Result
        display_disc_gauge(prediction[0],prediction[1])

        # Final Decision
        x = move[0]
        y = move[1]


        # Send your move to the server. Send (x,y) = (-1,-1) to tell the server you have no hand to play
        game_socket.send(pickle.dumps([x, y]))


def main__():
    board = np.zeros([8, 8])

    board[3, 4] = -1
    board[3, 3] = 1
    board[4, 3] = -1
    board[4, 4] = 1

    turn = BLACK_PLAYER

    Calc_Cost, move = minimax(board, depth=64, alpha=-math.inf, beta=math.inf, maximizing_player=True, turn=turn)

    x = move[0]
    y = move[1]


if __name__ == '__main__':
    main()