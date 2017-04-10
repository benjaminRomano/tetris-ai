from copy import deepcopy, copy

class HueristicAI():
    def __init__(self, game_state, height_weight, lines_weight, holes_weight, bumpiness_weight):
        self.game_state = game_state
        self.height_weight = height_weight
        self.lines_weight = lines_weight
        self.holes_weight = holes_weight
        self.bumpiness_weight = bumpiness_weight

    # 1 for forward, -1 for backward
    def compute_rotation_direction(self, start_rotation, target_rotation, num_rotations):
        flipDirection = 1
        if start_rotation > target_rotation:
            max_rotation = start_rotation
            min_rotation = target_rotation
        else:
            max_rotation = target_rotation
            min_rotation = start_rotation
            flipDirection = -1

        forwardDistance = num_rotations - max_rotation + min_rotation
        backwardDistance = max_rotation - min_rotation

        direction = 0

        if forwardDistance < backwardDistance:
            direction = 1
        else:
            direction = -1

        return flipDirection * direction

    def _get_piece_move(self, board, pieces, pieceIndex):
        best = None
        best_score = None
        best_actions = ['nothing']
        piece = pieces[pieceIndex]

        for num_rotation in range(piece['num_rotations']):
            rotated_piece = copy(piece)
            actions = []

            rotation_direction = self.compute_rotation_direction(rotated_piece['rotation'], num_rotation, rotated_piece['num_rotations'])

            isPossible = True

            while rotated_piece['rotation'] != num_rotation:
                if rotation_direction == 1:
                    if not self.game_state.rotateForward(board, rotated_piece):
                        isPossible = False
                        break

                    actions.append('rotateForward')
                else:
                    if not self.game_state.rotateBackward(board, rotated_piece):
                        isPossible = False
                        break
                    actions.append('rotateBackward')

            if not isPossible:
                continue

            while self.game_state.moveLeft(board, rotated_piece):
                actions.append('moveLeft')

            while self.game_state.isValidPosition(board, rotated_piece):
                placed_piece = copy(rotated_piece)
                test_board = deepcopy(board)

                self.game_state.drop(test_board, placed_piece)

                self.game_state.addToBoard(test_board, placed_piece)

                actions.append('drop')

                # Might not have enough space to move piece into the correct orientation
                if self.is_action_sequence_feasible(board, piece, actions):

                    cleared = self.game_state.removeCompleteLines(test_board)

                    if pieceIndex == len(pieces) - 1:
                        score = self.height_weight * self.game_state.get_aggregate_height(test_board) \
                            + self.lines_weight * cleared \
                            + self.holes_weight * self.game_state.get_num_holes(test_board) \
                            + self.bumpiness_weight * self.game_state.get_bumpiness(test_board) 
                    else:
                        _, _, score = self._get_piece_move(test_board, pieces, pieceIndex + 1)

                    if score > best_score or best_score == None:
                        best_score = score
                        best = copy(placed_piece)
                        best_actions = copy(actions)

                if not self.game_state.moveRight(board, rotated_piece):
                    break
                else:
                    actions.remove('drop')
                    if 'moveLeft' in actions:
                        actions.remove('moveLeft')
                    else:
                        actions.append('moveRight')

        return (best, [self.game_state.action_to_input(action) for action in best_actions], best_score)

    def is_action_sequence_feasible(self, board, piece, action_sequence):
        board = deepcopy(board)
        piece = copy(piece)

        for action in action_sequence:
            if action == 'moveLeft' and not self.game_state.moveLeft(board, piece):
                return False
            if action == 'moveRight' and not self.game_state.moveRight(board, piece):
                return False
            if action == 'drop':
                self.game_state.drop(board, piece)
            if action == 'rotateForward' and not self.game_state.rotateForward(board, piece):
                return False
            if action == 'rotateBackward' and not self.game_state.rotateBackward(board, piece):
                return False

        try:
            self.game_state.addToBoard(board, piece)
        except:
            return False

        return True

    def get_piece_move(self, board, pieces):
        return self._get_piece_move(board, pieces, 0)