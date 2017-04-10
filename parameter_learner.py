
import random
import math
from hueristic_ai import HueristicAI
from tetris import GameState
from copy import copy

class Candidate():
    def __init__(self, height_weight, lines_weight, holes_weight, bumpiness_weight):
        self.fitness = 0
        self.height_weight = height_weight
        self.lines_weight = lines_weight
        self.holes_weight = holes_weight
        self.bumpiness_weight = bumpiness_weight
        self.normalize()

    def normalize(self):
        denominator = math.sqrt(self.height_weight ** 2 + self.lines_weight ** 2 \
                + self.holes_weight ** 2 + self.bumpiness_weight ** 2)

        self.height_weight /= denominator
        self.lines_weight /= denominator
        self.holes_weight /= denominator
        self.bumpiness_weight /= denominator

    def mutate(self, mutate_value):
        quantity = random.uniform(-mutate_value, mutate_value)

        param_to_modify = random.randint(0, 3)

        if param_to_modify == 0:
            self.height_weight += quantity
        elif param_to_modify == 1:
            self.lines_weight += quantity
        elif param_to_modify == 2:
            self.holes_weight += quantity
        elif param_to_modify == 3:
            self.bumpiness_weight += quantity

        self.normalize()

    def display(self):
        print "height_weight = %s" % self.height_weight
        print "lines_weight = %s" % self.lines_weight
        print "holes_weight = %s" % self.holes_weight
        print "bumpiness_weight = %s" % self.bumpiness_weight

class ParameterLearner():

    def __init__(self, num_candidates=100, selection_rate=.3, tournament_rate=.1, mutation_rate=.05, mutation_value=.2):
        self.num_candidates = num_candidates
        self.selection_rate = selection_rate
        self.tournament_rate = tournament_rate
        self.mutation_rate = mutation_rate
        self.mutation_value = mutation_value

    def generate_random_candidate(self):
        height_weight = random.random() - .5
        lines_weight = random.random() - .5
        holes_weight = random.random() - .5
        bumpiness_weight = random.random() - .5

        return Candidate(height_weight, lines_weight, holes_weight, bumpiness_weight)

    def tune(self, num_of_games=5, max_num_of_moves=200, rounds=1000):
        candidates = []

        for _ in range(self.num_candidates):
            candidates.append(self.generate_random_candidate())

        # compute fitnesses
        self.compute_fitnesses(candidates, num_of_games, max_num_of_moves)

        candidates.sort(key=lambda x: x.fitness, reverse=True)

        count = 0

        while count < rounds:
            new_candidates = []
            for _ in range(int(self.num_candidates * self.selection_rate)):
                candidate1, candidate2 = self.select_tournament_pair(candidates, int(self.num_candidates * self.tournament_rate))
                candidate = self.cross_over(candidate1, candidate2)

                if random.random() < self.mutation_rate:
                    candidate.mutate(self.mutation_value)

                new_candidates.append(candidate)

            self.compute_fitnesses(new_candidates, num_of_games, max_num_of_moves)
            candidates = self.replace_candidates(candidates, new_candidates)
            total_fitness = 0

            for candidate in candidates:
                total_fitness += candidate.fitness

            print "Average fitness = %s" % (total_fitness / len(candidates))
            print "Highest fitness = %s" % candidates[0].fitness

            candidates[0].display()

            count += 1

        return candidates[0]

    def compute_fitnesses(self, candidates, num_of_games, max_num_of_moves):
        for candidate in candidates:
            ai = HueristicAI(GameState(), candidate.height_weight, candidate.lines_weight, \
                    candidate.holes_weight, candidate.bumpiness_weight)

            total_score = 0
            for _ in range(num_of_games):
                game_state = GameState()

                score = 0
                num_of_moves = 0
                terminal = False

                while num_of_moves < max_num_of_moves and not terminal:
                    score = game_state.score

                    if game_state.fallingPiece:
                        best_piece, _, _ = ai.get_piece_move(game_state.board, [game_state.fallingPiece, game_state.nextPiece])

                        game_state.fallingPiece = best_piece
                        _, _, terminal = game_state.frame_step(0)
                    else:
                        _, _, terminal = game_state.frame_step(0)

                    num_of_moves += 1

                total_score += score

            candidate.fitness = total_score

    def select_tournament_pair(self, candidates, ways):
        candidates = copy(candidates)
        candidates.sort(key=lambda x: x.fitness, reverse=True)

        indices = [i for i in range(len(candidates))]

        fittest_candidate_1_index = None
        fittest_candidate_2_index = None

        for _ in range(ways):
            selected_index = random.choice(indices)
            if fittest_candidate_1_index is None or selected_index < fittest_candidate_1_index:
                fittest_candidate_2_index = fittest_candidate_1_index
                fittest_candidate_1_index = selected_index
            elif fittest_candidate_2_index is None or selected_index < fittest_candidate_2_index:
                fittest_candidate_2_index = selected_index

        return candidates[fittest_candidate_1_index], candidates[fittest_candidate_2_index]

    def cross_over(self, candidate1, candidate2):
        height_weight = candidate1.fitness * candidate1.height_weight + candidate2.fitness * candidate2.height_weight
        lines_weight = candidate1.fitness * candidate1.lines_weight + candidate2.fitness * candidate2.lines_weight
        holes_weight = candidate1.fitness * candidate1.holes_weight + candidate2.fitness * candidate2.holes_weight
        bumpiness_weight = candidate1.fitness * candidate1.bumpiness_weight + candidate2.fitness * candidate2.bumpiness_weight

        return Candidate(height_weight, lines_weight, holes_weight, bumpiness_weight)

    def replace_candidates(self, candidates, new_candidates):
        candidates = candidates[:len(new_candidates) - 1]
        candidates = candidates + new_candidates

        candidates.sort(key=lambda x: x.fitness, reverse=True)
        return candidates


if __name__ == '__main__':
    parameter_learner = ParameterLearner()
    winner = parameter_learner.tune(num_of_games=3, max_num_of_moves=200)

    winner.display()