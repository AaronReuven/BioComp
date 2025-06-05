import numpy as np


class MagicSquareProblem:
    """
    Magic Square individual with in-place swaps and incremental O(1) fitness updates.
    """

    def __init__(self, size, seed=None, mode="standard"):
        """
        :param size: Side length N of the square.
        :param seed: RNG seed (or None for random).
        :param mode: "standard" or "most_perfect". If size % 4 == 0 and mode=="most_perfect",
                     extra most-perfect constraints should be added to try_swap() and fitness.
        """
        self.N = size
        self.mode = mode
        self.constant = size * (size * size + 1) // 2  #
        self.sub_square_constant = 2 * (size * size + 1)   #
        self.pair_constant = (size * size + 1)   #

        # Use a Python list of ints rather than NumPy array, to allow in-place integer operations
        rng = np.random.RandomState(seed)
        perm = rng.permutation(np.arange(1, size * size + 1))
        self.flat = perm.tolist()

        # Pre-allocate row sums, column sums, and diagonal sums
        self.row_sums = [0] * size
        self.col_sums = [0] * size
        self.main_diag = 0
        self.sec_diag = 0
        self.pairs_sums = [0] * size
        self.subsquare_sums = [0] * (size * size)

        # If most_perfect mode is required, you would also track half-diagonal pairs
        # and 2x2 block sums here.
        # data structures and updates in try_swap().

        # Compute initial sums and fitness
        self._compute_all_sums()
        self._compute_fitness()

    def _compute_all_sums(self):
        """
        Compute row_sums, col_sums, main_diag, sec_diag from self.flat.
        This is and is called only once at initialization or after a full rebuild.
        """
        N = self.N

        # Reset sums
        for i in range(N):
            self.row_sums[i] = 0
            self.col_sums[i] = 0
            self.pairs_sums[i] = 0
        for i in range(N*N):
            self.subsquare_sums[i] = 0
        self.main_diag = 0
        self.sec_diag = 0


        # Accumulate sums from the flat list
        for idx in range(N * N):
            v = self.flat[idx]
            r, c = divmod(idx, N)
            self.row_sums[r] += v
            self.col_sums[c] += v
            if r == c:
                self.main_diag += v
                if self.mode == 'most_perfect':
                    self.pairs_sums[r % (N//2)] += v
            if r + c == N - 1:
                self.sec_diag += v
                if self.mode == 'most_perfect':
                    self.pairs_sums[(N//2) + r % (N//2)] += v
            if self.mode == 'most_perfect':
                self.subsquare_sums[idx] += v
                self.subsquare_sums[(idx - 1) % (N*N)] += v
                self.subsquare_sums[(idx - N) % (N*N)] += v
                if idx % N == 0:
                    self.subsquare_sums[(idx + N - 1) % (N*N)] += v
                else:
                    self.subsquare_sums[(idx - N - 1) % (N*N)] += v

        pass
    #     2 , 12 , 5 , 3
    #     4 , 10 , 1 , 13
    #     8 , 11, 9, 16
    #     6, 14, 15, 7



    def _compute_fitness(self):
        """
        Compute the total  fitnessfrom row_sums, col_sums, and diagonals.
        Lower fitness is better; a perfect magic square has fitness == 0.
        """
        N = self.N
        M = self.constant
        M_P = self.pair_constant
        M_S = self.sub_square_constant
        f = 0

        for r in range(N):
            f += abs(self.row_sums[r] - M)
            f += abs(self.col_sums[r] - M)
            if self.mode == 'most_perfect':
                f += abs(self.pairs_sums[r] - M_P)
        if self.mode == 'most_perfect':
            for s in range(N*N):
                f += abs(self.subsquare_sums[s] - M_S)

        f += abs(self.main_diag - M)
        f += abs(self.sec_diag - M)

        self.fitness = int(f)

    def try_swap(self, i, j, keep_swap=False):
        """
        Attempt to swap cells at flat-indices i and j.
        If keep_swap is False, revert to the original configuration after computing candidate fitness.
        If keep_swap is True, commit the swap and update all sums and fitness in place.

        Returns the candidate fitness.
        """
        N = self.N
        M = self.constant
        M_P = self.pair_constant
        M_S = self.sub_square_constant

        #  swap
        v1 = self.flat[i]
        v2 = self.flat[j]
        r1, c1 = divmod(i, N)
        r2, c2 = divmod(j, N)

        old_penalty = abs(self.row_sums[r1] - M) + abs(self.col_sums[c1] - M)
        if r1 == c1:
            old_penalty += abs(self.main_diag - M)
            if self.mode == 'most_perfect':
                old_penalty += abs(self.pairs_sums[r1 % (N//2)] - M_P)
        if r1 + c1 == N - 1:
            old_penalty += abs(self.sec_diag - M)
            if self.mode == 'most_perfect':
                old_penalty += abs(self.pairs_sums[(N//2) + r1 % (N//2)] - M_P)
        if self.mode == 'most_perfect':
            old_penalty += abs(self.subsquare_sums[i] - M_S)
            old_penalty += abs(self.subsquare_sums[(i - 1) % (N * N)] - M_S)
            old_penalty += abs(self.subsquare_sums[(i - N) % (N * N)] - M_S)
            if i % N == 0:
                old_penalty += abs(self.subsquare_sums[(i + N - 1) % (N * N)] - M_S)
            else:
                old_penalty += abs(self.subsquare_sums[(i - N - 1) % (N * N)] - M_S)

        old_penalty += abs(self.row_sums[r2] - M) + abs(self.col_sums[c2] - M)
        if r2 == c2:
            old_penalty += abs(self.main_diag - M)
            if self.mode == 'most_perfect':
                old_penalty += abs(self.pairs_sums[r2 % (N//2)] - M_P)
        if r2 + c2 == N - 1:
            old_penalty += abs(self.sec_diag - M)
            if self.mode == 'most_perfect':
                old_penalty += abs(self.pairs_sums[(N//2) + r2 % (N//2)] - M_P)
        if self.mode == 'most_perfect':
            old_penalty += abs(self.subsquare_sums[j] - M_S)
            old_penalty += abs(self.subsquare_sums[(j - 1) % (N * N)] - M_S)
            old_penalty += abs(self.subsquare_sums[(j - N) % (N * N)] - M_S)
            if j % N == 0:
                old_penalty += abs(self.subsquare_sums[(j + N - 1) % (N * N)] - M_S)
            else:
                old_penalty += abs(self.subsquare_sums[(j - N - 1) % (N * N)] - M_S)




        self.flat[i], self.flat[j] = v2, v1

        self.row_sums[r1] += (v2 - v1)
        self.col_sums[c1] += (v2 - v1)
        if r1 == c1:
            self.main_diag += (v2 - v1)
            if self.mode == 'most_perfect':
                self.pairs_sums[r1 % (N//2)] += (v2 - v1)
        if r1 + c1 == N - 1:
            self.sec_diag += (v2 - v1)
            if self.mode == 'most_perfect':
                self.pairs_sums[(N//2) + r1 % (N // 2)] += (v2 - v1)
        if self.mode == 'most_perfect':
            self.subsquare_sums[i] += (v2 - v1)
            self.subsquare_sums[(i - 1) % (N * N)] += (v2 - v1)
            self.subsquare_sums[(i - N) % (N * N)] += (v2 - v1)
            if i % N == 0:
                self.subsquare_sums[(i + N - 1) % (N * N)] += (v2 - v1)
            else:
                self.subsquare_sums[(i - N - 1) % (N * N)] += (v2 - v1)


        self.row_sums[r2] += (v1 - v2)
        self.col_sums[c2] += (v1 - v2)
        if r2 == c2:
            self.main_diag += (v1 - v2)
            if self.mode == 'most_perfect':
                self.pairs_sums[r2 % (N//2)] += (v1 - v2)
        if r2 + c2 == N - 1:
            self.sec_diag += (v1 - v2)
            if self.mode == 'most_perfect':
                self.pairs_sums[(N//2) + r2 % (N // 2)] += (v1 - v2)
        if self.mode == 'most_perfect':
            self.subsquare_sums[j] += (v1 - v2)
            self.subsquare_sums[(j - 1) % (N * N)] += (v1 - v2)
            self.subsquare_sums[(j - N) % (N * N)] += (v1 - v2)
            if j % N == 0:
                self.subsquare_sums[(j + N - 1) % (N * N)] += (v1 - v2)
            else:
                self.subsquare_sums[(j - N - 1) % (N * N)] += (v1 - v2)


        new_penalty = abs(self.row_sums[r1] - M) + abs(self.col_sums[c1] - M)
        if r1 == c1:
            new_penalty += abs(self.main_diag - M)
            if self.mode == 'most_perfect':
                new_penalty += abs(self.pairs_sums[r1 % (N//2)] - M_P)
        if r1 + c1 == N - 1:
            new_penalty += abs(self.sec_diag - M)
            if self.mode == 'most_perfect':
                new_penalty += abs(self.pairs_sums[(N//2) + r1 % (N//2)] - M_P)
        if self.mode == 'most_perfect':
            new_penalty += abs(self.subsquare_sums[i] - M_S)
            new_penalty += abs(self.subsquare_sums[(i - 1) % (N * N)] - M_S)
            new_penalty += abs(self.subsquare_sums[(i - N) % (N * N)] - M_S)
            if i % N == 0:
                new_penalty += abs(self.subsquare_sums[(i + N - 1) % (N * N)] - M_S)
            else:
                new_penalty += abs(self.subsquare_sums[(i - N - 1) % (N * N)] - M_S)

        new_penalty += abs(self.row_sums[r2] - M) + abs(self.col_sums[c2] - M)
        if r2 == c2:
            new_penalty += abs(self.main_diag - M)
            if self.mode == 'most_perfect':
                new_penalty += abs(self.pairs_sums[r2 % (N//2)] - M_P)
        if r2 + c2 == N - 1:
            new_penalty += abs(self.sec_diag - M)
            if self.mode == 'most_perfect':
                new_penalty += abs(self.pairs_sums[(N//2) + r2 % (N//2)] - M_P)
        if self.mode == 'most_perfect':
            new_penalty += abs(self.subsquare_sums[j] - M_S)
            new_penalty += abs(self.subsquare_sums[(j - 1) % (N * N)] - M_S)
            new_penalty += abs(self.subsquare_sums[(j - N) % (N * N)] - M_S)
            if j % N == 0:
                new_penalty += abs(self.subsquare_sums[(j + N - 1) % (N * N)] - M_S)
            else:
                new_penalty += abs(self.subsquare_sums[(j - N - 1) % (N * N)] - M_S)

        candidate_fit = self.fitness - old_penalty + new_penalty

        if not keep_swap:
            # Revert all changes

            # Swap the values back
            self.flat[i], self.flat[j] = v1, v2

            # Revert row/column/diagonal sums
            self.row_sums[r1] -= (v2 - v1)
            self.col_sums[c1] -= (v2 - v1)
            if r1 == c1:
                self.main_diag -= (v2 - v1)
                if self.mode == 'most_perfect':
                    self.pairs_sums[r1 % (N // 2)] -= (v2 - v1)
            if r1 + c1 == N - 1:
                self.sec_diag -= (v2 - v1)
                if self.mode == 'most_perfect':
                    self.pairs_sums[(N // 2) + r1 % (N // 2)] -= (v2 - v1)

            if self.mode == 'most_perfect':
                self.subsquare_sums[i] -= (v2 - v1)
                self.subsquare_sums[(i - 1) % (N * N)] -= (v2 - v1)
                self.subsquare_sums[(i - N) % (N * N)] -= (v2 - v1)
                if i % N == 0:
                    self.subsquare_sums[(i + N - 1) % (N * N)] -= (v2 - v1)
                else:
                    self.subsquare_sums[(i - N - 1) % (N * N)] -= (v2 - v1)

            self.row_sums[r2] -= (v1 - v2)
            self.col_sums[c2] -= (v1 - v2)
            if r2 == c2:
                self.main_diag -= (v1 - v2)
                if self.mode == 'most_perfect':
                    self.pairs_sums[r2 % (N // 2)] -= (v1 - v2)
            if r2 + c2 == N - 1:
                self.sec_diag -= (v1 - v2)
                if self.mode == 'most_perfect':
                    self.pairs_sums[(N // 2) + r2 % (N // 2)] -= (v1 - v2)

            if self.mode == 'most_perfect':
                self.subsquare_sums[j] -= (v1 - v2)
                self.subsquare_sums[(j - 1) % (N * N)] -= (v1 - v2)
                self.subsquare_sums[(j - N) % (N * N)] -= (v1 - v2)
                if j % N == 0:
                    self.subsquare_sums[(j + N - 1) % (N * N)] -= (v1 - v2)
                else:
                    self.subsquare_sums[(j - N - 1) % (N * N)] -= (v1 - v2)

            return self.fitness
        else:
            self.fitness = candidate_fit
            return candidate_fit

    def copy(self):
        """
        Create a fresh clone of this individual, preserving all sums and fitness.
        Used for elitism or returning final solution.
        """
        clone = MagicSquareProblem(self.N, seed=None, mode=self.mode)
        clone.flat = list(self.flat)              # shallow copy of the list of ints
        clone.row_sums = list(self.row_sums)      # copy row sums
        clone.col_sums = list(self.col_sums)      # copy column sums
        clone.main_diag = self.main_diag
        clone.sec_diag = self.sec_diag
        clone.subsquare_sums = list(self.subsquare_sums)
        clone.pairs_sums = list(self.pairs_sums)
        clone.fitness = self.fitness
        return clone


class GeneticAlgorithm:
    """
    Genetic Algorithm driver that uses incremental swap-based fitness updates.
    """

    def __init__(
        self,
        problem_cls,
        problem_args=None,
        mutation_rate=0.05,
        elitism=0,
        learning_type=None,
        learning_cap=None,
        pop_size=100,
        seed=None,
    ):
        """
        :param problem_cls:   The class (MagicSquareProblem) used to instantiate individuals.
        :param problem_args:  Dict of args for problem_cls, e.g. {'size': N, 'mode': 'standard'}.
        :param mutation_rate: Float in [0,1] for per-cell swap‐mutation probability.
        :param elitism:       Number of top individuals to copy unchanged each generation.
        :param learning_type: None, "lamarkian", or "darwinian" for local search strategy.
        :param learning_cap:  Number of random-swap attempts per individual when learning_type is set.
        :param pop_size:      Total population size.
        :param seed:          RNG seed for initial population creation.
        """
        self.pop_size = pop_size
        self.problem_cls = problem_cls
        self.problem_args = problem_args or {}
        self.mutation_rate = mutation_rate
        self.elitism = elitism
        self.learning_type = learning_type
        self.learning_cap = learning_cap

        rng = np.random.RandomState(seed)
        self.population = []
        for _ in range(pop_size):
            indiv = self.problem_cls(**self.problem_args, seed=rng.randint(1 << 30))
            self.population.append(indiv)

    def generation_step(self, population):
        """
        1) Evaluate all fitnesses and sort ascending
         Copy over 'elitism' best individuals.
        Fill the rest by tournament‐selecting parents, one‐point crossover, and swap‐mutation.
        """
        # 1) Get current fitness
        fitness_list = [ind.fitness for ind in population]
        # Sort population indices by fitness ascending
        idx_sorted = sorted(range(len(population)), key=lambda i: fitness_list[i])
        sorted_pop = [population[i] for i in idx_sorted]
        sorted_fit = [fitness_list[i] for i in idx_sorted]

        new_pop = []
        #  Elitism:  the top elitismindividuals
        for i in range(self.elitism):
            new_pop.append(sorted_pop[i].copy())

        # 3)  rest of the population
        while len(new_pop) < self.pop_size:
            # Tournament selection of parent1
            i1, i2 = np.random.randint(0, self.pop_size), np.random.randint(0, self.pop_size)
            parent1 = sorted_pop[i1] if sorted_fit[i1] < sorted_fit[i2] else sorted_pop[i2]

            # Tournament selection of parent2
            j1, j2 = np.random.randint(0, self.pop_size), np.random.randint(0, self.pop_size)
            parent2 = sorted_pop[j1] if sorted_fit[j1] < sorted_fit[j2] else sorted_pop[j2]

            # build a brandnew flat list, then compute sums/fitness
            cut = np.random.randint(1, parent1.N * parent1.N)
            child_flat = [0] * (parent1.N * parent1.N)
            used = set()
            # Copy first parent up to cut
            for idx in range(cut):
                child_flat[idx] = parent1.flat[idx]
                used.add(parent1.flat[idx])
            # Fill remaining from parent2, skipping duplicates
            write_idx = cut
            for val in parent2.flat:
                if val not in used:
                    child_flat[write_idx] = val
                    write_idx += 1
                    if write_idx == parent1.N * parent1.N:
                        break

            # Create a new MagicSquareProblem from this flat list
            child = self.problem_cls(parent1.N, seed=None, mode=parent1.mode)
            child.flat = list(child_flat)
            child._compute_all_sums()
            child._compute_fitness()

            # Swap-mutation: each index has chance to be chosen; then we pair up chosen indices
            N2 = child.N * child.N
            rnd = np.random.random(N2)
            mut_idxs = np.where(rnd < self.mutation_rate)[0]
            if mut_idxs.size % 2 == 1:
                if mut_idxs.size != N2:
                    extra = next(i for i in range(N2) if i not in mut_idxs)
                    mut_idxs = np.append(mut_idxs, extra)
                else:
                    drop = np.random.choice(mut_idxs)
                    mut_idxs = mut_idxs[mut_idxs != drop]
            np.random.shuffle(mut_idxs)
            for k in range(0, mut_idxs.size, 2):
                a, b = mut_idxs[k], mut_idxs[k + 1]
                # Use try_swap with keep_swap=True to commit the swap and fitness update
                child.try_swap(a, b, keep_swap=True)

            new_pop.append(child)

        return new_pop

    def learning_step(self, population):
        """
        Apply local search (Lamarkian or Darwinian) to each individual if learning_type is set.
        Otherwise, return the population unchanged.
        """
        if not self.learning_type:
            return population

        rng = np.random.RandomState()  # separate RNG for local search
        for indiv in population:
            best_score = indiv.fitness
            for _ in range(self.learning_cap):
                N2 = indiv.N * indiv.N
                chosen = rng.choice(N2, size=N2, replace=False)
                improved = False
                for idx in chosen:
                    idx2 = rng.randint(0, N2)
                    while idx2 == idx:
                        idx2 = rng.randint(0, N2)
                    # Try the swap without committing
                    new_score = indiv.try_swap(idx, idx2, keep_swap=False)
                    if new_score < best_score:
                        # Commit the swap
                        indiv.try_swap(idx, idx2, keep_swap=True)
                        best_score = new_score
                        improved = True
                if not improved:
                    break
        return population

    def play(self, max_steps=100):
        """
        Run the GA for up to max_steps generations:
        1) (Optional) local search on each individual
        2) Produce next generation
        3) Track the best individual so far
        4) Stop early if fitness == 0
        Returns (best_individual, best_fitness).
        """
        best = None
        best_f = float("inf")

        for gen in range(max_steps):
            # 1) Local (Lamarkian/Darwinian) search
            self.population = self.learning_step(self.population)
            # 2) Next generation
            self.population = self.generation_step(self.population)
            # 3) Track the current best
            candidate = min(self.population, key=lambda x: x.fitness)
            if candidate.fitness < best_f:
                best_f = candidate.fitness
                best = candidate.copy()
            # 4) Early exit if perfect magic square found
            if best_f == 0:
                break

        return best, best_f


if __name__ == "__main__":
    N = 5
    POP_SIZE = 100
    MUT_RATE = 0.05
    ELITISM = 2
    LEARNING_TYPE = "lamarkian"  # or "darwinian", or None
    LEARNING_CAP = 10            # number of random-swap trials per individual
    MAX_GEN = 500
    SEED = 123

    ga = GeneticAlgorithm(
        MagicSquareProblem,
        problem_args={"size": N, "mode": "standard"},
        mutation_rate=MUT_RATE,
        elitism=ELITISM,
        learning_type=LEARNING_TYPE,
        learning_cap=LEARNING_CAP,
        pop_size=POP_SIZE,
        seed=SEED,
    )

    solution, score = ga.play(max_steps=MAX_GEN)
    print("Best fitness:", score)
    if solution is not None:
        print(np.array(solution.flat).reshape(N, N))

