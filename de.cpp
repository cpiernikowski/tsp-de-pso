#include "tsp.hpp"
#include "common.hpp"
#include <iostream>
#include <iomanip>

using namespace tsp;

struct DE_params {
    //static constexpr unsigned NP = 1000;
    // constexpr double CR = 0;
    static constexpr double F = 0.7;
};

class DE_population : public Continous_population {
    // dystrybucje używane w funkcji `evolve` - nie ma sensu konstruować je za każdym wywołaniem tej funkcji,
    // nie powinny one też być statyczne, ponieważ wtedy wprowadziłoby to ograniczenie - każda populacja w programie
    // musiałaby mieć taką samą długość chromosomu każdego osobnika. to ogranicznie raczej nie przeszkadzałoby w tym co chcę osiągnąć tym programem, ale po co sobie zamykać furtki na przyszłość
    std::uniform_real_distribution<double> evolve_distrib_r;
    std::uniform_int_distribution<index_t> evolve_distrib_chromosome_index;
    std::uniform_int_distribution<index_t> evolve_distrib_indivi_index;
    index_cost_pair current_best_info;
    double CR;

    static constexpr double CR_default = 0.05;

public:
    DE_population(std::size_t pop_size, const TSP_Graph& graph_ref)
    : Continous_population(pop_size, graph_ref)
    , evolve_distrib_r(0.0, 1.0)
    , evolve_distrib_chromosome_index(0, graph_ref.n_cities() - 1)
    , evolve_distrib_indivi_index(0, pop_size - 1)
    , CR{CR_default}
    {
        current_best_info.make_invalid();
    }

    void evolve(std::mt19937& gen) {
        const auto random_idx_exclusive = [this, &gen](auto... other_idxes) -> index_t {
            index_t out;
            do out = evolve_distrib_indivi_index(gen); while (((out == other_idxes) || ...));
            return out;
        };

        using gene_value_type = Continous_TSP_solution_set::value_type;
        const auto normalize = [](gene_value_type val) -> gene_value_type {
            // val może być od -2 do 2 (w zaleznosci od F)
            val = std::fmod(val, gene_value_type{1.0});

            if (val < gene_value_type{0.0}) { // sprawdzic czy ten check jest potrzebny w ogole - czy fmod zwraca tez ujemne?
                //val += gene_value_type{1.0};
                val = -val;
            }

            return val;
        };

        const auto n_genes_local = n_genes(); // mozna by było to pominąć, ale jakoś mi się nie podoba wywoływanie tej funkcji tak dużo razy, nawet jeśli jest optymalizowana przez kompilator do zwykłego read'a. przy tym rozwiązaniu mamy teoretycznie mniejszą zależność od grafu

        for (index_t i = 0; i < n; ++i) {
            const auto idx_a = random_idx_exclusive(i);
            const auto idx_b = random_idx_exclusive(i, idx_a);
            const auto idx_c = random_idx_exclusive(i, idx_a, idx_b);

            const auto& a = pop[idx_a];
            const auto& b = pop[idx_b];
            const auto& c = pop[idx_c];

            auto& x = pop[i]; // wektor bazowy

            const auto R = evolve_distrib_chromosome_index(gen);

            for (index_t j = 0; j < n_genes_local; ++j) {
                const auto ri = evolve_distrib_r(gen);

                if (ri < this->CR || j == R) {
                    trial.set(
                        j,
                        normalize(a.at(j) + DE_params::F * (b.at(j) - c.at(j)))
                    );
                } else {
                    trial.set(j, x.at(j));
                }
            }

            // 2opt loop
            auto discrete_trial = trial.discretize<DONT_CACHE_THE_COST>(n_genes_local);

            auto discrete_trial_swap2opt = [&discrete_trial](index_t start, index_t end) {
                while (start < end) {
                    auto tmp = discrete_trial.at(start);
                    discrete_trial.set(start, discrete_trial.at(end));
                    discrete_trial.set(end, tmp);
                
                    ++start;
                    --end;
                }
            };

            std::uniform_real_distribution<double> dis_2opt(0.0, 1.0); // czasem uzywam, czasem nie - idea jest taka zeby tylko dla jakiegoś procenta populacji przeprowadzać 2opt
            const auto rand_2opt = dis_2opt(gen);

            bool improved = true;
            const auto max_iters_2opt = n_genes_local;
            std::remove_const_t<decltype(max_iters_2opt)> iters_counter = 0;

            while (improved && iters_counter < max_iters_2opt) {
                distance_t best_delta = 0;
                static constexpr index_t invalid_best_index_ij = std::numeric_limits<index_t>::max();
                index_t best_i = invalid_best_index_ij;
                index_t best_j = invalid_best_index_ij;

                for (index_t i_2opt = 0; i_2opt < n_genes_local - 2; ++i_2opt) {
                    for (index_t j_2opt = i_2opt + 2; j_2opt < n_genes_local - 1; ++j_2opt) {

                        // przykład dla miast A B C D E F G:
                        // dla i_2opt = 1 czyli indeks miasta B
                        // dla j_2opt = 5 czyli indeks miasta F
                        // jeśli odleglosc_miedzy(B, C)+odleglosc_miedzy(F, G) > odleglosc_miedzy(B, F)+odleglosc_miedzy(C, G):
                        //      zamień_kolejnością(od C do F) # czyli od indeksu i_2opt+1 do j_2opt
                        // wynik: A B F E D C G
                        // trzeba zamienić kolejność, żeby D nadal było połączone z E, oraz E było połączone z F, tak jak przed zmianą
                        
                        const auto old_distance = graph.distance(discrete_trial.at(i_2opt), discrete_trial.at(i_2opt + 1))
                                                + graph.distance(discrete_trial.at(j_2opt), discrete_trial.at((j_2opt + 1)));

                        const auto new_distance = graph.distance(discrete_trial.at(i_2opt), discrete_trial.at(j_2opt))
                                                + graph.distance(discrete_trial.at(i_2opt + 1), discrete_trial.at((j_2opt + 1)));

                        const auto distance_delta = old_distance - new_distance;

                        if (distance_delta > best_delta) {
                            best_delta = distance_delta;
                            best_i = i_2opt;
                            best_j = j_2opt;
                        }
                    }
                }

                if (best_delta > 0) {
                    assert(best_i != invalid_best_index_ij && best_j != invalid_best_index_ij);
                    discrete_trial_swap2opt(best_i + 1, best_j); // + 1 bo 2opt działa tak, że best_i to początkowa krawedz, która jeszcze jest ok, dopiero od następnej chcemy zrobić swapa, aż do best_j (nie do best_j + 1!)
                }
                else {
                    improved = false;
                }
                ++iters_counter;
            }
            // end 2opt loop. todo: zobic z tego fn call?

            if (discrete_trial.total_cost(graph) < x.total_cost(graph)) {
                x.set_from_discrete<DONT_CACHE_THE_COST>(discrete_trial, n_genes_local);
            }
        }

        //const auto best_this_iter = this->best();
        //if (best_this_iter.cost >= current_best_info.cost) {
        //    this->CR = 0.0;
        //}
        //else {
        //    this->CR = CR_default;
        //    current_best_info = best_this_iter;
        //}
    }

    const auto& get_best_info() const noexcept {
        return current_best_info;
    }
};


#include <chrono>

int main() {
    // poczytac o innych formach dyskretyzacji, bardziej wydajnych dla mojego typu danych (0.0 do 1.0) (radix sort?)

    /*
        
    */

    using namespace std::chrono;
    auto start = high_resolution_clock::now();

    const TSP_Graph graph(280, "./ALL_tsp/sformatowane/a280.tsp");
    std::random_device rd;
    std::mt19937 mt(rd());
 
    DE_population pop(100, graph); // liczba osobników w populacji
    pop.generate_random(mt);

    std::cout << "\n\n best cost: " << pop.best().cost;

    for (int i = 0; i < 30; ++i) { // liczba ewolucji
        pop.evolve(mt);
        const auto best = pop.best();
        std::cout << "\n\n best cost: " << best.cost;
        std::cout << "\n";

        const auto& best_indiv = pop.get(best.index);
        best_indiv.print(std::cout, pop.n_genes());
        std::cout << '\n';
        best_indiv.discretize(pop.n_genes()).print(std::cout, pop.n_genes());
        std::cout << "=========\n";
    }

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start).count();

    std::cout << "\nCzas wykonania: " << duration << " ms" << std::endl;

    return EXIT_SUCCESS;
}