#ifndef COMMON_HPP
#define COMMON_HPP

#include "tsp.hpp"

using namespace tsp;

template <typename Individual_type, typename Trial_type>
class Population {
public: 
    using individual_type = Individual_type;
    using trial_type = Trial_type;
    // static_assert ze individual_type ma implementacje generate_random i total_cost
protected:
    [[maybe_unused]] bool debug_population_initialized = false;
    std::size_t n;
    individual_type* pop; // niestety nie da się użyć unique_ptr - przy inicjalizacji unique_ptr<T[]> standard przewiduje tylko domyślną konstrukcję elementów w tablicy, co jest niemożliwe w tym przypadku - typ DE_continous_TSP_solution_set nie ma domyślnego konstruktora
    const TSP_Graph& graph;
    trial_type trial;

public:
    struct index_cost_pair {
        index_t index;
        distance_t cost;

        // chce zeby to jednak był agregat, więc bez ctorów

        void make_invalid() noexcept {
            index = static_cast<index_t>(-1);
        }

        bool is_invalid() const noexcept {
            return (index == static_cast<index_t>(-1));
        }
    };

    Population(std::size_t pop_size, const TSP_Graph& graph_ref)
    : n{pop_size}
    , pop(static_cast<decltype(pop)>(operator new[](pop_size * sizeof(individual_type))))
    , graph{graph_ref}
    , trial(graph_ref.n_cities()) // to samo co n_genes()
    {
        for (index_t i = 0; i < n; ++i) {
            new(pop + i) individual_type(n_genes());
        }
    }

    // niepotrzebne - wolę usunąć
    Population(const Population&) = delete;
    Population& operator=(const Population&) = delete;
    Population(Population&&) = delete;
    Population& operator=(Population&&) = delete;

    std::size_t n_genes() const noexcept { // ile genów ma chromosom
        return graph.n_cities();
    }

    void generate_random(std::mt19937& gen) {
        for (index_t i = 0; i < n; ++i) {
            pop[i].generate_random(gen, n_genes());
        }
        debug_population_initialized = true;
    }

    index_cost_pair best() const {
        assert(debug_population_initialized && "best(): population wasn't initialized");
        index_cost_pair out = {0, pop[0].total_cost(graph)};

        for (index_t i = 1; i < n; ++i) {
            const auto c = pop[i].total_cost(graph);
            if (out.cost > c) {
                out = {i, c};
            }
        }

        return out;
    }

    const individual_type& get(index_t i) const noexcept {
        return pop[i];
    }

    auto size() const noexcept {
        return n;
    }

    ~Population() noexcept {
        for (index_t i = 0; i < n; ++i) {
            pop[i].~individual_type(); // nie jest trywialnie destruktowalny - ma w sobie unique_ptr
        }
        operator delete[](pop);
    }
};

template <typename ChromosomeType> // jakies ograniczenia na typ
void perform_2opt(ChromosomeType& ch, const TSP_Graph& graph, index_t max_iters) {
    const auto ch_ptr = ch.mutable_get_ptr(); // w teorii szybsze niz uzywanie at i set
    const auto n_genes_local = graph.n_cities();

    auto swap2opt = [ch_ptr](index_t start, index_t end) {
                while (start < end) {
                    //auto tmp = ch.at(start);
                    //ch.set(start, ch.at(end));
                    //ch.set(end, tmp);
                    auto tmp = ch_ptr[start];
                    ch_ptr[start] = ch_ptr[end];
                    ch_ptr[end] = tmp;
                
                    ++start;
                    --end;
                }
            };

    bool improved = true;
            const index_t max_iters_2opt = max_iters;
            index_t iters_counter = 0;

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
                        
                        const auto old_distance = graph.distance(ch_ptr[i_2opt], ch_ptr[i_2opt + 1])
                                                + graph.distance(ch_ptr[j_2opt], ch_ptr[j_2opt + 1]);

                        const auto new_distance = graph.distance(ch_ptr[i_2opt], ch_ptr[j_2opt])
                                                + graph.distance(ch_ptr[i_2opt + 1], ch_ptr[j_2opt + 1]);

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
                    swap2opt(best_i + 1, best_j); // + 1 bo 2opt działa tak, że best_i to początkowa krawedz, która jeszcze jest ok, dopiero od następnej chcemy zrobić swapa, aż do best_j (nie do best_j + 1!)
                }
                else {
                    improved = false;
                }
                ++iters_counter;
            }
}

#endif
