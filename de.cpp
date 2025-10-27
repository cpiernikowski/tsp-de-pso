#include "tsp.hpp"
#include <iostream>
#include <iomanip>
#define TSP_DE_POPULATION_ENABLE_EVOLUTION_SAMPLE_APPROACH

struct DE_params {
    static constexpr double NP = 10.0;
    static constexpr double CR = 0.9;
    static constexpr double F = 0.8; // typowe wartości z wikipedii
};

class DE_continous_TSP_solution_set : public base_TSP_solution_set<double> {
public:
    DE_continous_TSP_solution_set(std::size_t n_chromosomes) : base_TSP_solution_set(n_chromosomes) {
    }

    void generate_random(std::mt19937& gen, std::size_t n_chromosomes) {
        std::uniform_real_distribution<double> dis(0.0, 1.0);
        for (int i = 0; i < n_chromosomes; ++i) {
            values[i] = dis(gen); // literatura twierdzi, że nie trzeba sprawdzać i zmieniać duplikatów - przy sortowaniu podczas dyskretyzacji duplikaty nie dadzą problemów, szczególnie że prawie nigdy nie wystąpią
        }
    }

    distance_t total_cost(const TSP_Graph& graph) const {
        return discretize(graph.n_cities()).total_cost(graph);
    }

    TSP_solution_set discretize(std::size_t n_chromosomes) const {
        TSP_solution_set out(n_chromosomes);

        using my_dict = std::pair<double, index_t>;
        std::unique_ptr<my_dict[]> val_index_map = std::make_unique<my_dict[]>(n_chromosomes);

        for (index_t i = 0; i < n_chromosomes; ++i) {
            val_index_map[i] = {values[i], i};
        }

        auto my_dict_comparator = [](const my_dict& a, const my_dict& b) {
            return a.first < b.first;
        };

        std::sort(val_index_map.get(), std::next(val_index_map.get(), n_chromosomes), my_dict_comparator);

        for (index_t i = 0; i < n_chromosomes; ++i) {
            out.set(i, val_index_map[i].second);
        }

        return out; // RVO
    }
};

class DE_population {
    DE_continous_TSP_solution_set* pop; // nie da się użyć unique_ptr - przy inicjalizacji standard przewiduje tylko domyślną konstrukcję, co jest niemożliwe w tym przypadku - typ DE_continous_TSP_solution_set nie ma domyślnego konstruktora
    std::size_t n;
    DE_continous_TSP_solution_set trial;
    #ifdef TSP_DE_POPULATION_ENABLE_EVOLUTION_SAMPLE_APPROACH
    std::unique_ptr<index_t[]> array_indexes;
    #endif
    const TSP_Graph& graph;

    // dystrybucje używane w funkcji `evolve` - nie ma sensu konstruować je za każdym wywołaniem tej funkcji,
    // nie powinny one też być statyczne, ponieważ wtedy wprowadziłoby to ograniczenie - każda populacja w programie
    // musiałaby mieć taką samą długość chromosomu każdego osobnika. to ogranicznie raczej nie przeszkadzałoby w tym co chcę osiągnąć tym programem, ale po co sobie zamykać furtki na przyszłość
    std::uniform_real_distribution<double> evolve_distrib_r;
    std::uniform_int_distribution<index_t> evolve_distrib_chromosome_index;

public:
    DE_population(std::size_t n, const TSP_Graph& graph_ref)
    : n{n}
    , pop(static_cast<decltype(pop)>(operator new[](n * sizeof(DE_continous_TSP_solution_set))))
    , trial(graph_ref.n_cities())
    , array_indexes(std::make_unique<index_t[]>(n))
    , graph{graph_ref}
    , evolve_distrib_r(0.0, 1.0)
    , evolve_distrib_chromosome_index(0, n_indivi())
    {
        for (index_t i = 0; i < n; ++i) {
            new(pop + i) DE_continous_TSP_solution_set(n_indivi()); // zauważona optymalizacja to implementacji: w programie każdy osobnik będzie miał te same n - nie ma sensu żeby każdy z nich to zapisywał
        }

        #ifdef TSP_DE_POPULATION_ENABLE_EVOLUTION_SAMPLE_APPROACH
        std::iota(array_indexes.get(), std::next(array_indexes.get(), n), 0);
        #endif
    }

    std::size_t n_indivi() const noexcept {
        return graph.n_cities();
    }

    void generate_random(std::mt19937& gen) {
        for (int i = 0; i < n; ++i) {
            pop[i].generate_random(gen, n_indivi());
        }
    }

    struct index_cost_pair {
        index_t index;
        distance_t cost;
    };

    index_cost_pair best() const {
        index_cost_pair out = {0, pop[0].total_cost(graph)};

        for (index_t i = 1; i < n; ++i) {
            const auto c = pop[i].total_cost(graph);
            if (out.cost > c) {
                out = {i, c};
            }
        }

        return out;
    }

    const DE_continous_TSP_solution_set& get(index_t i) const noexcept {
        return pop[i];
    }

    auto get_n() const noexcept {
        return n;
    }

    ~DE_population() {
        for (index_t i = 0; i < n; ++i) {
            pop[i].~DE_continous_TSP_solution_set(); // nie jest trywialnie destruktowalny - ma w sobie unique_ptr
        }
        delete[] pop;
    }

    void evolve(std::mt19937& gen) {
        for (index_t i = 0; i < n; ++i) {
            index_t random_idxes[3];
            std::sample(array_indexes.get(), std::next(array_indexes.get(), n), random_idxes, sizeof(random_idxes) / (sizeof(*random_idxes)), gen);

            const auto& a = pop[random_idxes[0]];
            const auto& b = pop[random_idxes[1]];
            const auto& c = pop[random_idxes[2]];

            auto& x = pop[i];

            const auto R = evolve_distrib_chromosome_index(gen);

            for (index_t j = 0; j < n_indivi(); ++j) {
                const auto ri = evolve_distrib_r(gen);

                if (ri < DE_params::CR || j == R) {
                    trial.set(
                        j,
                        std::clamp(a.at(j) + DE_params::F * (b.at(j) - c.at(j)), 0.0, 1.0)
                    );
                } else {
                    trial.set(j, x.at(j));
                }

            }

            if (trial.total_cost(graph) <= x.total_cost(graph)) {
                x.write_copy_of(trial, n_indivi());
            }
        }
    }
};


int main() {
    // cacheowanie wyniku total cost
    // poczytaj w paperach jak jest stosowana ruletka/torunament
    // bo zcegos mi tu brakuje wlasnie jeszcze - gdzie to powinno byc?
    const TSP_Graph graph(10, "./tsp_example1.txt");
    std::random_device rd;
    std::mt19937 mt(rd());

    DE_population pop(500, graph);
    pop.generate_random(mt);

    std::cout << "\n\n best cost: " << pop.best().cost;

    for (int i = 0; i < 30; ++i)
        pop.evolve(mt);

    std::cout << "\n\n best cost: " << pop.best().cost;

    

    return EXIT_SUCCESS;
}