#include "tsp.hpp"
#include <iostream>
#include <iomanip>

struct DE_params {
    static constexpr double NP = 10.0;
    static constexpr double CR = 0.9;
    static constexpr double F = 0.8; // typowe wartości z wikipedii
};

template<bool> class t_DE_continous_TSP_solution_set;

using Trial_vector_type_DE_continous_TSP_solution_set = t_DE_continous_TSP_solution_set<DONT_CACHE_THE_COST>;
// ten typ jest taki sam jak DE_continous_TSP_solution_set, ale bez cache'owania - dla trial vector'a jest to totalnie niepotrzebne, bo za każdym razem gdy obliczany jest koszt, wektor jest inny (cache'owanie nigdy nie wystąpi)

using DE_continous_TSP_solution_set = t_DE_continous_TSP_solution_set<CACHE_THE_COST>; // defaultowo cache'ujemy - jest to na 100% ulepszenie, nie trzeba testować. brak cache'owania tylko w typie dla trial vectora (powód wyżej)

template <bool should_cache = CACHE_THE_COST>
class t_DE_continous_TSP_solution_set : public base_TSP_solution_set<double, t_DE_continous_TSP_solution_set<should_cache>, should_cache> {
    using _my_base = base_TSP_solution_set<double, t_DE_continous_TSP_solution_set<should_cache>, should_cache>;
public:
    t_DE_continous_TSP_solution_set(std::size_t n_chromosomes) : _my_base(n_chromosomes) {
    }

    void generate_random(std::mt19937& gen, std::size_t n_chromosomes) {
        std::uniform_real_distribution<double> dis(0.0, 1.0);
        for (int i = 0; i < n_chromosomes; ++i) {
            this->values[i] = dis(gen); // literatura twierdzi, że nie trzeba sprawdzać i zmieniać duplikatów - przy sortowaniu podczas dyskretyzacji duplikaty nie dadzą problemów, szczególnie że prawie nigdy nie wystąpią
        }
        assert(!this->cached_cost_up_to_date);
    }

    distance_t _compute_cost(const TSP_Graph& graph) const {
        return discretize(graph.n_cities())._compute_cost(graph);
    }

    TSP_solution_set discretize(std::size_t n_chromosomes) const {
        TSP_solution_set out(n_chromosomes);

        using my_dict = std::pair<double, index_t>;
        std::unique_ptr<my_dict[]> val_index_map = std::make_unique<my_dict[]>(n_chromosomes);

        for (index_t i = 0; i < n_chromosomes; ++i) {
            val_index_map[i] = {this->values[i], i};
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
    DE_continous_TSP_solution_set* pop; // nie da się użyć unique_ptr - przy inicjalizacji unique_ptr<T[]> standard przewiduje tylko domyślną konstrukcję elementów w tablicy, co jest niemożliwe w tym przypadku - typ DE_continous_TSP_solution_set nie ma domyślnego konstruktora
    std::size_t n;
    Trial_vector_type_DE_continous_TSP_solution_set trial;
    //std::unique_ptr<index_t[]> array_indexes; // zakres 0..n, używany w funkcji evolve do losowania trzech indexów osobników którzy będą użyci do krosowania
    const TSP_Graph& graph;

    // dystrybucje używane w funkcji `evolve` - nie ma sensu konstruować je za każdym wywołaniem tej funkcji,
    // nie powinny one też być statyczne, ponieważ wtedy wprowadziłoby to ograniczenie - każda populacja w programie
    // musiałaby mieć taką samą długość chromosomu każdego osobnika. to ogranicznie raczej nie przeszkadzałoby w tym co chcę osiągnąć tym programem, ale po co sobie zamykać furtki na przyszłość
    std::uniform_real_distribution<double> evolve_distrib_r;
    std::uniform_int_distribution<index_t> evolve_distrib_chromosome_index;
    std::uniform_int_distribution<index_t> evolve_distrib_indivi_index;

public:
    DE_population(std::size_t n, const TSP_Graph& graph_ref)
    : n{n}
    , pop(static_cast<decltype(pop)>(operator new[](n * sizeof(DE_continous_TSP_solution_set))))
    , trial(graph_ref.n_cities())
    //, array_indexes(std::make_unique<index_t[]>(n))
    , graph{graph_ref}
    , evolve_distrib_r(0.0, 1.0)
    , evolve_distrib_chromosome_index(0, n_indivi() - 1)
    , evolve_distrib_indivi_index(0, n - 1)
    {
        for (index_t i = 0; i < n; ++i) {
            new(pop + i) DE_continous_TSP_solution_set(n_indivi()); // zauważona optymalizacja to implementacji: w programie każdy osobnik będzie miał te same n - nie ma sensu żeby każdy z nich to zapisywał
        }
    }

    // niepotrzebne - wolę usunąć
    DE_population(const DE_population&) = delete;
    DE_population& operator=(const DE_population&) = delete;
    DE_population(DE_population&&) = delete;
    DE_population& operator=(DE_population&&) = delete;

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
        operator delete[](pop);
    }

    void evolve(std::mt19937& gen) {
        for (index_t i = 0; i < n; ++i) {
            index_t idx_a, idx_b, idx_c;
            //std::sample(array_indexes.get(), std::next(array_indexes.get(), n), random_idxes, sizeof(random_idxes) / (sizeof(*random_idxes)), gen);

            do idx_a = evolve_distrib_indivi_index(gen); while (idx_a == i);
            do idx_b = evolve_distrib_indivi_index(gen); while (idx_b == i || idx_b == idx_a);
            do idx_c = evolve_distrib_indivi_index(gen); while (idx_c == i || idx_c == idx_a || idx_c == idx_b);

            const auto& a = pop[idx_a];
            const auto& b = pop[idx_b];
            const auto& c = pop[idx_c];

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
    // poczytaj w paperach jak jest stosowana ruletka/torunament
    // bo zcegos mi tu brakuje wlasnie jeszcze - gdzie to powinno byc?
    // poczytac o innych formach dyskretyzacji, bardziej wydajniejszych (radix sort?)
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