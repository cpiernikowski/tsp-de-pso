#include "tsp.hpp"
#include <iostream>
#include <iomanip>

struct DE_params {
    //static constexpr unsigned NP = 1000;
    static constexpr double CR = 0.9;
    static constexpr double F = 0.9; // typowe wartości z wikipedii
};

template<bool> class t_DE_continous_TSP_solution_set;

using Trial_vector_type_DE_continous_TSP_solution_set = t_DE_continous_TSP_solution_set<DONT_CACHE_THE_COST>;
// ten typ jest taki sam jak DE_continous_TSP_solution_set, ale bez cache'owania - dla trial vector'a jest to totalnie niepotrzebne, bo za każdym razem gdy obliczany jest koszt, wektor jest inny (cache'owanie nigdy nie wystąpi)

using DE_continous_TSP_solution_set = t_DE_continous_TSP_solution_set<CACHE_THE_COST>; // defaultowo cache'ujemy - jest to na 100% ulepszenie, nie trzeba testować. brak cache'owania tylko w typie dla trial vectora (powód wyżej)

template <bool should_cache = CACHE_THE_COST>
class t_DE_continous_TSP_solution_set : public base_TSP_solution_set<double, t_DE_continous_TSP_solution_set<should_cache>, should_cache> {
    using _my_base = base_TSP_solution_set<double, t_DE_continous_TSP_solution_set<should_cache>, should_cache>;
    using typename _my_base::value_type;
public:
    t_DE_continous_TSP_solution_set(std::size_t n_chromosomes) : _my_base(n_chromosomes) {
    }

    void set_from_discrete(const TSP_solution_set& discrete, std::size_t n_genes) {
        const double denom = static_cast<double>(n_genes - 1);
        for (index_t rank = 0; rank < n_genes; ++rank) {
            const city_index_t city = discrete.at(rank);
            this->set(city, static_cast<value_type>(static_cast<double>(rank) / denom));
        }
    }

    void generate_random(std::mt19937& gen, std::size_t n_chromosomes) {
        std::uniform_real_distribution<double> dis(0.0, 1.0);//make static?
        for (index_t i = 0; i < n_chromosomes; ++i) {
            this->values[i] = dis(gen); // literatura twierdzi, że nie trzeba sprawdzać i zmieniać duplikatów - przy sortowaniu podczas dyskretyzacji duplikaty nie dadzą problemów, szczególnie że prawie nigdy nie wystąpią
        }
        assert(!this->cached_cost_up_to_date);
    }

    distance_t _compute_cost(const TSP_Graph& graph) const {
        return discretize(graph.n_cities())._compute_cost(graph);
    }

    TSP_solution_set discretize(std::size_t n_chromosomes) const {
        TSP_solution_set out(n_chromosomes);

        using my_dict = std::pair<value_type, index_t>; // zmieniono z double na value_type. nie pownno sie nic zjebac
        std::unique_ptr<my_dict[]> val_index_map = std::make_unique<my_dict[]>(n_chromosomes);

        for (index_t i = 0; i < n_chromosomes; ++i) {
            val_index_map[i] = {this->values[i], i};
        }

        auto my_dict_comparator = [](const my_dict& a, const my_dict& b) {
            return a.first < b.first;
        };

        std::sort(val_index_map.get(), std::next(val_index_map.get(), n_chromosomes), my_dict_comparator);

        for (index_t i = 0; i < n_chromosomes; ++i) {
            out.set(i, static_cast<city_index_t>(val_index_map[i].second));
        }

        return out; // RVO
    }
};

class DE_population {
    std::size_t n;
    DE_continous_TSP_solution_set* pop; // nie da się użyć unique_ptr - przy inicjalizacji unique_ptr<T[]> standard przewiduje tylko domyślną konstrukcję elementów w tablicy, co jest niemożliwe w tym przypadku - typ DE_continous_TSP_solution_set nie ma domyślnego konstruktora
    const TSP_Graph& graph;
    Trial_vector_type_DE_continous_TSP_solution_set trial;
    //std::unique_ptr<index_t[]> array_indexes; // zakres 0..n, używany w funkcji evolve do losowania trzech indexów osobników którzy będą użyci do krosowania

    // dystrybucje używane w funkcji `evolve` - nie ma sensu konstruować je za każdym wywołaniem tej funkcji,
    // nie powinny one też być statyczne, ponieważ wtedy wprowadziłoby to ograniczenie - każda populacja w programie
    // musiałaby mieć taką samą długość chromosomu każdego osobnika. to ogranicznie raczej nie przeszkadzałoby w tym co chcę osiągnąć tym programem, ale po co sobie zamykać furtki na przyszłość
    std::uniform_real_distribution<double> evolve_distrib_r;
    std::uniform_int_distribution<index_t> evolve_distrib_chromosome_index;
    std::uniform_int_distribution<index_t> evolve_distrib_indivi_index;

public:
    DE_population(std::size_t n, const TSP_Graph& graph_ref)
    : n{n} // zmien nazwe argumentu
    , pop(static_cast<decltype(pop)>(operator new[](n * sizeof(DE_continous_TSP_solution_set))))
    , graph{graph_ref}
    , trial(graph_ref.n_cities())
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
        for (index_t i = 0; i < n; ++i) {
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

    auto get_n() const noexcept { // zmien na size()
        return n;
    }

    ~DE_population() {
        for (index_t i = 0; i < n; ++i) {
            pop[i].~DE_continous_TSP_solution_set(); // nie jest trywialnie destruktowalny - ma w sobie unique_ptr
        }
        operator delete[](pop);
    }

    void evolve(std::mt19937& gen) {
        const auto random_idx_exclusive = [this, &gen](auto... other_idxes) -> index_t {
            index_t out;
            do out = evolve_distrib_indivi_index(gen); while (((out == other_idxes) || ...));
            return out;
        };

        const auto normalize = [](double val) -> double {
            val = std::fmod(val, 1.0);
            if (val < 0.0) {
                val += 1.0;
            }
            return val;
        };

        for (index_t i = 0; i < n; ++i) {
            const auto idx_a = random_idx_exclusive(i);
            const auto idx_b = random_idx_exclusive(i, idx_a);
            const auto idx_c = random_idx_exclusive(i, idx_a, idx_b);

            const auto& a = pop[idx_a];
            const auto& b = pop[idx_b];
            const auto& c = pop[idx_c];

            auto& x = pop[i]; // wektor bazowy

            const auto R = evolve_distrib_chromosome_index(gen);
            const auto n_genes = n_indivi();

            for (index_t j = 0; j < n_genes; ++j) {
                const auto ri = evolve_distrib_r(gen);

                if (ri < DE_params::CR || j == R) {
                    trial.set(
                        j,
                        normalize(a.at(j) + DE_params::F * (b.at(j) - c.at(j)))
                    );
                } else {
                    trial.set(j, x.at(j)); // pomysl kiedys nad przecniesiem tego triala tutaj do funkcji - moze bedzie bardziej cache friendly - NA KONIEC
                }
            }

            auto swap2opt = [](TSP_solution_set& discrete_trial, index_t start, index_t end) {
                while (start < end) {
                    //std::swap(discrete_trial.values[start], discrete_trial.values[end]);
                    auto tmp = discrete_trial.at(start);
                    discrete_trial.set(start, discrete_trial.at(end));
                    discrete_trial.set(end, tmp);

                    ++start;
                    --end;
                }
            };

            // 2opt loop
            auto discrete_trial = trial.discretize(n_genes); // gdy uda sie zrobic dyskretny bez cachowania to w tej metodzie daj template <bool>
            bool improved = true;
            do {
                distance_t best_delta = 0;
                static constexpr index_t invalid_best_index_ij = std::numeric_limits<index_t>::max();
                index_t best_i = invalid_best_index_ij;
                index_t best_j = invalid_best_index_ij; // polegają na tym, że czytane są tylko w wypadku, gdy best_delta > 0, a inicjalizowane są własnie wtedy, gdy best_delta przyjmuje wartość > 0, jest to więc bezpieczne

                for (index_t i_2opt = 0; i_2opt < n_genes - 2; ++i_2opt) {
                    for (index_t j_2opt = i_2opt + 2; j_2opt < n_genes - 1; ++j_2opt) {
                        if (i_2opt == 0 && j_2opt == n_genes - 1) {
                            continue;
                        }

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
                    swap2opt(discrete_trial, best_i + 1, best_j);
                }
                else {
                    improved = false;
                }
            } while (improved);
            // end 2opt loop. todo: make this a function call

            if (discrete_trial.total_cost(graph) < x.total_cost(graph)) {
                x.set_from_discrete(discrete_trial, n_genes);
            }
        }
    }
};


int main() {
    // poczytac o innych formach dyskretyzacji, bardziej wydajniejszych (radix sort?)
    // lepiej poudkladac w klasach skladowe i zwrocic uwage na koejnosc inicjalizacji - zrobic taka sama w jakiej sa zadeklarowane
    const TSP_Graph graph(52, "./berlin52.txt");
    //const TSP_Graph graph(10, "./tsp_example1.txt");
    std::random_device rd;
    std::mt19937 mt(rd());
 
    DE_population pop(30, graph);
    pop.generate_random(mt);

    std::cout << "\n\n best cost: " << pop.best().cost;

    for (int i = 0; i < 10; ++i) {
        pop.evolve(mt);
        const auto best = pop.best();
        std::cout << "\n\n best cost: " << pop.best().cost;
        std::cout << "\n";

        const auto& best_indiv = pop.get(best.index);
        best_indiv.print(std::cout, pop.n_indivi());
        std::cout << '\n';
        best_indiv.discretize(pop.n_indivi()).print(std::cout, pop.n_indivi());
        std::cout << "=========\n";
    }

    

    return EXIT_SUCCESS;
}