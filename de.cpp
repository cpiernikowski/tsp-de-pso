#include "tsp.hpp"
#include <iostream>
#include <iomanip>

using namespace tsp;

struct DE_params {
    //static constexpr unsigned NP = 1000;
    static constexpr double CR = 0.9;
    static constexpr double F = 0.9; // typowe wartości z wikipedii
};

template<bool> class t_DE_continous_TSP_solution_set;

using Trial_vector_type_DE_continous_TSP_solution_set = t_DE_continous_TSP_solution_set<DONT_CACHE_THE_COST>;
// ten typ jest taki sam jak DE_continous_TSP_solution_set, ale bez cache'owania - dla trial vector'a jest to totalnie niepotrzebne, bo za każdym razem gdy obliczany jest koszt, wektor jest inny (cache'owanie nigdy nie wystąpi)

using DE_continous_TSP_solution_set = t_DE_continous_TSP_solution_set<CACHE_THE_COST>; // defaultowo cache'ujemy brak cache'owania tylko w typie dla trial vectora (powód wyżej)

template <bool should_cache = CACHE_THE_COST> // przenies ten template do internal ns?
class t_DE_continous_TSP_solution_set : public base_TSP_solution_set<double, t_DE_continous_TSP_solution_set<should_cache>, should_cache> {

    using _my_base = base_TSP_solution_set<double, t_DE_continous_TSP_solution_set<should_cache>, should_cache>;

public:
    using typename _my_base::value_type;

    t_DE_continous_TSP_solution_set(std::size_t n_chromosomes) : _my_base(n_chromosomes) {
    }

    template <bool local_should_cache = CACHE_THE_COST>
    void set_from_discrete(const t_TSP_solution_set<local_should_cache>& discrete, std::size_t n_genes) {
        const double denom = static_cast<double>(n_genes - 1);

        for (index_t rank = 0; rank < n_genes; ++rank) {
            const auto city = discrete.at(rank);
            this->set(city, static_cast<value_type>(static_cast<double>(rank) / denom));
        }
    }

    void generate_random(std::mt19937& gen, std::size_t n_chromosomes) {
        static std::uniform_real_distribution<double> dis(0.0, 1.0);

        for (index_t i = 0; i < n_chromosomes; ++i) {
            this->values[i] = dis(gen); // literatura twierdzi, że nie trzeba sprawdzać i zmieniać duplikatów - przy sortowaniu podczas dyskretyzacji duplikaty nie dadzą problemów, szczególnie że prawie nigdy nie wystąpią
        }
        
        if constexpr (should_cache) {
            assert(!_my_base::cache.up_to_date);
        }
    }

    distance_t _compute_cost(const TSP_Graph& graph) const {
        return discretize(graph.n_cities())._compute_cost(graph);
    }

    template <bool local_should_cache = CACHE_THE_COST>
    auto discretize(std::size_t n_chromosomes) const {
        t_TSP_solution_set<local_should_cache> out(n_chromosomes);

        using my_dict = std::pair<value_type, index_t>;
        static std::unique_ptr<my_dict[]> val_index_map = std::make_unique<my_dict[]>(n_chromosomes);

        for (index_t i = 0; i < n_chromosomes; ++i) {
            val_index_map[i] = {this->values[i], i};
        }

        const auto my_dict_comparator = [](const my_dict& a, const my_dict& b) {
            return a.first < b.first;
        };

        std::sort(val_index_map.get(), std::next(val_index_map.get(), n_chromosomes), my_dict_comparator); // pomyslec czy jest jakis algorytm sortujący dobrze radzacy sobie z tym typem danych (double od 0 do 1)

        for (index_t i = 0; i < n_chromosomes; ++i) {
            out.set(i, static_cast<city_index_t>(val_index_map[i].second));
        }

        return out; // RVO
    }
};

class DE_population {
    std::size_t n;
    DE_continous_TSP_solution_set* pop; // niestety nie da się użyć unique_ptr - przy inicjalizacji unique_ptr<T[]> standard przewiduje tylko domyślną konstrukcję elementów w tablicy, co jest niemożliwe w tym przypadku - typ DE_continous_TSP_solution_set nie ma domyślnego konstruktora
    const TSP_Graph& graph;
    Trial_vector_type_DE_continous_TSP_solution_set trial;

    // dystrybucje używane w funkcji `evolve` - nie ma sensu konstruować je za każdym wywołaniem tej funkcji,
    // nie powinny one też być statyczne, ponieważ wtedy wprowadziłoby to ograniczenie - każda populacja w programie
    // musiałaby mieć taką samą długość chromosomu każdego osobnika. to ogranicznie raczej nie przeszkadzałoby w tym co chcę osiągnąć tym programem, ale po co sobie zamykać furtki na przyszłość
    std::uniform_real_distribution<double> evolve_distrib_r;
    std::uniform_int_distribution<index_t> evolve_distrib_chromosome_index;
    std::uniform_int_distribution<index_t> evolve_distrib_indivi_index;

public:
    DE_population(std::size_t pop_size, const TSP_Graph& graph_ref)
    : n{pop_size}
    , pop(static_cast<decltype(pop)>(operator new[](pop_size * sizeof(DE_continous_TSP_solution_set))))
    , graph{graph_ref}
    , trial(graph_ref.n_cities()) // to samo co n_genes()
    , evolve_distrib_r(0.0, 1.0)
    , evolve_distrib_chromosome_index(0, graph_ref.n_cities() - 1)
    , evolve_distrib_indivi_index(0, pop_size - 1)
    {
        for (index_t i = 0; i < n; ++i) {
            new(pop + i) DE_continous_TSP_solution_set(n_genes());
        }
    }

    // niepotrzebne - wolę usunąć
    DE_population(const DE_population&) = delete;
    DE_population& operator=(const DE_population&) = delete;
    DE_population(DE_population&&) = delete;
    DE_population& operator=(DE_population&&) = delete;

    std::size_t n_genes() const noexcept { // ile genów ma chromosom
        return graph.n_cities();
    }

    void generate_random(std::mt19937& gen) {
        for (index_t i = 0; i < n; ++i) {
            pop[i].generate_random(gen, n_genes());
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

    auto size() const noexcept {
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

        using gene_value_type = DE_continous_TSP_solution_set::value_type;
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

                if (ri < DE_params::CR || j == R) {
                    trial.set(
                        j,
                        normalize(a.at(j) + DE_params::F * (b.at(j) - c.at(j)))
                    );
                } else {
                    trial.set(j, x.at(j));
                }
            }

            // 2opt loop
            auto discrete_trial = trial.discretize<DONT_CACHE_THE_COST>(n_genes_local); // gdy uda sie zrobic dyskretny bez cachowania to w tej metodzie daj template <bool>

            auto discrete_trial_swap2opt = [&discrete_trial](index_t start, index_t end) {
                while (start < end) {
                    auto tmp = discrete_trial.at(start);
                    discrete_trial.set(start, discrete_trial.at(end));
                    discrete_trial.set(end, tmp);
                
                    ++start;
                    --end;
                }
            };

            bool improved = true;
            const auto max_iters_2opt = n_genes_local;
            std::remove_const_t<decltype(max_iters_2opt)> iters_counter = 0;

            do {
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
            } while (improved && iters_counter < max_iters_2opt);
            // end 2opt loop. todo: make this a function call

            if (discrete_trial.total_cost(graph) < x.total_cost(graph)) {
                x.set_from_discrete<DONT_CACHE_THE_COST>(discrete_trial, n_genes_local);
            }
        }
    }
};

#include <chrono>

int main() {
    // poczytac o innych formach dyskretyzacji, bardziej wydajniejszych (radix sort?)
    const TSP_Graph graph(280, "./ALL_tsp/sformatowane/a280.tsp");
    std::random_device rd;
    std::mt19937 mt(rd());
 
    DE_population pop(10, graph);
    pop.generate_random(mt);

    std::cout << "\n\n best cost: " << pop.best().cost;

    using namespace std::chrono;
    auto start = high_resolution_clock::now();

    for (int i = 0; i < 5; ++i) {
        pop.evolve(mt);
        const auto best = pop.best();
        std::cout << "\n\n best cost: " << pop.best().cost;
        std::cout << "\n";

        const auto& best_indiv = pop.get(best.index);
        best_indiv.print(std::cout, pop.n_genes());
        std::cout << '\n';
        best_indiv.discretize(pop.n_genes()).print(std::cout, pop.n_genes());
        std::cout << "=========\n";
    }

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start).count();

    std::cout << "\nCzas wykonania: " << duration << " us" << std::endl;

    return EXIT_SUCCESS;
}