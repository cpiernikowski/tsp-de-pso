#ifndef COMMON_HPP
#define COMMON_HPP

#include "tsp.hpp"

using namespace tsp;

class Continous_population {
protected:
    std::size_t n;
    Continous_TSP_solution_set* pop; // niestety nie da się użyć unique_ptr - przy inicjalizacji unique_ptr<T[]> standard przewiduje tylko domyślną konstrukcję elementów w tablicy, co jest niemożliwe w tym przypadku - typ DE_continous_TSP_solution_set nie ma domyślnego konstruktora
    const TSP_Graph& graph;
    Continous_TSP_solution_set_no_caching trial;

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

    Continous_population(std::size_t pop_size, const TSP_Graph& graph_ref)
    : n{pop_size}
    , pop(static_cast<decltype(pop)>(operator new[](pop_size * sizeof(Continous_TSP_solution_set))))
    , graph{graph_ref}
    , trial(graph_ref.n_cities()) // to samo co n_genes()
    {
        for (index_t i = 0; i < n; ++i) {
            new(pop + i) Continous_TSP_solution_set(n_genes());
        }
    }

    // niepotrzebne - wolę usunąć
    Continous_population(const Continous_population&) = delete;
    Continous_population& operator=(const Continous_population&) = delete;
    Continous_population(Continous_population&&) = delete;
    Continous_population& operator=(Continous_population&&) = delete;

    std::size_t n_genes() const noexcept { // ile genów ma chromosom
        return graph.n_cities();
    }

    void generate_random(std::mt19937& gen) {
        for (index_t i = 0; i < n; ++i) {
            pop[i].generate_random(gen, n_genes());
        }
    }

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

    const Continous_TSP_solution_set& get(index_t i) const noexcept {
        return pop[i];
    }

    auto size() const noexcept {
        return n;
    }

    ~Continous_population() noexcept {
        for (index_t i = 0; i < n; ++i) {
            pop[i].~Continous_TSP_solution_set(); // nie jest trywialnie destruktowalny - ma w sobie unique_ptr
        }
        operator delete[](pop);
    }
};

#endif
