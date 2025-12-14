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

#endif
