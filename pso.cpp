#include "tsp.hpp"
#include "common.hpp"
#include <iostream> // debug

struct velocity_vec {
    std::unique_ptr<double[]> v;

    velocity_vec(std::size_t n)
    : v(std::make_unique<double[]>(n))
    {

    }

    void set(index_t i, double d) noexcept {
        v[i] = d;
    }

    double at(index_t i) const noexcept {
        return v[i];
    }

    double& mutable_at(index_t i) const noexcept {
        return v[i];
    }
};

template <bool should_cache = CACHE_THE_COST>
struct t_PSO_individual : public t_Continous_TSP_solution_set<should_cache> { // mozna by bylo trzymac w oddzielnych tablicach, ale wtedy musialby byc obliczany offset
    using base_chromosome = t_Continous_TSP_solution_set<should_cache>;
    velocity_vec v_vec;
    t_Continous_TSP_solution_set<CACHE_THE_COST> best_known_pos; // w tym ma byc caching

    t_PSO_individual(std::size_t n_indivi)
    : base_chromosome(n_indivi)
    , v_vec(n_indivi)
    , best_known_pos(n_indivi)
    {

    }

   void generate_random(std::mt19937& gen, std::size_t n_indivi) {
        static std::uniform_real_distribution<double> dis_ch(0.0, 1.0);
        const double vmax = 1.0 / static_cast<double>(n_indivi); 
        std::uniform_real_distribution<double> dis_vel(-vmax, vmax);

        for (index_t i = 0; i < n_indivi; ++i) {
            base_chromosome::values[i] = dis_ch(gen);
            v_vec.v[i] = dis_vel(gen);
        }

        best_known_pos.write_copy_of(*this, n_indivi);
        // sprawdz czy nie da sie dac po prostu *this?
        //if constepxr should cache.....
    }
/*
    decltype(auto) total_cost(const TSP_Graph& graph) const {
        return ch.total_cost(graph);
    }

    decltype(auto) discretize(std::size_t n_indivi) const {
        return ch.discretize(n_indivi);
    }

    decltype(auto) print(std::ostream& os, std::size_t n_indivi) const {
        return ch.print(os, n_indivi);
    }
        */
};

using PSO_individual = t_PSO_individual<CACHE_THE_COST>;
using PSO_individual_no_caching = t_PSO_individual<DONT_CACHE_THE_COST>;
// cacheowanie nie ma sensu - polzoenie zawsze sie zmienia
class PSO_population final : public Population<PSO_individual, PSO_individual_no_caching> {
    using _my_base = Population<PSO_individual, PSO_individual_no_caching>;

    Continous_TSP_solution_set best_known_pos; //  w tym ma byc caching
    double phi_p;
    double phi_g;
    double w;

public:
    PSO_population(std::size_t pop_size, const TSP_Graph& graph_ref, double phi_p, double phi_g, double w)
    : _my_base(pop_size, graph_ref)
    , best_known_pos(graph_ref.n_cities())
    , phi_p{phi_p}
    , phi_g{phi_g}
    , w{w}
    {

    }

    void generate_random(std::mt19937& gen) {
        for (index_t i = 0; i < n; ++i) {
            pop[i].generate_random(gen, n_genes());
        }

        debug_population_initialized = true;

        const auto& current_best = pop[best().index];
        best_known_pos.write_copy_of(current_best, n_genes());
    }

    index_cost_pair best() const { // override
        assert(debug_population_initialized && "best(): population wasn't initialized");
        index_cost_pair out = {0, pop[0].best_known_pos.total_cost(graph)};

        for (index_t i = 1; i < n; ++i) {
            const auto c = pop[i].best_known_pos.total_cost(graph);
            if (out.cost > c) {
                out = {i, c};
            }
        }

        return out;
    }

    void reflect_position_velocity(double& x, double& v, double vmax) {
        constexpr double X_MIN = 0.0;
        constexpr double X_MAX = 1.0;
        
        double V_MAX = vmax;
        double V_MIN = -V_MAX;

        // --- ODBICIE DLA x < 0 ---
        if (x < X_MIN) {
            x = X_MIN + (X_MIN - x);   // odbicie względem 0
            v = -v;
        }
        // --- ODBICIE DLA x > 1 ---
        else if (x > X_MAX) {
            x = X_MAX - (x - X_MAX);   // odbicie względem 1
            v = -v;
        }

        // --- JEŚLI PO ODBICIU DALEJ JEST POZA ZAKRESEM (duże v) ---
        // np. x = -2.3 → po odbiciu nadal < 0
        if (x < X_MIN) {
            x = X_MIN;
        } else if (x > X_MAX) {
            x = X_MAX;
        }

        // --- OGRANICZENIE PRĘDKOŚCI ---
        if (v < V_MIN) v = V_MIN;
        else if (v > V_MAX) v = V_MAX;
    }

    void evolve(std::mt19937& gen) {
        assert(debug_population_initialized);

        static std::uniform_real_distribution dis1(0.0, 1.0);
        const auto local_n_genes = n_genes();

        using gene_value_type = individual_type::value_type; // zmienic value_type na gene_type
        const auto normalized = [](gene_value_type val, double min, double max) -> gene_value_type {
            // val może być od -2 do 2 (w zaleznosci od F)
            return std::clamp(val, min, max);
        };

        for (index_t i = 0; i < n; ++i) {
            auto& particle_i = pop[i];

            for (index_t j = 0; j < local_n_genes; ++j) {
                const double r_p = dis1(gen);
                const double r_g = dis1(gen);
                auto& velocity_ij = particle_i.v_vec.mutable_at(j);
                auto& particle_ij = particle_i.mutable_at(j);

                velocity_ij = w * velocity_ij + phi_p * r_p
                            * (particle_i.best_known_pos.at(j) - particle_ij)
                            + phi_g * r_g * (best_known_pos.at(j) - particle_ij);

                //velocity_ij = normalized(velocity_ij, -1.0, 1.0); // zmienic te wartosci -1 i 1 na stałe
                velocity_ij = std::clamp(velocity_ij, -1.0, 1.0);
                particle_ij += velocity_ij;
                if (dis1(gen) < 0.05) {   // 5% szansy na szum
                    particle_ij += std::normal_distribution<double>(0, 0.02)(gen);
                }
                reflect_position_velocity(particle_ij, velocity_ij, 1.0 / static_cast<double>(local_n_genes));
                //particle_ij = normalized(particle_ij, 0.0, 1.0);
            }

            auto discrete_ch = particle_i.discretize(local_n_genes);
            perform_2opt(discrete_ch, graph, 10);

            const auto particle_i_cost = discrete_ch.total_cost(graph);
            if (particle_i_cost < particle_i.best_known_pos.total_cost(graph)) {
                particle_i.set_from_discrete(discrete_ch, local_n_genes);
                particle_i.best_known_pos.write_copy_of(particle_i, local_n_genes);

                if (particle_i_cost < best_known_pos.total_cost(graph)) {
                    best_known_pos.write_copy_of(particle_i, local_n_genes);
                }
            }
        }
    }

};

#include <chrono>

int main() {
    using namespace std::chrono;
    auto start = high_resolution_clock::now();

    const TSP_Graph graph(280, "./ALL_tsp/sformatowane/a280.tsp");
    std::random_device rd;
    std::mt19937 mt(rd());
 
    PSO_population pop(100, graph, 1.5, 1.5, 0.7); // liczba osobników w populacji
    pop.generate_random(mt);

    const auto best_start = pop.best().cost;

    std::cout << "\n\n best cost: " << best_start;

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

    const auto best_end = pop.best().cost;
    std::cout << "d: " << best_start - best_end;

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start).count();

    std::cout << "\nCzas wykonania: " << duration << " ms" << std::endl;

    return EXIT_SUCCESS;
}