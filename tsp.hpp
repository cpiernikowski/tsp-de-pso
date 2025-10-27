#ifndef TSP_HPP
#define TSP_HPP
#include <cstdlib>
#include <cstddef>
#include <fstream>
#include <string>
#include <cassert>
#include <cmath>
#include <memory>
#include <random>
#include <numeric>
#include <algorithm>
#include <iostream>

using index_t = std::size_t;
using distance_t = double;
using point_t = std::pair<int, int>;
using city_index_t = int;

static constexpr bool CACHE_THE_COST = true;
static constexpr bool DONT_CACHE_THE_COST = false;

namespace util {
    auto euclidean_distance(point_t first, point_t second) {
        return distance_t{std::hypot(second.first - first.first, second.second - first.second)};
    }

    template <typename T>
    class square_mat {
        std::unique_ptr<T[]> p;
        std::size_t n;

    public:
        square_mat(std::size_t n_rows)
        : p(std::make_unique<T[]>(n_rows * n_rows))
        , n{n_rows} {
        }

        T& at(index_t i, index_t j) noexcept {
            const auto idx = i * n + j;
            assert(idx < n * n);
            return p[idx];
        }

        const T& at(index_t i, index_t j) const noexcept {
            const auto idx = i * n + j;
            assert(idx < n * n);
            return p[idx];
        }

        void set(index_t i, index_t j, const T& t) noexcept {
            const auto idx = i * n + j;
            assert(idx < n * n);
            p[idx] = t;
        }
    };
}

class TSP_Graph { // symetryczny TSP
    util::square_mat<distance_t> distances;
    std::size_t n;

public:
    TSP_Graph(std::size_t n_cities, const char* filename)
    : distances(n_cities)
    , n{n_cities} {
        std::ifstream f(filename);
        auto coords = std::make_unique<point_t[]>(n);
        std::string line;

        for (int i = 0; std::getline(f, line); ++i) {
            const auto idx_y = line.find_last_of(' ') + 1;
            const int y = std::stoi(line.substr(idx_y));
            auto idx_x = idx_y - 2;
            for (; line[idx_x] != ' '; --idx_x);
            ++idx_x;
            const int x = std::stoi(line.substr(idx_x, idx_y - idx_x - 1));
            coords[i] = {x, y};
        }

        f.close();

        for (index_t i = 0; i < n; ++i) {
            for (index_t j = 0; j < n; ++j) {
                // dałoby się oczywiście przechować tylko górną część macierzy, ale to by się wiązało z większą ilością operacji przy obliczaniu indexu
                // ([i*n+j], a [i * n - (i * (i - 1)) / 2 + (j - i))]) oraz zamiana `i` z `j` jeśli j < i, zdecydowałem więc, że wolę poświęcić połowę więcej pamięci dla tablicy
                // szczególnie biorąc pod uwagę, że tablica będzie bardzo często indeksowana przy implementacji PSO i DE
                distances.set(i, j, util::euclidean_distance(coords[i], coords[j]));
            }
        }
    }

    auto distance(index_t i, index_t j) const noexcept {
        return distances.at(i, j);
    }

    auto n_cities() const noexcept {
        return n;
    }

    void print(std::ostream& os, const char* tail) const {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                os << distances.at(i, j) << ' ';
            }
            os << '\n';
        }
        os << tail;
    }
};

template <typename T, typename Derived, bool should_cache = CACHE_THE_COST>
class base_TSP_solution_set {
    using value_type = T;
protected:
    std::unique_ptr<value_type[]> values;
    mutable bool cached_cost_up_to_date = false;
    mutable distance_t cost_cache;

    base_TSP_solution_set(std::size_t n_chromosomes)
    : values(std::make_unique<value_type[]>(n_chromosomes)) {
    }

    base_TSP_solution_set(base_TSP_solution_set&& other)
    : values(std::move(other.values)) {
    }

    base_TSP_solution_set& operator=(base_TSP_solution_set&& other) {
        if constexpr (should_cache) {
            cached_cost_up_to_date = false;
        }
        values = std::move(other.values);
        return *this;
    }

public:
    distance_t total_cost(const TSP_Graph& graph) const {
        if constexpr (should_cache) {
            if (cached_cost_up_to_date)
                return cost_cache;

            static_assert(
                std::is_same_v<
                    distance_t,
                    decltype(std::declval<const Derived&>()._compute_cost(std::declval<const TSP_Graph&>()))
                >,
                "Derived class must implement: distance_t _compute_cost(const TSP_Graph&) const" // metoda _compute_cost nie powinna być wywoływana nigdzie poza tą metodą
            );

            cost_cache = static_cast<const Derived&>(*this)._compute_cost(graph);
            cached_cost_up_to_date = true;
            return cost_cache;
        }
        else {
            return static_cast<const Derived&>(*this)._compute_cost(graph);
        }
    }

    const T* get_values_raw_ptr() const noexcept {
        return values.get();
    }

    template <typename OtherDerived, bool OtherCache>
    void write_copy_of(const base_TSP_solution_set<T, OtherDerived, OtherCache>& other, std::size_t n_chromosomes) {
        if constexpr (should_cache) {
            cached_cost_up_to_date = false;
        }
        std::copy_n(other.get_values_raw_ptr(), n_chromosomes, values.get());
    }

    const value_type& at(index_t i) const noexcept {
        return values[i];
    }

    void set(index_t i, value_type t) noexcept {
        if constexpr (should_cache) {
            cached_cost_up_to_date = false;
        }
        values[i] = t;
    }

    void print(std::ostream& os, std::size_t n_chromosomes, const char* tail = "") const {
        const auto end_idx = n_chromosomes - 1;

        os << '[';

        for (index_t i = 0; i < end_idx; ++i) {
            os << values[i] << ", ";
        }

        os << values[end_idx] << ']' << tail;
    }
};

class TSP_solution_set : public base_TSP_solution_set<city_index_t, TSP_solution_set> {
public:
    TSP_solution_set(std::size_t n) : base_TSP_solution_set(n) {
    }

    distance_t _compute_cost(const TSP_Graph& graph) const {
        distance_t out{};
        const auto n = graph.n_cities();

        for (int i = 1; i < n; ++i) {
            out += graph.distance(values[i - 1], values[i]);
        }

        out += graph.distance(values[n - 1], values[0]); // powrót do miasta startowego

        return out;
    }

    void generate_random(std::mt19937& gen, std::size_t n_chromosomes) {
        const auto values_end = std::next(values.get(), n_chromosomes);
        std::iota(values.get(), values_end, 0);
        std::shuffle(values.get(), values_end, gen);
        assert(!cached_cost_up_to_date); // ta metoda raczej zawsze będzie używana tuż po inicjalizacji osobnika, więc ta linijka nie powinna być potrzebna, ale jednak dla pewności daję assert'a, bo nie chcę ustawiać tej zmiennej na false za każdym razem
    }
};



#endif
