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
#include <ostream>
#include <iostream>
#include <sstream>

namespace tsp {

using index_t = std::size_t;
using distance_t = int; // nie double! zgodność z TSPLIB aby móc opierać się na wynikach z niej
// moze zmienic na tsp_lib_distance_t?
// trzeba zobaczyć co z problemami z innymi jednostkami (np floating-point)
using point_t = std::pair<int, int>;
using city_index_t = index_t;

static constexpr bool CACHE_THE_COST = true;
static constexpr bool DONT_CACHE_THE_COST = false;

namespace util {
    distance_t euclidean_distance2(const point_t& a, const point_t& b) {
        return static_cast<distance_t>(std::round(std::hypot(b.first - a.first, b.second - a.second))); // kompatybilne z TSPLIB (stąd round)
    }

    distance_t euclidean_distance(const point_t& pa, const point_t& pb) {
        const auto d1 = pb.first - pa.first;
        const auto d2 = pb.second - pa.second;
        const auto a = d1 * d1;
        const auto b = d2 * d2;
        return static_cast<distance_t>(std::sqrt(static_cast<double>(a + b)) + 0.5);
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
    TSP_Graph(std::size_t n_cities, const char* filename) // defaultowo inty
    : distances(n_cities)
    , n{n_cities} {
        std::ifstream f(filename);
        auto coords = std::make_unique<point_t[]>(n);
        std::string line;

        for (index_t i = 0; std::getline(f, line); ++i) { //todo: kompatybilnosc ze zmiennoprzecinkowymi wartosciami w pliku danych
            int x{}, y{};
            int multiplier = 1;
            const char* p = line.c_str() + line.length() - 1; // line.end() - 1
        
            while (*p != ' ') {
                y += (*p - '0') * multiplier;
                multiplier *= 10;
                --p;
            }
        
            while (*p == ' ') {
                --p;
            }
        
            multiplier = 1;
        
            while (*p != ' ') {
                x += (*p - '0') * multiplier;
                multiplier *= 10;
                --p;
            }
        
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
        for (index_t i = 0; i < n; ++i) {
            for (index_t j = 0; j < n; ++j) {
                os << distances.at(i, j) << ' ';
            }
            os << '\n';
        }
        os << tail;
    }
};

namespace detail {
    template <bool enable>
    struct cache_t {
        bool up_to_date = false;
        distance_t cost;
    };

    template <>
    struct cache_t<false> {
    };
}

template <typename T, typename Derived, bool should_cache = CACHE_THE_COST>
class base_TSP_solution_set {
protected:
    using value_type = T;
    std::unique_ptr<value_type[]> values;
    // musi być mutable, bo total_cost powinen być const metodą
    mutable detail::cache_t<should_cache> cache;

    base_TSP_solution_set(std::size_t n_chromosomes)
    : values(std::make_unique<value_type[]>(n_chromosomes)) {
    }

    base_TSP_solution_set(base_TSP_solution_set&& other)
    : values(std::move(other.values)) {
    }

    base_TSP_solution_set(const base_TSP_solution_set&) = delete; // uzywac write_copy_of
    base_TSP_solution_set& operator=(const base_TSP_solution_set&) = delete;

    base_TSP_solution_set& operator=(base_TSP_solution_set&& other) {
        if constexpr (should_cache) {
            cache.up_to_date = false;
        }
        values = std::move(other.values);
        return *this;
    }

public:
    distance_t total_cost(const TSP_Graph& graph) const {
        static_assert(
            std::is_same_v<
                distance_t,
                decltype(std::declval<const Derived&>()._compute_cost(std::declval<const TSP_Graph&>()))
            >,
            "Derived class must implement: distance_t _compute_cost(const TSP_Graph&) const" // metoda _compute_cost nie powinna być wywoływana nigdzie poza tą metodą
        );

        if constexpr (should_cache) {
            if (!cache.up_to_date) {
                cache.cost = static_cast<const Derived&>(*this)._compute_cost(graph);
                cache.up_to_date = true;
            }

            return cache.cost;
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
            cache.up_to_date = false;
        }
        std::copy_n(other.get_values_raw_ptr(), n_chromosomes, values.get());
    }

    const value_type& at(index_t i) const noexcept {
        // niestety nie mozemy sprawdzić czy index jest in-bounds - wywolujacy musi to zapewnić, inaczej UB. tak samo w set()
        return values[i];
    }

    void set(index_t i, value_type t) noexcept {
        if constexpr (should_cache) {
            cache.up_to_date = false;
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

template <bool should_cache>
class t_TSP_solution_set : public base_TSP_solution_set<city_index_t, t_TSP_solution_set<should_cache>, should_cache> {

    using _my_base = base_TSP_solution_set<city_index_t, t_TSP_solution_set<should_cache>, should_cache>;
    using _my_base::values;

public:
    using _my_base::value_type;

    t_TSP_solution_set(std::size_t n) : _my_base(n) {
    }

    distance_t _compute_cost(const TSP_Graph& graph) const {
        distance_t out{};
        const auto n = graph.n_cities();

        for (index_t i = 1; i < n; ++i) {
            out += graph.distance(values[i - 1], values[i]);
        }

        out += graph.distance(values[n - 1], values[0]); // powrót do miasta startowego

        return out;
    }

    void generate_random(std::mt19937& gen, std::size_t n_chromosomes) {
        const auto values_end = std::next(values.get(), n_chromosomes);

        std::iota(values.get(), values_end, 0);
        std::shuffle(values.get(), values_end, gen);

        if constexpr (should_cache) {
            assert(!_my_base::cache.up_to_date); // ta metoda raczej zawsze będzie używana tuż po inicjalizacji osobnika, więc ta linijka nie powinna być potrzebna, ale jednak dla pewności daję assert'a, bo nie chcę ustawiać tej zmiennej na false za każdym razem
        }
    }
};

using TSP_solution_set = t_TSP_solution_set<CACHE_THE_COST>;
using TSP_solution_set_no_caching = t_TSP_solution_set<DONT_CACHE_THE_COST>;

} // namespace tsp

#endif // ifndef TSP_HPP
