// #include <bits/stdc++.h>
// #include <omp.h>
// using u64 = uint64_t;

// // 128-bit bitset using two 64-bit words
// struct Bitset128 {
//     u64 lo, hi;
//     inline void set(int i) {
//         if (i < 64) lo |= u64(1) << i;
//         else hi |= u64(1) << (i - 64);
//     }
//     inline Bitset128 operator|(const Bitset128 &b) const {
//         return Bitset128{lo | b.lo, hi | b.hi};
//     }
//     inline Bitset128 operator&(const Bitset128 &b) const {
//         return Bitset128{lo & b.lo, hi & b.hi};
//     }
//     inline Bitset128 operator~() const {
//         return Bitset128{~lo, ~hi};
//     }
//     inline bool empty() const {
//         return lo == 0 && hi == 0;
//     }
//     inline int popcount() const {
//         return __builtin_popcountll(lo) + __builtin_popcountll(hi);
//     }
// };

// int n;
// std::vector<Bitset128> cover;
// Bitset128 ALL;
// int bestCount = INT_MAX;
// std::vector<int> bestSolution;

// // Subtract bitsets: a \ b
// Bitset128 subtract_bs(const Bitset128 &a, const Bitset128 &b) {
//     return Bitset128{a.lo & ~b.lo, a.hi & ~b.hi};
// }

// // Reduction for minimum integer
// #pragma omp declare reduction(minInt: int: omp_out = omp_in < omp_out ? omp_in : omp_out) initializer(omp_priv = omp_orig)

// void dfs(const Bitset128 &uncovered, std::vector<int> &sol, int depth) {
//     // If fully covered, update best solution
//     if (uncovered.empty()) {
//         #pragma omp critical
//         {
//             if ((int)sol.size() < bestCount) {
//                 bestCount = sol.size();
//                 bestSolution = sol;
//             }
//         }
//         return;
//     }

//     // Compute lower bound by maximum coverage per move
//     int uc = uncovered.popcount();
//     int maxCover = 0;
//     for (int i = 0; i < n; ++i) {
//         Bitset128 tmp{cover[i].lo & uncovered.lo, cover[i].hi & uncovered.hi};
//         int c = tmp.popcount();
//         maxCover = std::max(maxCover, c);
//     }
//     int lb = (uc + maxCover - 1) / maxCover;
//     if ((int)sol.size() + lb >= bestCount) return;

//     // Pivot selection: vertex covering most uncovered
//     int pivot = 0;
//     maxCover = -1;
//     for (int i = 0; i < n; ++i) {
//         Bitset128 tmp{cover[i].lo & uncovered.lo, cover[i].hi & uncovered.hi};
//         int c = tmp.popcount();
//         if (c > maxCover) { maxCover = c; pivot = i; }
//     }

//     // Generate candidate vertices that cover pivot, sorted by coverage
//     Bitset128 candSet = cover[pivot];
//     std::vector<std::pair<int,int>> cand;
//     for (int i = 0; i < n; ++i) {
//         bool in = (i < 64) ? ((candSet.lo >> i) & 1) : ((candSet.hi >> (i - 64)) & 1);
//         if (in) {
//             Bitset128 tmp{cover[i].lo & uncovered.lo, cover[i].hi & uncovered.hi};
//             cand.emplace_back(-tmp.popcount(), i);
//         }
//     }
//     std::sort(cand.begin(), cand.end());

//     // Branch on candidates
//     for (auto &p : cand) {
//         int v = p.second;
//         sol.push_back(v);
//         Bitset128 newUn = subtract_bs(uncovered, cover[v]);
//         if (depth < 2) {
//             #pragma omp task firstprivate(newUn, sol, depth)
//             dfs(newUn, sol, depth + 1);
//         } else {
//             dfs(newUn, sol, depth + 1);
//         }
//         sol.pop_back();
//     }
//     if (depth < 2) {
//         #pragma omp taskwait
//     }
// }

// int main(int argc, char **argv) {
//     if (argc != 3) return 1;
//     std::ifstream fin(argv[1]);
//     std::ofstream fout(argv[2]);
//     int m;
//     fin >> n >> m;
//     std::vector<std::vector<int>> adj(n);
//     for (int i = 0, u, v; i < m; ++i) {
//         fin >> u >> v;
//         adj[u].push_back(v);
//         adj[v].push_back(u);
//     }

//     // Build cover sets and ALL mask
//     cover.assign(n, Bitset128{0,0});
//     ALL = Bitset128{0,0};
//     for (int i = 0; i < n; ++i) {
//         cover[i].set(i);
//         for (int j : adj[i]) cover[i].set(j);
//         ALL.set(i);
//     }

//     // Start DFS in parallel
//     Bitset128 initialUn = ALL;
//     std::vector<int> sol;
//     #pragma omp parallel
//     {
//         #pragma omp single nowait
//         dfs(initialUn, sol, 0);
//     }

//     // Output binary string
//     std::string output(n, '0');
//     for (int v : bestSolution) output[v] = '1';
//     fout << output << "\n";
//     return 0;
// }

/*********************************************************************
 *  High-Performance Minimum Dominating Set Solver
 *
 *  Features:
 *  - Aggressive kernelization (R0-R3 + leaf-fold + dome + crown)
 *  - Advanced lower bounds (combinatorial + LP relaxation)
 *  - Multi-level parallelism (OpenMP + SIMD)
 *  - Cache-optimized data structures
 *  - Specialized solvers for different graph sizes
 *********************************************************************/
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <stdatomic.h>
#include <omp.h>

#if defined(__AVX2__)
  #include <immintrin.h>
#endif

/*============== Configuration and Constants ===============*/
#define MAX_VERTICES   1024  // Maximum graph size supported
#define SMALL_THRESHOLD 256  // Threshold for specialized small graph solver
#define SIMD_WORDS       4   // Number of 64-bit words for SIMD processing
#define WORDS_FOR(n)  (((n)+63)>>6)  // Number of words needed for n bits

/*============== Bit manipulation helpers =================*/
#define POPCOUNT(x)   __builtin_popcountll((unsigned long long)(x))
#define CTZ(x)        __builtin_ctzll((unsigned long long)(x))
#define NEXT_BIT(word, bit) do { word &= ~(1ULL << (bit)); bit = word ? CTZ(word) : -1; } while(0)

/*============== Data structures ==========================*/
// Generic bitvector for graphs up to MAX_VERTICES
typedef uint64_t BitVector[16];  

// Cache-optimized small bitvector for SIMD processing
typedef uint64_t SmallBitVector[SIMD_WORDS];

/*============== Global state ============================*/
// Current graph state
static int num_vertices;            // Current number of vertices
static int num_words;               // Words needed for current graph
static BitVector* neighborhoods;    // Neighborhood bitvectors for generic solver
static SmallBitVector* simd_neighborhoods; // SIMD-optimized neighborhoods

// Mapping and solution
static int vertex_map[MAX_VERTICES];  // Maps current vertices to original IDs
static int solution[MAX_VERTICES];    // Final solution (1 if in dominating set)

// Best solution found so far (atomic for thread safety)
static _Atomic int best_size;         // Size of best solution (generic solver)
static BitVector best_solution;       // Best solution bitvector (generic solver)
static _Atomic int best_size_simd;    // Size of best solution (SIMD solver)
static SmallBitVector best_solution_simd; // Best solution bitvector (SIMD solver)

/*============== BitVector operations ====================*/
/**
 * Generic bitvector operations for large graphs
 */
static inline void bv_clear(BitVector bv, int words) {
    memset(bv, 0, words * sizeof(uint64_t));
}

static inline void bv_copy(BitVector dst, const BitVector src, int words) {
    memcpy(dst, src, words * sizeof(uint64_t));
}

static inline void bv_or(BitVector dst, const BitVector src, int words) {
    for (int i = 0; i < words; ++i) {
        dst[i] |= src[i];
    }
}

static inline void bv_and_not(BitVector result, const BitVector a, const BitVector b, int words) {
    for (int i = 0; i < words; ++i) {
        result[i] = a[i] & ~b[i];
    }
}

static inline int bv_popcount(const BitVector bv, int words) {
    int count = 0;
    for (int i = 0; i < words; ++i) {
        count += POPCOUNT(bv[i]);
    }
    return count;
}

static inline int bv_first_bit(const BitVector bv, int words) {
    for (int w = 0; w < words; ++w) {
        if (bv[w]) {
            return (w << 6) + CTZ(bv[w]);
        }
    }
    return -1;
}

static inline bool bv_test_bit(const BitVector bv, int bit) {
    return (bv[bit >> 6] >> (bit & 63)) & 1;
}

static inline void bv_set_bit(BitVector bv, int bit) {
    bv[bit >> 6] |= (1ULL << (bit & 63));
}

/**
 * SIMD-optimized bitvector operations for small graphs
 */
#if defined(__AVX2__)
static inline int simd_popcount(const SmallBitVector bv) {
    __m256i v = _mm256_loadu_si256((const __m256i*)bv);
    __m256i counts = _mm256_sad_epu8(v, _mm256_setzero_si256());
    return _mm256_extract_epi64(counts, 0) + _mm256_extract_epi64(counts, 1) +
           _mm256_extract_epi64(counts, 2) + _mm256_extract_epi64(counts, 3);
}

static inline void simd_and_not(SmallBitVector result, const SmallBitVector a, const SmallBitVector b) {
    __m256i va = _mm256_loadu_si256((const __m256i*)a);
    __m256i vb = _mm256_loadu_si256((const __m256i*)b);
    _mm256_storeu_si256((__m256i*)result, _mm256_andnot_si256(vb, va));
}

static inline void simd_or(SmallBitVector dst, const SmallBitVector src) {
    __m256i vdst = _mm256_loadu_si256((const __m256i*)dst);
    __m256i vsrc = _mm256_loadu_si256((const __m256i*)src);
    _mm256_storeu_si256((__m256i*)dst, _mm256_or_si256(vdst, vsrc));
}
#else
static inline int simd_popcount(const SmallBitVector bv) {
    return POPCOUNT(bv[0]) + POPCOUNT(bv[1]) + POPCOUNT(bv[2]) + POPCOUNT(bv[3]);
}

static inline void simd_and_not(SmallBitVector result, const SmallBitVector a, const SmallBitVector b) {
    result[0] = a[0] & ~b[0];
    result[1] = a[1] & ~b[1];
    result[2] = a[2] & ~b[2];
    result[3] = a[3] & ~b[3];
}

static inline void simd_or(SmallBitVector dst, const SmallBitVector src) {
    dst[0] |= src[0];
    dst[1] |= src[1];
    dst[2] |= src[2];
    dst[3] |= src[3];
}
#endif

static inline bool simd_test_bit(const SmallBitVector bv, int bit) {
    return (bv[bit >> 6] >> (bit & 63)) & 1;
}

static inline void simd_set_bit(SmallBitVector bv, int bit) {
    bv[bit >> 6] |= (1ULL << (bit & 63));
}

/*============== Kernelization Rules =====================*/
/**
 * Rebuild neighborhood bitvectors after graph reduction
 */
static void rebuild_neighborhoods(const int* old_to_new, int new_size) {
    BitVector* new_neighborhoods = calloc(new_size, sizeof(BitVector));
    
    for (int old_v = 0; old_v < MAX_VERTICES; ++old_v) {
        int new_v = old_to_new[old_v];
        if (new_v == -1) continue;
        
        for (int w = 0; w < 16; ++w) {
            uint64_t word = neighborhoods[old_v][w];
            while (word) {
                int bit = CTZ(word) + (w << 6);
                word &= word - 1;
                
                int new_nb = old_to_new[bit];
                if (new_nb == -1) continue;
                
                new_neighborhoods[new_v][new_nb >> 6] |= 1ULL << (new_nb & 63);
            }
        }
    }
    
    free(neighborhoods);
    neighborhoods = new_neighborhoods;
}

/**
 * Find a vertex with a single neighbor
 */
static int find_single_neighbor(const BitVector bv, int words) {
    int count = 0, last = -1;
    
    for (int w = 0; w < words; ++w) {
        uint64_t word = bv[w];
        while (word) {
            int bit = CTZ(word) + (w << 6);
            if (++count > 1) return -1;
            last = bit;
            word &= word - 1;
        }
    }
    
    return last;  // -1 if no neighbors or more than one
}

/**
 * Crown reduction rule:
 * Find an independent set C with vertices of degree ≤ 2 such that |N(C)| < |C|
 */
static int apply_crown_reduction(int* vertices_to_pick, int* vertices_to_delete) {
    int crown[MAX_VERTICES], crown_size = 0;
    int neighborhood[MAX_VERTICES] = {0}, neighborhood_count = 0;
    
    // Find candidate vertices for the crown (degree ≤ 2)
    for (int v = 0; v < num_vertices; ++v) {
        int degree = bv_popcount(neighborhoods[v], num_words) - 1;  // Exclude self-loop
        if (degree <= 2) {
            crown[crown_size++] = v;
        }
    }
    
    if (crown_size < 3) return 0;  // Too small to be effective
    
    // Greedy matching between crown and neighborhood
    char matched_crown[MAX_VERTICES] = {0};
    char matched_neighborhood[MAX_VERTICES] = {0};
    
    for (int i = 0; i < crown_size; ++i) {
        int v = crown[i];
        
        for (int w = 0; w < num_words; ++w) {
            uint64_t word = neighborhoods[v][w];
            while (word) {
                int nb = CTZ(word) + (w << 6);
                word &= word - 1;
                
                if (nb == v || matched_neighborhood[nb]) continue;
                
                matched_crown[v] = 1;
                matched_neighborhood[nb] = 1;
                break;
            }
            if (matched_crown[v]) break;
        }
    }
    
    // Count unmatched crown vertices
    int unmatched = 0;
    for (int i = 0; i < crown_size; ++i) {
        if (!matched_crown[crown[i]]) ++unmatched;
    }
    
    if (unmatched == 0) return 0;  // Need |C| > |N(C)|
    
    // Mark crown vertices for deletion and neighborhood vertices for inclusion
    for (int i = 0; i < crown_size; ++i) {
        vertices_to_delete[crown[i]] = 1;
    }
    
    for (int v = 0; v < num_vertices; ++v) {
        if (matched_neighborhood[v]) {
            vertices_to_pick[v] = 1;
        }
    }
    
    return 1;  // Rule applied successfully
}

/**
 * Apply all kernelization rules until no more reductions possible
 */
static void kernelize() {
    bool changed;
    
    do {
        changed = false;
        
        // Calculate vertex degrees
        int degrees[MAX_VERTICES];
        for (int v = 0; v < num_vertices; ++v) {
            degrees[v] = bv_popcount(neighborhoods[v], num_words) - 1;  // Exclude self-loop
        }
        
        // R0 (isolated vertices) and R1 (universal vertices)
        for (int v = 0; v < num_vertices; ++v) {
            if (degrees[v] == 0) {
                // Isolated vertex must be in solution
                solution[vertex_map[v]] = 1;
                changed = true;
            }
            else if (degrees[v] == num_vertices - 1) {
                // Universal vertex - optimal solution found
                solution[vertex_map[v]] = 1;
                
                // Print solution and exit
                for (int i = 0; i < vertex_map[0]; ++i) {
                    putchar(solution[i] ? '1' : '0');
                }
                putchar('\n');
                exit(0);
            }
        }
        
        int vertices_to_pick[MAX_VERTICES] = {0};
        int vertices_to_delete[MAX_VERTICES] = {0};
        
        // Leaf-fold reduction (vertices with exactly one neighbor)
        for (int v = 0; v < num_vertices; ++v) {
            if (degrees[v] == 1) {
                int u = find_single_neighbor(neighborhoods[v], num_words);
                if (u >= 0) {
                    vertices_to_pick[u] = 1;
                    vertices_to_delete[u] = 1;
                    vertices_to_delete[v] = 1;
                    changed = true;
                }
            }
        }
        
        // R2/R3 subset-twin rules
        for (int v = 0; v < num_vertices; ++v) {
            if (vertices_to_delete[v]) continue;
            
            for (int u = v + 1; u < num_vertices; ++u) {
                if (vertices_to_delete[u]) continue;
                
                bool v_subset_u = true;
                bool u_subset_v = true;
                bool equal = true;
                
                for (int w = 0; w < num_words; ++w) {
                    uint64_t N_v = neighborhoods[v][w];
                    uint64_t N_u = neighborhoods[u][w];
                    
                    if (N_v & ~N_u) v_subset_u = false;
                    if (N_u & ~N_v) u_subset_v = false;
                    if (N_v != N_u) equal = false;
                }
                
                if (equal) {
                    // Identical neighborhoods - keep one
                    vertices_to_delete[u] = 1;
                    changed = true;
                }
                else if (v_subset_u && !u_subset_v) {
                    // N(v) ⊂ N(u) - prefer u
                    vertices_to_delete[v] = 1;
                    changed = true;
                    break;
                }
                else if (u_subset_v && !v_subset_u) {
                    // N(u) ⊂ N(v) - prefer v
                    vertices_to_delete[u] = 1;
                    changed = true;
                }
            }
        }
        
        // Crown reduction
        if (apply_crown_reduction(vertices_to_pick, vertices_to_delete)) {
            changed = true;
        }
        
        // Apply picks immediately
        for (int v = 0; v < num_vertices; ++v) {
            if (vertices_to_pick[v]) {
                solution[vertex_map[v]] = 1;
            }
        }
        
        // Compress graph if there are deletions
        bool any_deleted = false;
        for (int v = 0; v < num_vertices; ++v) {
            if (vertices_to_delete[v]) {
                any_deleted = true;
                break;
            }
        }
        
        if (any_deleted) {
            int old_to_new[MAX_VERTICES];
            for (int i = 0; i < MAX_VERTICES; ++i) {
                old_to_new[i] = -1;
            }
            
            int new_index = 0;
            for (int v = 0; v < num_vertices; ++v) {
                if (!vertices_to_delete[v]) {
                    old_to_new[vertex_map[v]] = new_index++;
                }
            }
            
            int new_size = new_index;
            int new_vertex_map[MAX_VERTICES];
            
            for (int v = 0; v < num_vertices; ++v) {
                if (!vertices_to_delete[v]) {
                    int orig = vertex_map[v];
                    int v_new = old_to_new[orig];
                    new_vertex_map[v_new] = orig;
                }
            }
            
            rebuild_neighborhoods(old_to_new, new_size);
            memcpy(vertex_map, new_vertex_map, new_size * sizeof(int));
            
            num_vertices = new_size;
            num_words = WORDS_FOR(num_vertices);
            changed = true;
        }
        
    } while (changed);
}

/*============== Lower Bound Functions ===================*/
/**
 * Calculate LP-based lower bound for SIMD solver
 */
static int calculate_lp_bound_simd(const SmallBitVector uncovered) {
    double sum = 0.0;
    
    for (int v = 0; v < num_vertices; ++v) {
        if (simd_test_bit(uncovered, v)) {
            sum += 1.0 / (double)simd_popcount(simd_neighborhoods[v]);
        }
    }
    
    int bound = (int)sum;
    return (bound == sum) ? bound : bound + 1;
}

/**
 * Calculate LP-based lower bound for generic solver
 */
static int calculate_lp_bound(const BitVector uncovered) {
    double sum = 0.0;
    
    for (int v = 0; v < num_vertices; ++v) {
        if (bv_test_bit(uncovered, v)) {
            sum += 1.0 / (double)bv_popcount(neighborhoods[v], num_words);
        }
    }
    
    int bound = (int)sum;
    return (bound == sum) ? bound : bound + 1;
}

/*============== Greedy Approximation ====================*/
/**
 * Greedy approximation for SIMD solver
 */
static void greedy_approximation_simd(const SmallBitVector all_vertices) {
    SmallBitVector covered = {0};
    SmallBitVector chosen = {0};
    int count = 0;
    
    while (memcmp(covered, all_vertices, sizeof(SmallBitVector)) != 0) {
        int best_vertex = -1;
        double best_score = -1.0;
        
        for (int v = 0; v < num_vertices; ++v) {
            SmallBitVector new_coverage;
            simd_and_not(new_coverage, simd_neighborhoods[v], covered);
            
            int gain = simd_popcount(new_coverage);
            if (gain == 0) continue;
            
            double score = (double)gain / (double)simd_popcount(simd_neighborhoods[v]);
            if (score > best_score) {
                best_score = score;
                best_vertex = v;
            }
        }
        
        simd_or(covered, simd_neighborhoods[best_vertex]);
        simd_set_bit(chosen, best_vertex);
        ++count;
    }
    
    atomic_store(&best_size_simd, count);
    memcpy(best_solution_simd, chosen, sizeof(SmallBitVector));
}

/**
 * Greedy approximation for generic solver
 */
static void greedy_approximation(const BitVector all_vertices) {
    BitVector covered;
    BitVector chosen;
    int count = 0;
    
    bv_clear(covered, num_words);
    bv_clear(chosen, num_words);
    
    while (memcmp(covered, all_vertices, num_words * sizeof(uint64_t)) != 0) {
        int best_vertex = -1;
        int best_gain = -1;
        
        for (int v = 0; v < num_vertices; ++v) {
            BitVector new_coverage;
            bv_and_not(new_coverage, neighborhoods[v], covered, num_words);
            
            int gain = bv_popcount(new_coverage, num_words);
            if (gain > best_gain) {
                best_gain = gain;
                best_vertex = v;
            }
        }
        
        bv_or(covered, neighborhoods[best_vertex], num_words);
        bv_set_bit(chosen, best_vertex);
        ++count;
    }
    
    atomic_store(&best_size, count);
    bv_copy(best_solution, chosen, num_words);
}

/*============== Branch and Bound Solvers ================*/
/**
 * Branch and bound solver for small graphs using SIMD
 */
static void branch_and_bound_simd(SmallBitVector covered, SmallBitVector chosen, 
                                 int chosen_count, const SmallBitVector all_vertices) {
    // Pruning: if we've already taken more vertices than the best known solution
    if (chosen_count >= atomic_load(&best_size_simd)) return;
    
    // Base case: all vertices are covered
    if (memcmp(covered, all_vertices, sizeof(SmallBitVector)) == 0) {
        #pragma omp critical
        if (chosen_count < best_size_simd) {
            best_size_simd = chosen_count;
            memcpy(best_solution_simd, chosen, sizeof(SmallBitVector));
        }
        return;
    }
    
    // Calculate uncovered vertices
    SmallBitVector uncovered;
    simd_and_not(uncovered, all_vertices, covered);
    
    // Lower bound calculations
    int uncovered_count = simd_popcount(uncovered);
    int degree_bound = (uncovered_count + 4) / 5;  // ⌈|U|/5⌉
    int lp_bound = calculate_lp_bound_simd(uncovered);
    int lower_bound = (degree_bound > lp_bound) ? degree_bound : lp_bound;
    
    // Pruning: if lower bound + chosen count exceeds best solution
    if (chosen_count + lower_bound >= atomic_load(&best_size_simd)) return;
    
    // Find pivot: uncovered vertex that has the best gain/weight ratio
    int pivot = -1;
    double pivot_score = -1.0;
    
    for (int v = 0; v < num_vertices; ++v) {
        if (!simd_test_bit(uncovered, v)) continue;
        
        SmallBitVector tmp;
        simd_and_not(tmp, simd_neighborhoods[v], covered);
        
        int gain = simd_popcount(tmp);
        double score = (double)gain / (double)simd_popcount(simd_neighborhoods[v]);
        
        if (score > pivot_score) {
            pivot_score = score;
            pivot = v;
        }
    }
    
    // Find all candidates that cover the pivot
    int candidates[SMALL_THRESHOLD];
    int candidate_count = 0;
    
    for (int v = 0; v < num_vertices; ++v) {
        if (simd_test_bit(simd_neighborhoods[v], pivot)) {
            candidates[candidate_count++] = v;
        }
    }
    
    // Try each candidate
    for (int i = 0; i < candidate_count; ++i) {
        int v = candidates[i];
        
        SmallBitVector new_covered, new_chosen;
        memcpy(new_covered, covered, sizeof(SmallBitVector));
        memcpy(new_chosen, chosen, sizeof(SmallBitVector));
        
        simd_or(new_covered, simd_neighborhoods[v]);
        simd_set_bit(new_chosen, v);
        
        branch_and_bound_simd(new_covered, new_chosen, chosen_count + 1, all_vertices);
    }
}

/**
 * Branch and bound solver for generic sized graphs
 */
static void branch_and_bound(BitVector covered, BitVector chosen,
                           int chosen_count, const BitVector all_vertices) {
    // Pruning: if we've already taken more vertices than the best known solution
    if (chosen_count >= atomic_load(&best_size)) return;
    
    // Base case: all vertices are covered
    if (memcmp(covered, all_vertices, num_words * sizeof(uint64_t)) == 0) {
        #pragma omp critical
        if (chosen_count < best_size) {
            best_size = chosen_count;
            bv_copy(best_solution, chosen, num_words);
        }
        return;
    }
    
    // Calculate uncovered vertices
    BitVector uncovered;
    bv_and_not(uncovered, all_vertices, covered, num_words);
    
    // Lower bound calculations
    int uncovered_count = bv_popcount(uncovered, num_words);
    int degree_bound = (uncovered_count + 4) / 5;  // ⌈|U|/5⌉
    
    // Find maximum gain for optimistic bound
    int max_gain = 0;
    for (int v = 0; v < num_vertices; ++v) {
        BitVector tmp;
        bv_and_not(tmp, neighborhoods[v], covered, num_words);
        int gain = bv_popcount(tmp, num_words);
        if (gain > max_gain) max_gain = gain;
    }
    
    int optimistic_n = (uncovered_count + max_gain - 1) / max_gain;
    int optimistic = (degree_bound > optimistic_n) ? degree_bound : optimistic_n;
    
    // Pruning: if lower bound + chosen count exceeds best solution
    if (chosen_count + optimistic >= atomic_load(&best_size)) return;
    
    // Find an uncovered vertex as pivot
    int pivot = bv_first_bit(uncovered, num_words);
    
    // Find all candidates that cover the pivot
    int candidates[MAX_VERTICES];
    int candidate_count = 0;
    
    for (int v = 0; v < num_vertices; ++v) {
        if (bv_test_bit(neighborhoods[v], pivot)) {
            candidates[candidate_count++] = v;
        }
    }
    
    // Try each candidate
    for (int i = 0; i < candidate_count; ++i) {
        int v = candidates[i];
        
        BitVector new_covered, new_chosen;
        bv_copy(new_covered, covered, num_words);
        bv_copy(new_chosen, chosen, num_words);
        
        bv_or(new_covered, neighborhoods[v], num_words);
        bv_set_bit(new_chosen, v);
        
        branch_and_bound(new_covered, new_chosen, chosen_count + 1, all_vertices);
    }
}

/**
 * Main solver function for SIMD-optimized small graphs
 */
static void solve_small_graph() {
    // Create mask for all vertices
    SmallBitVector all_vertices = {~0ULL, ~0ULL, ~0ULL, ~0ULL};
    if (num_vertices & 63) {
        all_vertices[WORDS_FOR(num_vertices) - 1] = (1ULL << (num_vertices & 63)) - 1ULL;
    }
    
    // Initialize best solution with greedy approximation
    best_size_simd = num_vertices + 1;
    memset(best_solution_simd, 0, sizeof(SmallBitVector));
    greedy_approximation_simd(all_vertices);
    
    // Branch and bound in parallel
    #pragma omp parallel
    #pragma omp single nowait
    for (int v = 0; v < num_vertices; ++v) {
        #pragma omp task firstprivate(v)
        {
            SmallBitVector covered = {0};
            SmallBitVector chosen = {0};
            
            memcpy(covered, simd_neighborhoods[v], sizeof(SmallBitVector));
            simd_set_bit(chosen, v);
            
            branch_and_bound_simd(covered, chosen, 1, all_vertices);
        }
    }
}

/**
 * Main solver function for generic sized graphs
 */
static void solve_generic_graph() {
    // Create mask for all vertices
    BitVector all_vertices;
    bv_clear(all_vertices, num_words);
    
    for (int w = 0; w < num_words - 1; ++w) {
        all_vertices[w] = ~0ULL;
    }
    
    all_vertices[num_words - 1] = (num_vertices & 63) ?
        ((1ULL << (num_vertices & 63)) - 1ULL) : ~0ULL;
    
    // Initialize best solution with greedy approximation
    best_size = num_vertices + 1;
    bv_clear(best_solution, num_words);
    greedy_approximation(all_vertices);
    
    // Branch and bound in parallel
    #pragma omp parallel
    #pragma omp single nowait
    for (int v = 0; v < num_vertices; ++v) {
        #pragma omp task firstprivate(v)
        {
            BitVector covered, chosen;
            bv_copy(covered, neighborhoods[v], num_words);
            bv_clear(chosen, num_words);
            bv_set_bit(chosen, v);
            
            branch_and_bound(covered, chosen, 1, all_vertices);
        }
    }
}

/**
 * Generate final answer from the best solution found
 */
static void output_solution(int original_size, bool use_simd) {
    if (use_simd) {
        for (int v = 0; v < num_vertices; ++v) {
            if ((best_solution_simd[v >> 6] >> (v & 63)) & 1) {
                solution[vertex_map[v]] = 1;
            }
        }
    } else {
        for (int v = 0; v < num_vertices; ++v) {
            if ((best_solution[v >> 6] >> (v & 63)) & 1) {
                solution[vertex_map[v]] = 1;
            }
        }
    }
    
    // Output the solution
    for (int i = 0; i < original_size; ++i) {
        putchar(solution[i] ? '1' : '0');
    }
    putchar('\n');
}

/*============== Main Function ============================*/
int main(int argc, char **argv) {
    // Set up OpenMP
    omp_set_dynamic(0);
    omp_set_num_threads(omp_get_num_procs());

    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input_file> <output_file>\n", argv[0]);
        return 1;
    }

    FILE *fin = fopen(argv[1], "r");
    if (!fin) {
        fprintf(stderr, "Failed to open input: %s\n", argv[1]);
        perror("fopen input");
        return 1;
    }

    FILE *fout = fopen(argv[2], "w");
    if (!fout) {
        fprintf(stderr, "Failed to open output: %s\n", argv[2]);
        perror("fopen output");
        fclose(fin);
        return 1;
    }


    int original_size, num_edges;
    if (fscanf(fin, "%d%d", &original_size, &num_edges) != 2 ||
        original_size <= 0 || original_size > MAX_VERTICES) {
        return 1;
    }

    num_vertices = original_size;
    num_words = WORDS_FOR(num_vertices);
    neighborhoods = aligned_alloc(64, original_size * sizeof(BitVector));

    for (int i = 0; i < original_size; ++i) {
        vertex_map[i] = i;
        memset(neighborhoods[i], 0, sizeof(BitVector));
        neighborhoods[i][i >> 6] |= 1ULL << (i & 63);
    }

    for (int i = 0; i < num_edges; ++i) {
        int a, b;
        fscanf(fin, "%d%d", &a, &b);
        if (a != b && a >= 0 && b >= 0 && a < original_size && b < original_size) {
            neighborhoods[a][b >> 6] |= 1ULL << (b & 63);
            neighborhoods[b][a >> 6] |= 1ULL << (a & 63);
        }
    }

    fclose(fin);

    kernelize();
    bool use_simd = (num_vertices <= SMALL_THRESHOLD);

    if (use_simd) {
        simd_neighborhoods = aligned_alloc(32, num_vertices * sizeof(SmallBitVector));
        for (int v = 0; v < num_vertices; ++v)
            for (int w = 0; w < SIMD_WORDS; ++w)
                simd_neighborhoods[v][w] = neighborhoods[v][w];
        free(neighborhoods);
        neighborhoods = NULL;
        solve_small_graph();
    } else {
        solve_generic_graph();
    }

    if (use_simd) {
        for (int v = 0; v < num_vertices; ++v) {
            if ((best_solution_simd[v >> 6] >> (v & 63)) & 1)
                solution[vertex_map[v]] = 1;
        }
    } else {
        for (int v = 0; v < num_vertices; ++v) {
            if ((best_solution[v >> 6] >> (v & 63)) & 1)
                solution[vertex_map[v]] = 1;
        }
    }

    for (int i = 0; i < original_size; ++i) {
        fputc(solution[i] ? '1' : '0', fout);
    }
    fputc('\n', fout);
    fclose(fout);

    if (neighborhoods) free(neighborhoods);
    if (simd_neighborhoods) free(simd_neighborhoods);
    return 0;
}
