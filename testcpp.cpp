// powerplant.cpp
// Highly‑tuned Minimum Dominating Set with greedy bound, atomic prune, kernelization, OpenMP

#include <bits/stdc++.h>
#include <omp.h>
using u64 = uint64_t;

// 128‑bit bitset
struct Bitset128 {
    u64 lo=0, hi=0;
    inline void set(int i) {
        if (i<64) lo |= u64(1)<<i;
        else       hi |= u64(1)<<(i-64);
    }
    inline bool full(int n) const {
        if (n<=64) return lo == (~u64(0)>>(64-n));
        return lo==~u64(0) && hi==(~u64(0)>>(128-n));
    }
};
// scalar OR is cheapest on two words
static inline Bitset128 bs_union(const Bitset128 &a, const Bitset128 &b) {
    return Bitset128{ a.lo|b.lo, a.hi|b.hi };
}

int n;
std::vector<Bitset128> coverBS;
std::vector<std::vector<int>> adj;
std::atomic<int> best;
std::vector<int> bestSet;
omp_lock_t set_lock;

// Greedy 2‑approximation to seed `best` and `bestSet`
void greedy_seed() {
    Bitset128 dom{};
    std::vector<int> sol;
    while (!dom.full(n)) {
        int pick=-1, maxcov=0;
        // pick vertex covering most **new** nodes
        for (int u=0;u<n;++u) {
            u64 new_lo = coverBS[u].lo & ~dom.lo;
            u64 new_hi = coverBS[u].hi & ~dom.hi;
            int c = __builtin_popcountll(new_lo) + __builtin_popcountll(new_hi);
            if (c>maxcov) { maxcov=c; pick=u; }
        }
        sol.push_back(pick);
        dom.lo |= coverBS[pick].lo;
        dom.hi |= coverBS[pick].hi;
    }
    best.store((int)sol.size(), std::memory_order_relaxed);
    omp_set_lock(&set_lock);
      bestSet = sol;
    omp_unset_lock(&set_lock);
}

// Recursive DFS with branch‑&‑bound, kernelization on degree‑1
void dfs(const Bitset128 &dom, int cnt, std::vector<int> &cur) {
    // fast prune
    int ub = best.load(std::memory_order_relaxed);
    if (cnt >= ub) return;

    // if fully covered, update
    if (dom.full(n)) {
        std::cerr << "[DEBUG] Found new solution with size: " << cnt << "\n";
        // try update best
        while (cnt < ub) {
            if (best.compare_exchange_weak(ub, cnt, std::memory_order_relaxed)) {
                omp_set_lock(&set_lock);
                  bestSet = cur;
                omp_unset_lock(&set_lock);
                break;
            }
        }
        return;
    }

    // kernelization: look for any undominated vertex of degree 1 in the **remaining** subgraph
    for (int u=0; u<n; ++u) {
        bool isDom = (u<64 ? ((dom.lo>>u)&1) : ((dom.hi>>(u-64))&1));
        if (!isDom) {
            // count undominated neighbors
            int deg1count = 0, ngh = -1;
            for (int v: adj[u]) {
                bool vDom = (v<64?((dom.lo>>v)&1):((dom.hi>>(v-64))&1));
                if (!vDom) { if (++deg1count>1) break; ngh=v; }
            }
            if (deg1count==1) {
                // must pick `ngh`
                cur.push_back(ngh);
                Bitset128 nd = bs_union(dom, coverBS[ngh]);
                dfs(nd, cnt+1, cur);
                cur.pop_back();
                return;      // after applying a rule, stop and do not branch further here
            }
        }
    }

    // pick undominated u
    int u=0;
    for (;;++u) {
        bool isDom=(u<64?((dom.lo>>u)&1):((dom.hi>>(u-64))&1));
        if (!isDom) break;
    }

    // branch on u
    {
        cur.push_back(u);
        dfs(bs_union(dom, coverBS[u]), cnt+1, cur);
        cur.pop_back();
    }
    // branch on each neighbor v
    for (int v: adj[u]) {
        cur.push_back(v);
        dfs(bs_union(dom, coverBS[v]), cnt+1, cur);
        cur.pop_back();
    }
}

int main(int argc,char**argv){
    if(argc<3){ std::cerr<<"Usage: "<<argv[0]<<" <in> <out>\n"; return 1; }
    omp_init_lock(&set_lock);

    std::ifstream fin(argv[1]);
    int m;
    fin>>n>>m;
    adj.assign(n,{});
    coverBS.assign(n,{});
    for(int i=0,u,v;i<m;++i){
        fin>>u>>v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    fin.close();
    // build cover sets
    for(int i=0;i<n;++i){
        coverBS[i].lo=coverBS[i].hi=0;
        coverBS[i].set(i);
        for(int v:adj[i]) coverBS[i].set(v);
    }

    // seed with greedy
    greedy_seed();

    // start parallel search
    Bitset128 start{};
    std::vector<int> cur;
    int u0=0;
    #pragma omp parallel
    #pragma omp single nowait
    {
        // force two large tasks at root
        // branch on u0
        #pragma omp task firstprivate(start)
        {
            std::vector<int> c={u0};
            dfs(bs_union(start,coverBS[u0]),1,c);
        }
        // branch on neighbors of u0
        for(int v:adj[u0]){
            #pragma omp task firstprivate(start,v)
            {
                std::vector<int> c={v};
                dfs(bs_union(start,coverBS[v]),1,c);
            }
        }
    }

    // write out bestSet
    std::ofstream fout(argv[2]);
    std::vector<char> out(n,'0');
    for(int v:bestSet) out[v]='1';
    for(char c:out) fout<<c;
    fout<<"\n";
    return 0;
}
