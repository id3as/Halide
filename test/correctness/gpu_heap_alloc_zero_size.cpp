#include "Halide.h"
#include <stdio.h>

using namespace Halide;

// A dynamically-sized per-thread allocation inside a GPU kernel goes into a
// heap buffer that the host sizes before the launch by replaying the
// kernel's loops around each use and taking the max. When the replay
// reaches no use on this launch (here, a reduction with a runtime extent of
// zero; in practice also a use under a specialization whose condition is
// false) that size is zero, and the device runtimes refuse a zero-byte
// allocation: Metal errors with "Failed to allocate buffer of size 0" and
// CUDA aborts. The launch must still succeed.
int main(int argc, char **argv) {
    Target t = get_jit_target_from_environment();
    if (!t.has_gpu_feature()) {
        printf("[SKIP] No GPU target enabled.\n");
        return 0;
    }

    Func f("f"), h("h"), g("g");
    Var x("x"), xi("xi");
    Param<int> n("n");
    RDom r(0, n);

    f(x) = x * 2;
    h(x) = 0;
    h(x) += f(x + r) + f(x + 2 * r);
    g(x) = h(x);

    // f's footprint, [x + r, x + 2r], depends on r, so its size is found on
    // the host by replaying the loop over r, which runs n times.
    g.gpu_tile(x, xi, 16);
    h.compute_at(g, xi);
    f.compute_at(h, r);

    const int size = 64;
    for (int taps : {3, 0}) {
        n.set(taps);
        Buffer<int> out = g.realize({size}, t);
        out.copy_to_host();
        for (int i = 0; i < size; i++) {
            int expected = taps == 0 ? 0 : 4 * taps * i + 3 * taps * (taps - 1);
            if (out(i) != expected) {
                printf("n=%d: out(%d) = %d instead of %d\n", taps, i, out(i), expected);
                return 1;
            }
        }
    }

    printf("Success!\n");
    return 0;
}
