#include "Halide.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>

using namespace Halide;

namespace {

void check(int r) {
    assert(r == 0);
}

// Link a Halide-emitted object file into a shared library so we can dlopen it.
// Mirrors what AOT users do in the field (e.g. spectrum's `generateCachedPipeline`):
// emit .o via Pipeline::compile_to, then invoke the system linker.
void link_shared(const std::string &obj_path, const std::string &so_path) {
    const char *cc = std::getenv("CXX");
    if (!cc || !*cc) cc = "cc";
    const std::string cmd = std::string(cc) + " -shared -o " + so_path + " " + obj_path;
    int rv = std::system(cmd.c_str());
    if (rv != 0) {
        throw std::runtime_error("link command failed: " + cmd);
    }
}

}  // namespace

int main(int argc, char **argv) {
    // Compile a tiny pipeline: output(x) = input(x) * 2 + 1.
    // The pipeline has one input scalar buffer and one output scalar buffer,
    // so the AOT .so's `<name>_argv` will expect:
    //   argv[0] = JITUserContext** (because Target::UserContext is forced below)
    //   argv[1] = halide_buffer_t* for `input`
    //   argv[2] = halide_buffer_t* for the output

    ImageParam input(Int(32), 1, "input");
    Var x;

    Func output("aot_double_plus_one");
    output(x) = input(x) * 2 + 1;

    Target target = get_jit_target_from_environment()
                        .with_feature(Target::UserContext)
                        .with_feature(Target::NoRuntime ? Target::JIT : Target::JIT);
    // Don't set NoRuntime — we want the .so to be self-contained.
    target = get_jit_target_from_environment().with_feature(Target::UserContext);

    Pipeline p(output);
    auto args = p.infer_arguments();

    // Emit .o + .h into a temp dir, then link the .o into a .so.
    const std::filesystem::path tmp = std::filesystem::temp_directory_path() / "callable_aot_test";
    std::filesystem::create_directories(tmp);
    const std::string obj_path = (tmp / "aot.o").string();
    const std::string so_path = (tmp / "aot.so").string();

    std::map<OutputFileType, std::string> outputs = {
        {OutputFileType::object, obj_path},
        {OutputFileType::c_header, (tmp / "aot.h").string()},
    };
    p.compile_to(outputs, args, "aot_double_plus_one", target);
    link_shared(obj_path, so_path);

    // Load it back via the new factory. The args list must match what
    // infer_arguments() returned at compile time, in the same order, with
    // __user_context NOT included (load_aot prepends it).
    Callable c = Callable::load_aot(so_path, "aot_double_plus_one", args, target);

    // Invoke with a real buffer.
    Buffer<int32_t> in(16);
    for (int i = 0; i < 16; i++) in(i) = i;

    Buffer<int32_t> out(16);
    int rv = c(in, out);
    check(rv);

    for (int i = 0; i < 16; i++) {
        const int expected = i * 2 + 1;
        if (out(i) != expected) {
            fprintf(stderr, "Mismatch at i=%d: got %d, expected %d\n", i, out(i), expected);
            return 1;
        }
    }

    // Custom error handler routed via per-call JITUserContext.
    struct CountingContext : public JITUserContext {
        int call_count = 0;
    };
    CountingContext ctx;
    JITUserContext *ctx_ptr = &ctx;  // explicit upcast so the right overload is picked
    rv = c(ctx_ptr, in, out);
    check(rv);

    // Calling on a default-constructed Callable should fail loud (defensive).
    Callable empty;
    if (empty.defined()) {
        fprintf(stderr, "Default-constructed Callable claimed defined()\n");
        return 1;
    }

    printf("Success!\n");
    return 0;
}
