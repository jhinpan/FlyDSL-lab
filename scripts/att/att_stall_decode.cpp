// Decode a rocprofv3 ATT (.att) shader-data blob into per-PC stall statistics.
//
//   attdec <att-file> <codeobj-id>:<load_base>:<load_size>:<file> [...]
//
// Emits CSV on stdout: code_object_id,address,category,executions,stall_cycles,
// total_cycles. `stall` is the cycles the wave sat on the instruction before it
// issued; `duration - stall` is the issue time.
#include <rocprofiler-sdk/experimental/thread-trace/trace_decoder.h>

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace {

struct Key
{
    uint64_t co, addr;
    bool operator<(const Key& o) const { return co != o.co ? co < o.co : addr < o.addr; }
};

struct Agg
{
    uint64_t count = 0, stall = 0, dur = 0;
    uint32_t cat = 0;
};

std::map<Key, Agg> g_agg;
uint64_t           g_waves = 0, g_inst = 0;
uint64_t           g_state[ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_LAST] = {0};
uint64_t           g_info[ROCPROFILER_THREAD_TRACE_DECODER_INFO_LAST]    = {0};

void
callback(rocprofiler_thread_trace_decoder_record_type_t type,
         void*                                          events,
         uint64_t                                       size,
         void*)
{
    if(type == ROCPROFILER_THREAD_TRACE_DECODER_RECORD_WAVE)
    {
        auto* w = static_cast<rocprofiler_thread_trace_decoder_wave_t*>(events);
        for(uint64_t i = 0; i < size; i++)
        {
            g_waves++;
            for(uint64_t k = 0; k < w[i].timeline_size; k++)
            {
                auto& t = w[i].timeline_array[k];
                if(t.type >= 0 && t.type < ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_LAST)
                    g_state[t.type] += static_cast<uint64_t>(t.duration);
            }
            for(uint64_t k = 0; k < w[i].instructions_size; k++)
            {
                auto& in = w[i].instructions_array[k];
                auto& a  = g_agg[Key{in.pc.code_object_id, in.pc.address}];
                a.count++;
                a.stall += in.stall;
                a.dur += static_cast<uint64_t>(in.duration);
                a.cat = in.category;
                g_inst++;
            }
        }
    }
    else if(type == ROCPROFILER_THREAD_TRACE_DECODER_RECORD_INFO)
    {
        auto* info = static_cast<rocprofiler_thread_trace_decoder_info_t*>(events);
        for(uint64_t i = 0; i < size; i++)
            if(info[i] < ROCPROFILER_THREAD_TRACE_DECODER_INFO_LAST) g_info[info[i]]++;
    }
}

std::vector<char>
slurp(const std::string& path)
{
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if(!f) return {};
    auto              n = f.tellg();
    std::vector<char> buf(static_cast<size_t>(n));
    f.seekg(0);
    f.read(buf.data(), n);
    return buf;
}

}  // namespace

int
main(int argc, char** argv)
{
    if(argc < 3)
    {
        fprintf(stderr, "usage: %s <att-file> <id:base:size:file> ...\n", argv[0]);
        return 2;
    }

    rocprofiler_thread_trace_decoder_id_t handle{};
    const char* libpath = getenv("ATT_DECODER_PATH");
    if(!libpath) libpath = "/opt/rocm/lib";
    if(rocprofiler_thread_trace_decoder_create(&handle, libpath) != ROCPROFILER_STATUS_SUCCESS)
    {
        fprintf(stderr, "decoder create failed (path=%s)\n", libpath);
        return 1;
    }

    std::vector<std::vector<char>> keepalive;
    for(int i = 2; i < argc; i++)
    {
        std::string s = argv[i];
        size_t      p1 = s.find(':'), p2 = s.find(':', p1 + 1), p3 = s.find(':', p2 + 1);
        uint64_t    id   = strtoull(s.substr(0, p1).c_str(), nullptr, 0);
        uint64_t    base = strtoull(s.substr(p1 + 1, p2 - p1 - 1).c_str(), nullptr, 0);
        uint64_t    sz   = strtoull(s.substr(p2 + 1, p3 - p2 - 1).c_str(), nullptr, 0);
        auto        buf  = slurp(s.substr(p3 + 1));
        if(buf.empty()) continue;
        keepalive.push_back(std::move(buf));
        auto& b = keepalive.back();
        rocprofiler_thread_trace_decoder_codeobj_load(handle, id, base, sz, b.data(), b.size());
    }

    auto att = slurp(argv[1]);
    if(att.empty())
    {
        fprintf(stderr, "empty att file\n");
        return 1;
    }
    auto st = rocprofiler_trace_decode(handle, callback, att.data(), att.size(), nullptr);
    fprintf(stderr, "decode status=%d waves=%" PRIu64 " insts=%" PRIu64 "\n", (int) st, g_waves, g_inst);
    static const char* wname[] = {"EMPTY", "IDLE", "EXEC", "WAIT", "STALL"};
    for(int i = 0; i < ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_LAST; i++)
        fprintf(stderr, "  wavestate %-6s %" PRIu64 " cycles\n", wname[i], g_state[i]);
    static const char* iname[] = {"NONE", "DATA_LOST", "STITCH_INCOMPLETE", "WAVE_INCOMPLETE"};
    for(int i = 0; i < ROCPROFILER_THREAD_TRACE_DECODER_INFO_LAST; i++)
        if(g_info[i]) fprintf(stderr, "  info %-20s %" PRIu64 "\n", iname[i], g_info[i]);

    printf("code_object_id,address,category,executions,stall_cycles,total_cycles\n");
    for(auto& [k, a] : g_agg)
        printf("%" PRIu64 ",%" PRIu64 ",%u,%" PRIu64 ",%" PRIu64 ",%" PRIu64 "\n",
               k.co, k.addr, a.cat, a.count, a.stall, a.dur);

    rocprofiler_thread_trace_decoder_destroy(handle);
    return 0;
}
