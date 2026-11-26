/* WeakBench trusted Metal host. The Python front end supplies validated, generated MSL
   and float32 buffers in its private working directory. This binary never loads
   candidate libraries, executables or host pointers. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

typedef struct {
    uint32_t M, N, K, L, na, nb;
} Shape;

static void fail(NSString *code) {
    fprintf(stderr, "%s\n", code.UTF8String);
    exit(2);
}

static NSData *readData(NSString *file) {
    NSData *data = [NSData dataWithContentsOfFile:file];
    if (!data)
        fail(@"missing_buffer");
    return data;
}

static NSString *filename(NSUInteger idx, NSString *suffix) {
    return [NSString stringWithFormat:@"%lu-%@", (unsigned long)idx, suffix];
}

static id<MTLComputePipelineState> compile(id<MTLDevice> device, NSString *file) {
    NSString *source = [[NSString alloc] initWithData:readData(file) encoding:NSUTF8StringEncoding];
    MTLCompileOptions *options = [MTLCompileOptions new];
    options.fastMathEnabled = NO;
    NSError *error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
    if (!library)
        fail(@"metal_library_compile_failed");
    id<MTLFunction> function = [library newFunctionWithName:@"evaluate"];
    id<MTLComputePipelineState> pipe = [device newComputePipelineStateWithFunction:function error:&error];
    if (!pipe)
        fail(@"metal_pipeline_compile_failed");
    return pipe;
}

static id<MTLBuffer> input(id<MTLDevice> device, NSData *bytes) {
    id<MTLBuffer> b = bytes.length ? [device newBufferWithBytes:bytes.bytes
                                                         length:bytes.length
                                                        options:MTLResourceStorageModeShared]
                                   : [device newBufferWithLength:4 options:MTLResourceStorageModeShared];
    if (!b)
        fail(@"metal_allocation");
    return b;
}

@interface Sample : NSObject {
  @public
    id<MTLBuffer> a, b, out, bad;
    NSData *gold;
    Shape s;
    double atol, rtol;
    BOOL timed;
    NSUInteger repeats;
}
@end
@implementation Sample
@end

static double submit(Sample *x, id<MTLCommandQueue> queue, id<MTLComputePipelineState> pipe,
                     NSUInteger threads, NSUInteger lanes, NSUInteger count) {
    if (threads > pipe.maxTotalThreadsPerThreadgroup)
        fail(@"threadgroup_unsupported");
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    if (!e)
        fail(@"metal_command_encoder");
    [e setComputePipelineState:pipe];
    [e setBuffer:x->a offset:0 atIndex:0];
    [e setBuffer:x->b offset:0 atIndex:1];
    [e setBuffer:x->out offset:0 atIndex:2];
    [e setBytes:&x->s length:sizeof(Shape) atIndex:3];
    [e setBuffer:x->bad offset:0 atIndex:4];
    MTLSize grid = MTLSizeMake((x->s.L + lanes - 1) / lanes, 1, 1);
    MTLSize group = MTLSizeMake(threads, 1, 1);
    for (NSUInteger j = 0; j < count; ++j)
        [e dispatchThreads:grid threadsPerThreadgroup:group];
    [e endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted || cb.error)
        fail(@"metal_command_failed");
    double us = (cb.GPUEndTime - cb.GPUStartTime) * 1e6 / count;
    if (!isfinite(us) || us <= 0)
        fail(@"hardware_timestamps_unavailable");
    return us;
}

static NSDictionary *verify(Sample *x, id<MTLCommandQueue> q, id<MTLComputePipelineState> p,
                            NSUInteger threads, NSUInteger lanes, double *us) {
    float *out = (float *)x->out.contents;
    const float *expected = x->gold.bytes;
    for (uint32_t i = 0; i < x->s.L; ++i)
        out[i] = NAN;
    *(uint32_t *)x->bad.contents = 0;
    *us = submit(x, q, p, threads, lanes, 1);
    uint32_t wrong = 0, nonfinite = 0;
    double maxAbs = 0;
    for (uint32_t i = 0; i < x->s.L; ++i) {
        if (!isfinite(out[i])) {
            ++wrong;
            ++nonfinite;
            continue;
        }
        double diff = fabs((double)out[i] - expected[i]);
        maxAbs = fmax(diff, maxAbs);
        if (diff > x->atol + x->rtol * fabs(expected[i]))
            ++wrong;
    }
    BOOL invalid = *(uint32_t *)x->bad.contents != 0;
    return @{
        @"passed" : @((BOOL)(!invalid && !wrong)),
        @"mismatches" : @(wrong),
        @"nonfinite" : @(nonfinite),
        @"invalid_operation" : @(invalid),
        @"max_abs_error" : @(maxAbs)
    };
}

int main(void) {
    @autoreleasepool {
        NSDictionary *config = [NSJSONSerialization JSONObjectWithData:readData(@"run.json")
                                                               options:0
                                                                 error:nil];
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device || ![device supportsFamily:MTLGPUFamilyApple1])
            fail(@"apple_gpu_unavailable");
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLComputePipelineState> pipes[2] = {compile(device, @"baseline.metal"),
                                                compile(device, @"candidate.metal")};
        NSArray *programs = config[@"programs"], *metadata = config[@"cases"], *order = config[@"order"];
        NSUInteger threads[2], lanes[2];
        for (int i = 0; i < 2; ++i) {
            threads[i] = [programs[i][@"threads"] unsignedIntegerValue];
            lanes[i] = [programs[i][@"outputs_per_thread"] unsignedIntegerValue];
        }
        NSMutableArray *samples = [NSMutableArray array], *rows = [NSMutableArray array];
        BOOL candidateOK = YES;
        for (NSUInteger i = 0; i < metadata.count; ++i) {
            @autoreleasepool {
                NSDictionary *m = metadata[i];
                Sample *x = [Sample new];
                NSArray *shape = m[@"shape"];
                NSData *ad = readData(filename(i, @"a")), *bd = readData(filename(i, @"b"));
                x->gold = readData(filename(i, @"expected"));
                x->s = (Shape){[shape[0] unsignedIntValue], [shape[1] unsignedIntValue],
                               [shape[2] unsignedIntValue], (uint32_t)(x->gold.length / 4),
                               (uint32_t)(ad.length / 4),   (uint32_t)(bd.length / 4)};
                x->a = input(device, ad);
                x->b = input(device, bd);
                x->out = [device newBufferWithLength:x->gold.length options:MTLResourceStorageModeShared];
                x->bad = [device newBufferWithLength:4 options:MTLResourceStorageModeShared];
                if (!x->out || !x->bad)
                    fail(@"metal_allocation");
                x->atol = [m[@"atol"] doubleValue];
                x->rtol = [m[@"rtol"] doubleValue];
                x->timed = [m[@"timing"] boolValue];
                double ru, cu;
                NSDictionary *base = verify(x, queue, pipes[0], threads[0], lanes[0], &ru);
                if (![base[@"passed"] boolValue])
                    fail(@"baseline_disagrees_with_oracle");
                NSDictionary *check = verify(x, queue, pipes[1], threads[1], lanes[1], &cu);
                candidateOK = candidateOK && [check[@"passed"] boolValue];
                x->repeats = MAX(1, MIN(64, (NSUInteger)ceil(2000.0 / fmax(ru, cu))));
                [samples addObject:x];
                [rows addObject:[@{
                          @"shape" : shape,
                          @"elements" : @(x->s.L),
                          @"validation" : check
                      } mutableCopy]];
            }
        }
        if (candidateOK)
            for (NSUInteger i = 0; i < samples.count; ++i) {
                @autoreleasepool {
                    Sample *x = samples[i];
                    if (!x->timed)
                        continue;
                    for (int warm = 0; warm < 2; ++warm)
                        for (int p = 0; p < 2; ++p)
                            submit(x, queue, pipes[p], threads[p], lanes[p], x->repeats);
                    NSMutableArray *times = [NSMutableArray array];
                    for (NSNumber *who in order[i]) {
                        int first = who.intValue;
                        double t[2];
                        for (int j = 0; j < 2; ++j) {
                            int p = first ^ j;
                            t[p] = submit(x, queue, pipes[p], threads[p], lanes[p], x->repeats);
                        }
                        [times addObject:@{
                            @"baseline_us" : @(t[0]),
                            @"candidate_us" : @(t[1]),
                            @"baseline_first" : @((BOOL)(!first))
                        }];
                    }
                    rows[i][@"rounds"] = times;
                    rows[i][@"dispatches_per_sample"] = @(x->repeats);
                }
            }
        NSDictionary *result = @{
            @"status" : candidateOK ? @"correct" : @"incorrect",
            @"backend" : @"metal",
            @"device" : device.name,
            @"registry_id" : @(device.registryID),
            @"simd_width" : @(pipes[1].threadExecutionWidth),
            @"cases" : rows
        };
        NSData *data = [NSJSONSerialization dataWithJSONObject:result options:0 error:nil];
        if (!data)
            fail(@"invalid_result");
        fwrite(data.bytes, 1, data.length, stdout);
        return 0;
    }
}
