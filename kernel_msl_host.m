// Trusted launcher for native Metal compute shaders. Only buffer data reaches the device.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdint.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <sys/resource.h>

static const NSUInteger Guard = 256;
static void fail(const char *code) {
    fprintf(stderr, "%s\n", code);
    exit(2);
}
static NSData *readFile(NSString *file) {
    NSData *data = [NSData dataWithContentsOfFile:file];
    if (!data)
        fail("missing_input");
    return data;
}
static void report(NSDictionary *data) {
    NSData *json = [NSJSONSerialization dataWithJSONObject:data options:0 error:nil];
    if (!json || fwrite(json.bytes, 1, json.length, stdout) != json.length)
        fail("report_error");
    exit(0);
}
static void submissionError(BOOL candidate, NSString *reason, BOOL compiled) {
    if (!candidate)
        fail("invalid_reference");
    report(@{
        @"status" : compiled ? @"incorrect" : @"compile_error",
        @"compiled" : @(compiled),
        @"reason" : reason,
        @"cases" : @[]
    });
}
static id<MTLComputePipelineState> pipeline(id<MTLDevice> d, BOOL candidate) {
    NSString *file = candidate ? @"candidate.metal" : @"baseline.metal";
    NSString *source = [[NSString alloc] initWithData:readFile(file) encoding:NSUTF8StringEncoding];
    MTLCompileOptions *opts = [MTLCompileOptions new];
    opts.fastMathEnabled = NO;
    NSError *error = nil;
    id<MTLLibrary> lib = [d newLibraryWithSource:source options:opts error:&error];
    if (!lib) {
        if ([error.domain isEqualToString:MTLLibraryErrorDomain] && error.code == MTLLibraryErrorCompileFailure)
            submissionError(candidate, @"compile_error", NO);
        fail("metal_compiler_unavailable");
    }
    id<MTLFunction> kernel = [lib newFunctionWithName:@"evaluate"];
    if (!kernel)
        submissionError(candidate, @"entry_missing", NO);
    if (kernel.functionType != MTLFunctionTypeKernel)
        submissionError(candidate, @"source_contract", NO);
    MTLComputePipelineReflection *reflection = nil;
    id<MTLComputePipelineState> pipe =
        [d newComputePipelineStateWithFunction:kernel
                                       options:MTLPipelineOptionBindingInfo | MTLPipelineOptionBufferTypeInfo
                                    reflection:&reflection
                                         error:&error];
    if (!pipe) {
        if ([error.domain isEqualToString:MTLLibraryErrorDomain] && error.code == MTLLibraryErrorCompileFailure)
            submissionError(candidate, @"compile_error", NO);
        // The toolchain may fail internally or the device may be unavailable.
        fail("pipeline_unavailable");
    }
    if (!reflection)
        fail("reflection_unavailable");
    unsigned seen = 0;
    for (id<MTLBinding> arg in reflection.bindings) {
        if (arg.type != MTLBindingTypeBuffer || arg.index > 3 || (seen & (1u << arg.index)))
            submissionError(candidate, @"source_contract", NO);
        seen |= 1u << arg.index;
        id<MTLBufferBinding> b = (id<MTLBufferBinding>)arg;
        BOOL dims = arg.index == 3;
        BOOL accessOK = arg.index == 2
                            ? (b.access == MTLBindingAccessReadWrite || b.access == MTLBindingAccessWriteOnly)
                            : (b.access == MTLBindingAccessReadOnly);
        if (!accessOK || b.bufferDataType != (dims ? MTLDataTypeUInt4 : MTLDataTypeFloat) ||
            b.bufferDataSize != (dims ? 16 : 4))
            submissionError(candidate, @"source_contract", NO);
    }
    if (!(seen & 4) || pipe.maxTotalThreadsPerThreadgroup < 64)
        submissionError(candidate, @"source_contract", NO);
    return pipe;
}

@interface Case : NSObject {
  @public
    NSArray<NSData *> *images;
    NSData *gold;
    NSArray *shape;
    double atol, rtol;
    BOOL timed;
}
@end
@implementation Case
@end

static Case *loadCase(NSUInteger i, NSDictionary *meta) {
    Case *c = [Case new];
    c->shape = meta[@"shape"];
    NSString *prefix = [NSString stringWithFormat:@"%lu-", i];
    NSData *a = readFile([prefix stringByAppendingString:@"a"]);
    NSData *b = readFile([prefix stringByAppendingString:@"b"]);
    c->gold = readFile([prefix stringByAppendingString:@"expected"]);
    uint32_t dims[4] = {[c->shape[0] unsignedIntValue],
                        [c->shape[1] unsignedIntValue],
                        [c->shape[2] unsignedIntValue],
                        (uint32_t)(c->gold.length / sizeof(double))};
    NSMutableData *blank = [NSMutableData dataWithLength:dims[3] * sizeof(float)];
    float *p = blank.mutableBytes;
    for (NSUInteger j = 0; j < dims[3]; ++j)
        p[j] = NAN;
    c->images = @[ a, b, blank, [NSData dataWithBytes:dims length:sizeof(dims)] ];
    c->timed = [meta[@"timing"] boolValue];
    c->atol = [meta[@"atol"] doubleValue];
    c->rtol = [meta[@"rtol"] doubleValue];
    return c;
}

static BOOL guardIntact(id<MTLBuffer> buffer, NSUInteger size) {
    const uint8_t *p = buffer.contents;
    for (NSUInteger i = 0; i < Guard; ++i) {
        if (p[i] != 0xa5 || p[Guard + size + i] != 0x5a)
            return NO;
    }
    return YES;
}
static NSDictionary *dispatch(Case *c, id<MTLCommandQueue> queue, id<MTLComputePipelineState> pipe, BOOL candidate,
                              double *elapsed) {
    id<MTLBuffer> buffers[4];
    for (NSUInteger i = 0; i < 4; ++i) {
        NSData *data = c->images[i];
        buffers[i] = [queue.device newBufferWithLength:2 * Guard + data.length
                                               options:MTLResourceStorageModeShared];
        if (!buffers[i])
            fail("metal_allocation");
        uint8_t *p = buffers[i].contents;
        memset(p, 0xa5, Guard);
        if (data.length)
            memcpy(p + Guard, data.bytes, data.length);
        memset(p + Guard + data.length, 0x5a, Guard);
    }
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [command computeCommandEncoder];
    if (!enc)
        fail("command_encoder_unavailable");
    [enc setComputePipelineState:pipe];
    for (NSUInteger i = 0; i < 4; ++i)
        [enc setBuffer:buffers[i] offset:Guard atIndex:i];
    NSUInteger length = c->gold.length / sizeof(double);
    [enc dispatchThreadgroups:MTLSizeMake((length + 63) / 64, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    [enc endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted || command.error) {
        NSError *error = command.error;
        if ([error.domain isEqualToString:MTLCommandBufferErrorDomain] &&
            (error.code == MTLCommandBufferErrorPageFault || error.code == MTLCommandBufferErrorStackOverflow))
            submissionError(candidate, @"device_execution_error", YES);
        fail("command_unavailable");
    }
    *elapsed = (command.GPUEndTime - command.GPUStartTime) * 1e6;
    if (!isfinite(*elapsed) || *elapsed <= 0)
        fail("hardware_timestamps_unavailable");
    BOOL intact = YES;
    for (NSUInteger i = 0; i < 4; ++i) {
        NSData *data = c->images[i];
        intact &= guardIntact(buffers[i], data.length);
        if (i != 2 && data.length)
            intact &= memcmp((uint8_t *)buffers[i].contents + Guard, data.bytes, data.length) == 0;
    }
    const float *got = (const float *)((uint8_t *)buffers[2].contents + Guard);
    const double *want = c->gold.bytes;
    NSUInteger wrong = 0, nonfinite = 0;
    double maximum = 0;
    for (NSUInteger i = 0; i < length; ++i) {
        if (!isfinite(got[i])) {
            ++wrong;
            ++nonfinite;
            continue;
        }
        double err = fabs((double)got[i] - want[i]);
        maximum = fmax(maximum, err);
        if (err > c->atol + c->rtol * fabs(want[i]))
            ++wrong;
    }
    BOOL passed = intact && wrong == 0;
    if (!candidate && !passed)
        fail("baseline_disagrees_with_oracle");
    return @{
        @"passed" : @(passed),
        @"invalid_operation" : @((BOOL)!intact),
        @"buffers_intact" : @(intact),
        @"mismatches" : @(wrong),
        @"nonfinite" : @(nonfinite),
        @"max_abs_error" : @(maximum)
    };
}

int main(int argc, const char **argv) {
    @autoreleasepool {
        if (argc != 2)
            fail("invalid_request");
        struct rlimit cpu = {90, 90}, file = {16 * 1024 * 1024, 16 * 1024 * 1024}, core = {0, 0};
        // Let optional Metal cache writes receive EFBIG at the limit. The default
        // SIGXFSZ action terminates the host before it can report the candidate.
        if (signal(SIGXFSZ, SIG_IGN) == SIG_ERR || setrlimit(RLIMIT_CPU, &cpu) || setrlimit(RLIMIT_FSIZE, &file) || setrlimit(RLIMIT_CORE, &core))
            fail("worker_limits_unavailable");
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device || ![device supportsFamily:MTLGPUFamilyApple1])
            fail("apple_gpu_unavailable");
        id<MTLComputePipelineState> candidate = pipeline(device, YES);
        NSMutableDictionary *result = [@{
            @"compiled" : @YES,
            @"device" : device.name,
            @"registry_id" : @(device.registryID),
            @"simd_width" : @(candidate.threadExecutionWidth),
            @"backend" : @"metal-native"
        } mutableCopy];
        if (strcmp(argv[1], "compile") == 0) {
            result[@"status"] = @"compiled";
            report(result);
        }
        if (strcmp(argv[1], "measure") != 0)
            fail("invalid_request");
        id<MTLComputePipelineState> pipes[2] = {pipeline(device, NO), candidate};
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue)
            fail("queue_unavailable");
        NSDictionary *cfg = [NSJSONSerialization JSONObjectWithData:readFile(@"run.json") options:0 error:nil];
        NSArray *metadata = cfg[@"cases"], *order = cfg[@"order"];
        NSMutableArray<Case *> *cases = [NSMutableArray array];
        NSMutableArray<NSMutableDictionary *> *rows = [NSMutableArray array];
        BOOL passed = YES;
        for (NSUInteger i = 0; i < metadata.count; ++i) {
            @autoreleasepool {
                Case *c = loadCase(i, metadata[i]);
                [cases addObject:c];
                double t;
                dispatch(c, queue, pipes[0], NO, &t);
                NSDictionary *check = dispatch(c, queue, pipes[1], YES, &t);
                passed &= [check[@"passed"] boolValue];
                [rows addObject:[@{
                          @"shape" : c->shape,
                          @"elements" : @(c->gold.length / sizeof(double)),
                          @"validation" : check
                      } mutableCopy]];
            }
        }
        for (NSUInteger i = 0; passed && i < cases.count; ++i) {
            Case *c = cases[i];
            if (!c->timed)
                continue;
            NSMutableArray *rounds = [NSMutableArray array];
            // Two warm-ups in opposite orders, then the registered balanced pairs.
            NSArray *sequence = [@[ @0, @1 ] arrayByAddingObjectsFromArray:order[i]];
            for (NSUInteger j = 0; passed && j < sequence.count; ++j) {
                @autoreleasepool {
                    double t[2];
                    int first = [sequence[j] intValue];
                    for (int n = 0; n < 2; ++n) {
                        int which = first ^ n;
                        NSDictionary *check = dispatch(c, queue, pipes[which], which == 1, &t[which]);
                        if (![check[@"passed"] boolValue]) {
                            rows[i][@"validation"] = check;
                            passed = NO;
                            break;
                        }
                    }
                    if (passed && j >= 2)
                        [rounds addObject:@{
                            @"baseline_first" : @((BOOL)!first),
                            @"baseline_us" : @(t[0]),
                            @"candidate_us" : @(t[1])
                        }];
                }
            }
            if (passed) {
                rows[i][@"rounds"] = rounds;
                rows[i][@"dispatches_per_sample"] = @1;
            }
        }
        if (!passed)
            for (NSMutableDictionary *row in rows) {
                [row removeObjectForKey:@"rounds"];
                [row removeObjectForKey:@"dispatches_per_sample"];
            }
        result[@"cases"] = rows;
        result[@"status"] = passed ? @"correct" : @"incorrect";
        report(result);
    }
}
