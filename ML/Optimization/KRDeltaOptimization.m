//
//  KRDeltaOptimization.m
//  KRDelta
//
//  Created by Kalvar Lin on 2017/6/5.
//  Copyright © 2017年 Kalvar Lin. All rights reserved.
//

#import "KRDeltaOptimization.h"

@interface KRDeltaOptimization ()

@end

@implementation KRDeltaOptimization

+ (instancetype)shared
{
    static dispatch_once_t pred;
    static KRDeltaOptimization *object = nil;
    dispatch_once(&pred, ^{
        object = [[KRDeltaOptimization alloc] init];
    });
    return object;
}

- (instancetype)initWithOptimization:(KRDeltaOptimizationMethods)method
{
    self = [super init];
    if(self)
    {
        _fixedInertia     = 0.5f;
        _method           = method;
        _lastDeltaWeights = [[NSMutableArray alloc] init];
        _deltaChanges     = [[NSMutableArray alloc] init];
    }
    return self;
}

- (instancetype)init
{
    return [self initWithOptimization:KRDeltaOptimizationDefault];
}

- (void)recordLastDeltaWeights:(NSArray <NSNumber *> *)lastChanges
{
    if(nil != lastChanges)
    {
        [_lastDeltaWeights removeAllObjects];
        [_lastDeltaWeights addObjectsFromArray:lastChanges];
    }
}

- (void)addDeltaChanges:(NSArray<NSNumber *> *)changes
{
    if(nil != changes)
    {
        [_deltaChanges addObject:changes];
    }
}

- (void)cleanDeltaChanges
{
    [_deltaChanges removeAllObjects];
}

// Original SGD formula: - learning rate * delta value * input value
- (void)runStandardSGD
{
    NSMutableArray *deltaWeights = [[NSMutableArray alloc] init];
    double deltaChanges          = _learningRate * _deltaValue;
    for(NSNumber *input in _inputs)
    {
        [deltaWeights addObject:@(deltaChanges * [input doubleValue])];
    }
    [self addDeltaChanges:deltaWeights];
}

- (NSArray <NSNumber *> *)standardBatchChanges
{
    __block NSInteger batchCount                 = [_deltaChanges count];
    __block NSMutableArray <NSNumber *> *changes = [[_deltaChanges firstObject] mutableCopy];
    [_deltaChanges removeObjectAtIndex:0];
    // Sum every delta changes, if batchSize = 1, the forloop won't be running.
    for(NSArray<NSNumber *> *deltaWeights in _deltaChanges)
    {
        // Sum and average eachs through forlooping every weight.
        [deltaWeights enumerateObjectsUsingBlock:^(NSNumber * _Nonnull deltaValue, NSUInteger idx, BOOL * _Nonnull stop) {
            double nowDeltaValue = [[changes objectAtIndex:idx] doubleValue];
            changes[idx]         = @((nowDeltaValue + [deltaValue doubleValue]) / batchCount);
        }];
    }
    [self cleanDeltaChanges];
    return changes;
}

- (NSArray <NSNumber *> *)optmizedBatchChanges
{
    __block NSMutableArray *optimizedChanges = [[NSMutableArray alloc] initWithArray:[self standardBatchChanges]];
    __weak typeof(self) weakSelf             = self;
    [_lastDeltaWeights enumerateObjectsUsingBlock:^(NSNumber * _Nonnull lastDeltaWeight, NSUInteger idx, BOOL * _Nonnull stop) {
        __strong typeof(weakSelf) strongSelf = weakSelf;
        double deltaWeight = [[optimizedChanges objectAtIndex:idx] doubleValue]; // SGD delta weight
        switch (_method)
        {
            case KRDeltaOptimizationFixedInertia:
                deltaWeight += (strongSelf.fixedInertia * [lastDeltaWeight doubleValue]);
                break;
            default:
                break;
        }
        optimizedChanges[idx] = @(deltaWeight); // Replaces array[idx] item.
    }];
    return optimizedChanges;
}

@end
