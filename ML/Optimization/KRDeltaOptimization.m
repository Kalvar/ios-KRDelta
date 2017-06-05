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
        _fixedInertia = 0.5f;
        _method       = method;
    }
    return self;
}

- (instancetype)init
{
    return [self initWithOptimization:KRDeltaOptimizationDefault];
}

- (NSArray <NSNumber *> *)optimizedDeltaWeights
{
    __block NSMutableArray *optimizedWeights = [NSMutableArray new];
    __block double deltaChanges              = _learningRate * _deltaValue;
    __weak typeof(self) weakSelf             = self;
    [_lastDeltaWeights enumerateObjectsUsingBlock:^(NSNumber * _Nonnull lastDeltaWeight, NSUInteger idx, BOOL * _Nonnull stop) {
        __strong typeof(weakSelf) strongSelf = weakSelf;
        double input       = [[strongSelf.inputs objectAtIndex:idx] doubleValue];
        double deltaWeight = deltaChanges * input;
        switch (_method)
        {
            case KRDeltaOptimizationFixedInertia:
                deltaWeight += (strongSelf.fixedInertia * [lastDeltaWeight doubleValue]);
                break;
            default:
                break;
        }
        [optimizedWeights addObject:@(deltaWeight)];
    }];
    return optimizedWeights;
}

- (NSArray <NSNumber *> *)standardDeltaWeights
{
    NSMutableArray *deltaWeights = [NSMutableArray new];
    double deltaChanges          = _learningRate * _deltaValue;
    for(NSNumber *input in _inputs)
    {
        [deltaWeights addObject:@(deltaChanges * [input doubleValue])];
    }
    return deltaWeights;
}

@end
