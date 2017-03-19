//
//  KRDeltaActivation.m
//  KRDelta
//
//  Created by kalvar_lin on 2016/12/6.
//  Copyright © 2016年 Kalvar Lin. All rights reserved.
//

#import "KRDeltaActivation.h"

@implementation KRDeltaActivation

- (instancetype)init
{
    self = [super init];
    if( self )
    {
        _partial = [[KRDeltaPartial alloc] init];
    }
    return self;
}

// Tanh() named Hyperbolic Tangent which is scope in [-1.0, 1.0]
// Formula is " ( 2.0 / (1.0 + e^(-入 * x)) ) - 1.0 ", the 入 default is 1.0, 越小越平滑
- (double)tanh:(double)x slope:(double)slope
{
    return (2.0f / (1.0f + exp(-slope * x))) - 1.0f;
}

// Sigmoid() which is scope in [0.0, 1.0]
- (double)sigmoid:(double)x slope:(double)slope
{
    return (1.0f / (1.0f + exp(-slope * x)));
}

// SGN() named Sign Function which is scope in (-1, 1) or (0, 1)
- (double)sgn:(double)x
{
    return (x >= 0.0f) ? 1.0f : -1.0f;
}

// RBF() is Gussian Function
- (double)rbf:(double)x sigma:(double)sigma
{
    // Formula : exp^( -s / ( 2.0f * sigma * sigma ) )
    return exp(-x / (2.0f * sigma * sigma));
}

// 其微分的值就是原來 if else 判斷式裡的值 (1, 0)
- (double)reLU:(double)x
{
    // if x >= 0, then x
    // if x < 0, then 0
    return MAX(0.0, x);
}

- (double)leakyReLU:(double)x
{
    return MAX(0.01f * x, x);
}

- (double)eLU:(double)x
{
    return (x >= 0.0f) ? x : 0.01 * (exp(x) - 1.0f);
}

@end
