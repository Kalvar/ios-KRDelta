//
//  KRDeltaPartial.m
//  KRDelta
//
//  Created by Kalvar Lin on 2017/3/17.
//  Copyright © 2017年 Kalvar Lin. All rights reserved.
//

#import "KRDeltaPartial.h"

@implementation KRDeltaPartial

#pragma mark - Partial Differential
// The x is tanh(x) that output result.
- (double)tanh:(double)x slope:(double)slope
{
    // The original formula : (1 - y^2) * (入 / 2)
    // Derivative = (1 - y) * (1 + y) = (1 - y^2)
    //            = ( 1.0f - outputValue ) * ( 1.0f * outputValue );
    // Optimized derivative method since this methods used tahn() that 入 is 1.0 not standard 2.0
    return (1.0f - ( x * x )) * (slope / 2.0f);
}

- (double)sigmoid:(double)x slope:(double)slope
{
    // Derivative = slope * (1 - y) * y
    return slope * x * ( 1.0f - x );
}

- (double)rbf:(double)x sigma:(double)sigma
{
    return -((2.0f * x) / (2.0f * sigma * sigma)) * exp(-x / (2.0f * sigma * sigma));
}

// The x is summed input value of net, not the activated output-value.
- (double)sgn:(double)x
{
    return (x >= 0.0f) ? 1.0 : -1.0;
}

- (double)reLU:(double)x
{
    return (x > 0.0f) ? 1.0f : 0.0f;
}

- (double)leakyReLU:(double)x
{
    return (x > 0.0f) ? 1.0f : 0.01f;
}

- (double)eLU:(double)x
{
    return (x >= 0.0f) ? x : 0.01 * exp(x);
}

@end
