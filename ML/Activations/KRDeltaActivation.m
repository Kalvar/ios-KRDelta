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
        
    }
    return self;
}

// Tanh() named Hyperbolic Tangent which is scope in [-1.0, 1.0]
// Formula is " ( 2.0 / (1.0 + e^(-入 * x)) ) - 1.0 ", the 入 default is 1.0, 越小越平滑
- (double)tanh:(double)x slope:(double)slope
{
    return ( 2.0f / ( 1.0f + pow(M_E, (-slope * x)) ) ) - 1.0f;
}

// Sigmoid() which is scope in [0.0, 1.0]
- (double)sigmoid:(double)x slope:(double)slope
{
    return ( 1.0f / ( 1.0f + pow(M_E, (-slope * x)) ) );
}

// SGN() named Sign Function which is scope in (-1, 1) or (0, 1)
- (float)sgn:(double)x
{
    return ( x >= 0.0f ) ? 1.0f : -1.0f;
}

// RBF() is Gussian Function
- (double)rbf:(double)x sigma:(double)sigma
{
    // Formula : exp^( -s / ( 2.0f * sigma * sigma ) )
    return pow(M_E, ((-x) / ( 2.0f * sigma * sigma )));
}

#pragma mark - Partial Differential
// Partial Differential of Tanh
- (double)partialTanh:(double)x slope:(double)slope
{
    // The original formula : (1 - y^2) * (入 / 2)
    // Derivative = (1 - y) * (1 + y) = (1 - y^2)
    //            = ( 1.0f - outputValue ) * ( 1.0f * outputValue );
    // Optimized derivative method since this methods used tahn() that 入 is 1.0 not standard 2.0
    return ( 1.0f - ( x * x ) ) * (slope / 2.0f);
}

- (double)partialSigmoid:(double)x slope:(double)slope
{
    // Derivative = slope * (1 - y) * y
    return slope * x * ( 1.0f - x );;
}

- (double)partialRBF:(double)x sigma:(double)sigma
{
    return -((2.0 * x) / (2.0 * sigma * sigma)) * pow(M_E, (-x) / (2.0 * sigma * sigma));
}

- (double)partialSgn:(double)x
{
    return x;
}

@end
