//
//  KRDeltaOptimization.h
//  KRDelta
//
//  Created by Kalvar Lin on 2017/6/5.
//  Copyright © 2017年 Kalvar Lin. All rights reserved.
//

#import <Foundation/Foundation.h>

typedef NS_ENUM(NSInteger, KRDeltaOptimizationMethods)
{
    KRDeltaOptimizationDefault,     // Default grandient decent.
    KRDeltaOptimizationFixedInertia // Fixed inertia that default is 0.5f
};

@interface KRDeltaOptimization : NSObject

@property (nonatomic) double fixedInertia;
@property (nonatomic) KRDeltaOptimizationMethods method;
@property (nonatomic) NSArray <NSNumber *> *inputs;
@property (nonatomic) NSArray <NSNumber *> *lastDeltaWeights;
@property (nonatomic) double learningRate; // Global learning rate.
@property (nonatomic) double deltaValue;   // Current delta value.

+ (instancetype)shared;
- (instancetype)initWithOptimization:(KRDeltaOptimizationMethods)method;
- (instancetype)init;

- (NSArray <NSNumber *> *)optimizedDeltaWeights;
- (NSArray <NSNumber *> *)standardDeltaWeights;

@end
