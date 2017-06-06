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
@property (nonatomic) NSMutableArray <NSNumber *> *lastDeltaWeights; // 上一次的權重改變量
@property (nonatomic) double learningRate; // Global learning rate.
@property (nonatomic) double deltaValue;   // Current delta value.
@property (nonatomic) NSMutableArray <NSArray <NSNumber *> *> *deltaChanges; // Standard SGD weight-changes: learning rate * delta value

+ (instancetype)shared;
- (instancetype)initWithOptimization:(KRDeltaOptimizationMethods)method;
- (instancetype)init;

- (void)recordLastDeltaWeights:(NSArray <NSNumber *> *)lastChanges;
- (void)addDeltaChanges:(NSArray <NSNumber *> *)changes;
- (void)cleanDeltaChanges;

- (void)runStandardSGD;
- (NSArray <NSNumber *> *)standardBatchChanges;
- (NSArray <NSNumber *> *)optmizedBatchChanges;

@end
