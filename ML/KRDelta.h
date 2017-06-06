//
//  KRDelta.h
//  KRDelta
//
//  Created by Kalvar ( ilovekalvar@gmail.com ) on 13/6/13.
//  Copyright (c) 2013 - 2014年 Kuo-Ming Lin. All rights reserved.
//

#import "KRDeltaFetcher.h"
#import "KRDeltaOptimization.h"

static NSInteger kKRDeltaFullBatch = 0;

typedef NS_ENUM(NSInteger, KRDeltaActiveFunctions)
{
    KRDeltaActivationSGN = 0,  // Sign Function 符號函數
    KRDeltaActivationTanh,     // Hyperbolic Tangent 雙曲正切函數
    KRDeltaActivationSigmoid,  // Sigmoid S形函數
    KRDeltaActivationRBF,      // RBF 高斯函數
    KRDeltaActivationReLU,
    KRDeltaActivationELU,
    KRDeltaActivationLeakyReLU
};

typedef void(^KRDeltaCompletion)(BOOL success, NSArray <NSNumber *> *weights, NSInteger totalIteration);
typedef void(^KRDeltaIteration)(NSInteger iteration, NSArray <NSNumber *> *weights);
typedef void(^KRDeltaDirectOutput)(NSArray <NSNumber *> *outputs);
typedef void(^KRDeltaBeforeUpdate)(NSInteger iteration, NSArray <NSNumber *> *deltaWeights, NSArray <NSNumber *> *lastDeltaWeights);

@interface KRDelta : NSObject <NSCoding>

@property (nonatomic, strong) NSMutableArray <NSArray <NSNumber *> *> *patterns;
@property (nonatomic, strong) NSMutableArray <NSNumber *> *weights;
@property (nonatomic, strong) NSMutableArray <NSNumber *> *targets;
@property (nonatomic, assign) double learningRate;
@property (nonatomic, assign) NSInteger maxIteration;
@property (nonatomic, assign) double convergenceValue;
@property (nonatomic, assign) double sigma;
@property (nonatomic, assign) NSInteger batchSize; // 0: full batch learning, >= 1: mini batch if equals 1 that means normal SGD steps.

@property (nonatomic, assign) KRDeltaActiveFunctions activeFunction;
@property (nonatomic, strong) KRDeltaOptimization *optimization;

@property (nonatomic, copy) KRDeltaCompletion trainingCompletion;
@property (nonatomic, copy) KRDeltaIteration trainingIteraion;
@property (nonatomic, copy) KRDeltaBeforeUpdate beforeUpdate;

+ (instancetype)sharedDelta;
- (instancetype)init;

- (void)addPatterns:(NSArray *)_inputs target:(double)_targetValue;
- (void)setupWeights:(NSArray *)_initWeights;
- (void)setupRandomMin:(float)_min max:(float)_max;
- (void)randomWeights;
- (void)updateWeightsFromChanges:(NSArray *)_weightChanges;

- (void)training;
- (void)trainingWithCompletion:(KRDeltaCompletion)_completionBlock;
- (void)trainingWithBeforeUpdate:(KRDeltaBeforeUpdate)beforeUpdate completion:(KRDeltaCompletion)completion;
- (void)trainingWithIteration:(KRDeltaIteration)_iterationBlock completion:(KRDeltaCompletion)_completionBlock;
- (void)directOutputByPatterns:(NSArray *)_inputs completion:(KRDeltaDirectOutput)_completionBlock;

- (void)setTrainingCompletion:(KRDeltaCompletion)block;
- (void)setTrainingIteraion:(KRDeltaIteration)block;
- (void)setBeforeUpdate:(KRDeltaBeforeUpdate)block;

@end
