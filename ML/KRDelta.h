//
//  KRDelta.h
//  KRDelta
//
//  Created by Kalvar ( ilovekalvar@gmail.com ) on 13/6/13.
//  Copyright (c) 2013 - 2014年 Kuo-Ming Lin. All rights reserved.
//

#import "KRDeltaFetcher.h"

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

typedef void(^KRDeltaCompletion)(BOOL success, NSArray *weights, NSInteger totalIteration);
typedef void(^KRDeltaIteration)(NSInteger iteration, NSArray *weights);
typedef void(^KRDeltaDirectOutput)(NSArray *outputs);

@interface KRDelta : NSObject <NSCoding>

@property (nonatomic, strong) NSMutableArray *patterns;
@property (nonatomic, strong) NSMutableArray *weights;
@property (nonatomic, strong) NSMutableArray *targets;
@property (nonatomic, assign) double learningRate;
@property (nonatomic, assign) NSInteger maxIteration;
@property (nonatomic, assign) double convergenceValue;
@property (nonatomic, assign) double sigma;

@property (nonatomic, assign) KRDeltaActiveFunctions activeFunction;

@property (nonatomic, copy) KRDeltaCompletion trainingCompletion;
@property (nonatomic, copy) KRDeltaIteration trainingIteraion;

+ (instancetype)sharedDelta;
- (instancetype)init;

- (void)addPatterns:(NSArray *)_inputs target:(double)_targetValue;
- (void)setupWeights:(NSArray *)_initWeights;
- (void)setupRandomMin:(float)_min max:(float)_max;
- (void)randomWeights;

- (void)training;
- (void)trainingWithCompletion:(KRDeltaCompletion)_completionBlock;
- (void)trainingWithIteration:(KRDeltaIteration)_iterationBlock completion:(KRDeltaCompletion)_completionBlock;
- (void)directOutputByPatterns:(NSArray *)_inputs completion:(KRDeltaDirectOutput)_completionBlock;

- (void)setTrainingCompletion:(KRDeltaCompletion)_block;
- (void)setTrainingIteraion:(KRDeltaIteration)_block;

@end
