//
//  KRDelta.h
//  KRDelta
//
//  Created by Kalvar ( ilovekalvar@gmail.com ) on 13/6/13.
//  Copyright (c) 2013 - 2014年 Kuo-Ming Lin. All rights reserved.
//

#import <Foundation/Foundation.h>

typedef enum KRDeltaActiveFunctions
{
    KRDeltaActiveFunctionBySgn  = 0,
    KRDeltaActiveFunctionByTanh,
    KRDeltaActiveFunctionBySigmoid
}KRDeltaActiveFunctions;

typedef void(^KRDeltaCompletion)(BOOL success, NSArray *weights, NSInteger totalIteration);
typedef void(^KRDeltaIteration)(NSInteger iteration, NSArray *weights);

@interface KRDelta : NSObject

@property (nonatomic, strong) NSMutableArray *patterns;
@property (nonatomic, strong) NSMutableArray *weights;
@property (nonatomic, strong) NSMutableArray *targets;
@property (nonatomic, assign) float learningRate;
@property (nonatomic, assign) NSInteger maxIteration;
@property (nonatomic, assign) float convergenceValue;

@property (nonatomic, assign) KRDeltaActiveFunctions activeFunction;

@property (nonatomic, copy) KRDeltaCompletion trainingCompletion;
@property (nonatomic, copy) KRDeltaIteration trainingIteraion;

+(instancetype)sharedDelta;
-(instancetype)init;

-(void)addPatterns:(NSArray *)_inputs target:(double)_targetValue;
-(void)setWeights:(NSArray *)_initWeights;
-(void)training;
-(void)trainingWithCompletion:(KRDeltaCompletion)_completion;

-(void)setTrainingCompletion:(KRDeltaCompletion)_block;
-(void)setTrainingIteraion:(KRDeltaIteration)_block;

@end
