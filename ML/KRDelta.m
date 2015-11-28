//
//  KRDelta.m
//  KRDelta
//
//  Created by Kalvar ( ilovekalvar@gmail.com ) on 13/6/13.
//  Copyright (c) 2013 - 2014年 Kuo-Ming Lin. All rights reserved.
//

#import "KRDelta.h"

@interface KRDelta ()

@property (nonatomic, assign) NSInteger iteration;
@property (nonatomic, assign) double sumError; // Iteration errors

@end

@implementation KRDelta (fixMatrix)

// ex : 0.5f * [1, 2]
-(NSArray *)_multiplyMatrix:(NSArray *)_matrix byNumber:(double)_number
{
    NSMutableArray *_array = [NSMutableArray new];
    for( NSNumber *_value in _matrix )
    {
        double _newValue = _number * [_value doubleValue];
        [_array addObject:[NSNumber numberWithDouble:_newValue]];
    }
    return _array;
}

// ex : [1, 2] + [3, 4]
-(NSArray *)_plusMatrix:(NSArray *)_matrix anotherMatrix:(NSArray *)_anotherMatrix
{
    NSMutableArray *_array = [NSMutableArray new];
    NSInteger _index       = 0;
    for( NSNumber *_value in _matrix )
    {
        double _newValue = [_value doubleValue] + [[_anotherMatrix objectAtIndex:_index] doubleValue];
        [_array addObject:[NSNumber numberWithDouble:_newValue]];
        ++_index;
    }
    return _array;
}

@end

@implementation KRDelta (fixDelta)

// Tanh() which scope is [-1.0, 1.0]
-(double)_fOfTanh:(float)_x
{
    return ( 2.0f / ( 1.0f + pow(M_E, (-1.0f * _x)) ) ) - 1.0f;
}

// Sigmoid() whici scope is [0.0, 1.0]
-(double)_fOfSigmoid:(float)_x
{
    return ( 1.0f / ( 1.0f + pow(M_E, (-1.0f * _x)) ) );
}

// SGN() which scope is (-1, 1) or (0, 1)
-(float)_fOfSgn:(double)_sgnValue
{
    return ( _sgnValue >= 0.0f ) ? 1.0f : -1.0f;
}

-(double)_fOfNetWithInputs:(NSArray *)_inputs
{
    double _sum      = 0.0f;
    NSInteger _index = 0;
    for( NSNumber *_xValue in _inputs )
    {
        _sum += [_xValue floatValue] * [[self.weights objectAtIndex:_index] floatValue];
        ++_index;
    }
    
    double _activatedValue = 0.0f;
    switch (self.activeFunction)
    {
        case KRDeltaActiveFunctionByTanh:
            _activatedValue = [self _fOfTanh:_sum];
            break;
        case KRDeltaActiveFunctionBySigmoid:
            _activatedValue = [self _fOfSigmoid:_sum];
            break;
        default:
            _activatedValue = [self _fOfSgn:_sum];
            break;
    }
    return _activatedValue;
}

// f'(net) method in different active functions
-(double)_fDashOfNet:(double)_outputValue
{
    double _dashedValue = 0.0f;
    switch (self.activeFunction)
    {
        case KRDeltaActiveFunctionByTanh:
            _dashedValue = ( 1.0f - ( _outputValue * _outputValue ) ) * 0.5f;
            break;
        case KRDeltaActiveFunctionBySigmoid:
            _dashedValue = _outputValue * ( 1.0f - _outputValue );
            break;
        default:
            _dashedValue = _outputValue;
            break;
    }
    return _dashedValue;
}

-(double)_mseBySumError:(double)_sumError
{
    return (_sumError / [self.patterns count]) * 0.5f;
}

-(void)_turningWeightsByInputs:(NSArray *)_inputs targetValue:(double)_targetValue
{
    NSArray *_weights           = self.weights;
    float _learningRate         = self.learningRate;
    double _netOutput           = [self _fOfNetWithInputs:_inputs];
    double _errorValue          = _targetValue - _netOutput;
    double _dashOutput          = [self _fDashOfNet:_netOutput];
    
    // new weights = learning rate * (target value - net output) * f'(net) * x1 + w1
    double _sigmaValue          = _learningRate * (_targetValue - _netOutput) * _dashOutput;
    NSArray *_deltaWeights      = [self _multiplyMatrix:_inputs byNumber:_sigmaValue];
    NSArray *_newWeights        = [self _plusMatrix:_weights anotherMatrix:_deltaWeights];
    
    [self.weights removeAllObjects];
    [self.weights addObjectsFromArray:_newWeights];
    
    // Sums the error values
    self.sumError              += ( _errorValue * _errorValue ) * 0.5f;
}

@end

@implementation KRDelta

+(instancetype)sharedDelta
{
    static dispatch_once_t pred;
    static KRDelta *_object = nil;
    dispatch_once(&pred, ^{
        _object = [[KRDelta alloc] init];
    });
    return _object;
}

-(instancetype)init
{
    self = [super init];
    if( self )
    {
        _learningRate     = 0.5f;
        _weights          = [NSMutableArray new];
        _patterns         = [NSMutableArray new];
        _targets          = [NSMutableArray new];
        
        _iteration        = 0;
        _sumError         = 0.0f;
        
        _maxIteration     = 1;
        _convergenceValue = 0.0f;
        
        _activeFunction   = KRDeltaActiveFunctionByTanh;
    }
    return self;
}

#pragma --mark Public Hebbian Algorithm
-(void)addPatterns:(NSArray *)_inputs target:(double)_targetValue
{
    [_patterns addObject:_inputs];
    [_targets addObject:[NSNumber numberWithDouble:_targetValue]];
}

-(void)setupWeights:(NSArray *)_initWeights
{
    if( [_weights count] > 0 )
    {
        [_weights removeAllObjects];
    }
    [_weights addObjectsFromArray:_initWeights];
}

-(void)training
{
    ++_iteration;
    _sumError               = 0.0f;
    NSInteger _patternIndex = -1;
    for( NSArray *_inputs in _patterns )
    {
        ++_patternIndex;
        [self _turningWeightsByInputs:_inputs targetValue:[[_targets objectAtIndex:_patternIndex] doubleValue]];
    }
    
    // One iteration done then doing next adjust conditions
    if( _iteration >= _maxIteration || [self _mseBySumError:_sumError] <= _convergenceValue )
    {
        if( nil != _trainingCompletion )
        {
            _trainingCompletion(YES, _weights, _iteration);
        }
    }
    else
    {
        if( nil != _trainingIteraion )
        {
            _trainingIteraion(_iteration, _weights);
        }
        [self training];
    }
}

-(void)trainingWithCompletion:(KRDeltaCompletion)_completionBlock
{
    _trainingCompletion = _completionBlock;
    [self training];
}

-(void)trainingWithIteration:(KRDeltaIteration)_iterationBlock completion:(KRDeltaCompletion)_completionBlock
{
    _trainingIteraion = _iterationBlock;
    [self trainingWithCompletion:_completionBlock];
}

-(void)directOutputByPatterns:(NSArray *)_inputs completion:(KRDeltaDirectOutput)_completionBlock
{
    double _netOutput = [self _fOfNetWithInputs:_inputs];
    if( nil != _completionBlock )
    {
        _completionBlock(@[[NSNumber numberWithDouble:_netOutput]]);
    }
}

#pragma --mark Block Setters
-(void)setTrainingCompletion:(KRDeltaCompletion)_block
{
    _trainingCompletion = _block;
}

-(void)setTrainingIteraion:(KRDeltaIteration)_block
{
    _trainingIteraion = _block;
}

@end


