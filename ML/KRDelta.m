//
//  KRDelta.m
//  KRDelta
//
//  Created by Kalvar ( ilovekalvar@gmail.com ) on 13/6/13.
//  Copyright (c) 2013 - 2014年 Kuo-Ming Lin. All rights reserved.
//

#import "KRDelta.h"
#import "KRDeltaActivation.h"
#import "KRDeltaMath.h"

@interface KRDelta ()

@property (nonatomic, strong) KRDeltaActivation *activation;
@property (nonatomic, strong) KRDeltaMath *math;
@property (nonatomic, assign) NSInteger iteration;
@property (nonatomic, assign) double sumError; // Iteration errors
@property (nonatomic, strong) NSMutableArray *randomScopes;
@property (nonatomic, weak) NSCoder *coder;

@end

@implementation KRDelta (NSCoding)

- (void)encodeObject:(id)object forKey:(NSString *)key
{
    if( nil != object )
    {
        [self.coder encodeObject:object forKey:key];
    }
}

- (id)decodeForKey:(NSString *)key
{
    return [self.coder decodeObjectForKey:key];
}

@end

@implementation KRDelta (fixDelta)

- (double)_activateOutputValue:(double)_netOutput
{
    double _activatedValue = 0.0f;
    switch (self.activeFunction)
    {
        case KRDeltaActiveFunctionTanh:
            _activatedValue = [self.activation tanh:_netOutput slope:1.0f];
            break;
        case KRDeltaActiveFunctionSigmoid:
            _activatedValue = [self.activation sigmoid:_netOutput slope:1.0f];
            break;
        case KRDeltaActiveFunctionRBF:
            _activatedValue = [self.activation rbf:_netOutput sigma:self.sigma];
            break;
        default:
            _activatedValue = [self.activation sgn:_netOutput];
            break;
    }
    return _activatedValue;
}

- (double)_fOfNetWithInputs:(NSArray *)_inputs
{
    double _sum      = 0.0f;
    NSInteger _index = 0;
    for( NSNumber *_xValue in _inputs )
    {
        _sum += [_xValue floatValue] * [[self.weights objectAtIndex:_index] floatValue];
        ++_index;
    }
    return [self _activateOutputValue:_sum];
}

// f'(net) method in different active functions
- (double)_fDashOfNet:(double)_outputValue
{
    double _dashedValue = 0.0f;
    switch (self.activeFunction)
    {
        case KRDeltaActiveFunctionTanh:
            _dashedValue = [self.activation partialTanh:_outputValue slope:1.0f];
            break;
        case KRDeltaActiveFunctionSigmoid:
            _dashedValue = [self.activation partialSigmoid:_outputValue slope:1.0f];
            break;
        case KRDeltaActiveFunctionRBF:
            _dashedValue = [self.activation partialRBF:_outputValue sigma:self.sigma];
            break;
        case KRDeltaActiveFunctionSgn:
        default:
            _dashedValue = [self.activation partialSgn:_outputValue];
            break;
    }
    return _dashedValue;
}

- (double)_calculateIterationError
{
    // Delta method defined formula of cost function
    return (self.sumError / [self.patterns count]) * 0.5f;
}

- (void)_sumError:(double)_errorValue
{
    self.sumError += ( _errorValue * _errorValue );
}

- (void)_turningWeightsByInputs:(NSArray *)_inputs targetValue:(double)_targetValue
{
    NSArray *_weights      = self.weights;
    float _learningRate    = self.learningRate;
    double _netOutput      = [self _fOfNetWithInputs:_inputs];
    double _errorValue     = _targetValue - _netOutput;
    double _dashOutput     = [self _fDashOfNet:_netOutput];
    
    // new weights = learning rate * (target value - net output) * f'(net) * x1 + w1
    double _sigmaValue     = _learningRate * _errorValue * _dashOutput;
    NSArray *_deltaWeights = [self.math multiplyMatrix:_inputs byNumber:_sigmaValue];
    NSArray *_newWeights   = [self.math plusMatrix:_weights anotherMatrix:_deltaWeights];
    
    [self.weights removeAllObjects];
    [self.weights addObjectsFromArray:_newWeights];
    
    [self _sumError:_errorValue];
}

@end

@implementation KRDelta

+ (instancetype)sharedDelta
{
    static dispatch_once_t pred;
    static KRDelta *_object = nil;
    dispatch_once(&pred, ^{
        _object = [[KRDelta alloc] init];
    });
    return _object;
}

- (instancetype)init
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
        _randomScopes     = [NSMutableArray new];
        [self setupRandomMin:-0.5f max:0.5f];
        
        _maxIteration     = 1;
        _convergenceValue = 0.0f;
        _sigma            = 2.0f;
        
        _activeFunction   = KRDeltaActiveFunctionTanh;
        _activation       = [[KRDeltaActivation alloc] init];
        _math             = [[KRDeltaMath alloc] init];
    }
    return self;
}

#pragma --mark Public Methods
- (void)addPatterns:(NSArray *)_inputs target:(double)_targetValue
{
    [_patterns addObject:_inputs];
    [_targets addObject:[NSNumber numberWithDouble:_targetValue]];
}

- (void)setupWeights:(NSArray *)_initWeights
{
    if( [_weights count] > 0 )
    {
        [_weights removeAllObjects];
    }
    [_weights addObjectsFromArray:_initWeights];
}

- (void)setupRandomMin:(float)_min max:(float)_max
{
    [_randomScopes removeAllObjects];
    [_randomScopes addObject:[NSNumber numberWithFloat:_min]];
    [_randomScopes addObject:[NSNumber numberWithFloat:_max]];
}

- (void)randomWeights
{
    // Follows the inputs count to decide how many weights it needs.
    [_weights removeAllObjects];
    NSInteger _inputNetCount = [[_patterns firstObject] count];
    double _inputMax         = [[_randomScopes lastObject] floatValue] / _inputNetCount;
    double _inputMin         = [[_randomScopes firstObject] floatValue] / _inputNetCount;
    for( int i=0; i<_inputNetCount; i++ )
    {
        [_weights addObject:[NSNumber numberWithDouble:[_math randomMax:_inputMax min:_inputMin]]];
    }
}

- (void)training
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
    if( _iteration >= _maxIteration || [self _calculateIterationError] <= _convergenceValue )
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

- (void)trainingWithCompletion:(KRDeltaCompletion)_completionBlock
{
    _trainingCompletion = _completionBlock;
    [self training];
}

- (void)trainingWithIteration:(KRDeltaIteration)_iterationBlock completion:(KRDeltaCompletion)_completionBlock
{
    _trainingIteraion = _iterationBlock;
    [self trainingWithCompletion:_completionBlock];
}

- (void)directOutputByPatterns:(NSArray *)_inputs completion:(KRDeltaDirectOutput)_completionBlock
{
    double _netOutput = [self _fOfNetWithInputs:_inputs];
    if( nil != _completionBlock )
    {
        _completionBlock(@[[NSNumber numberWithDouble:_netOutput]]);
    }
}

#pragma --mark Block Setters
- (void)setTrainingCompletion:(KRDeltaCompletion)_block
{
    _trainingCompletion = _block;
}

- (void)setTrainingIteraion:(KRDeltaIteration)_block
{
    _trainingIteraion = _block;
}

#pragma --mark NSCoding
- (void)encodeWithCoder:(NSCoder *)aCoder
{
    self.coder = aCoder;
    [self encodeObject:@(_learningRate) forKey:@"learningRate"];
    [self encodeObject:_weights forKey:@"weights"];
    [self encodeObject:@(_sigma) forKey:@"sigma"];
    [self encodeObject:@(_activeFunction) forKey:@"activeFunction"];
}

- (instancetype)initWithCoder:(NSCoder *)aDecoder
{
    self = [self init];
    if(self)
    {
        self.coder      = aDecoder;
        _learningRate   = [[self decodeForKey:@"learningRate"] floatValue];
        _weights        = [self decodeForKey:@"weights"];
        _sigma          = [[self decodeForKey:@"sigma"] floatValue];
        _activeFunction = [[self decodeForKey:@"activeFunction"] integerValue];
    }
    return self;
}

@end


