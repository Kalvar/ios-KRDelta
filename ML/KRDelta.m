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

- (double)activate:(double)netInput
{
    double _activatedValue = netInput;
    switch (self.activeFunction)
    {
        case KRDeltaActivationTanh:
            _activatedValue = [self.activation tanh:netInput slope:2.0f];
            break;
        case KRDeltaActivationSigmoid:
            _activatedValue = [self.activation sigmoid:netInput slope:1.0f];
            break;
        case KRDeltaActivationRBF:
            _activatedValue = [self.activation rbf:netInput sigma:self.sigma];
            break;
        case KRDeltaActivationSGN:
            _activatedValue = [self.activation sgn:netInput];
            break;
        case KRDeltaActivationReLU:
            _activatedValue = [self.activation reLU:netInput];
            break;
        case KRDeltaActivationELU:
            _activatedValue = [self.activation eLU:netInput];
            break;
        case KRDeltaActivationLeakyReLU:
            _activatedValue = [self.activation leakyReLU:netInput];
            break;
        default:
            break;
    }
    return _activatedValue;
}

- (double)sumNetInput:(NSArray *)inputs
{
    double _sumSingal = 0.0f;
    NSInteger _index  = 0;
    for( NSNumber *_xValue in inputs )
    {
        _sumSingal += [_xValue floatValue] * [[self.weights objectAtIndex:_index] doubleValue];
        ++_index;
    }
    return _sumSingal;
}

// f'(net) method in different active functions
- (double)partialOfNet:(double)value
{
    KRDeltaPartial *partial = self.activation.partial;
    double partialValue = value;
    switch (self.activeFunction)
    {
        case KRDeltaActivationTanh:
            partialValue = [partial tanh:value slope:1.0f];
            break;
        case KRDeltaActivationSigmoid:
            partialValue = [partial sigmoid:value slope:1.0f];
            break;
        case KRDeltaActivationRBF:
            partialValue = [partial rbf:value sigma:self.sigma];
            break;
        case KRDeltaActivationSGN:
            partialValue = [partial sgn:value];
            break;
        case KRDeltaActivationReLU:
            partialValue = [partial reLU:value];
            break;
        case KRDeltaActivationELU:
            partialValue = [partial eLU:value];
            break;
        case KRDeltaActivationLeakyReLU:
            partialValue = [partial leakyReLU:value];
            break;
        default:
            break;
    }
    return partialValue;
}

- (double)calculateIterationError
{
    // Delta method defined formula of cost function
    return (self.sumError / [self.patterns count]) * 0.5f;
}

- (void)sumError:(double)_errorValue
{
    self.sumError += ( _errorValue * _errorValue );
}

// 判斷活化函式是否為線性
- (BOOL)isLinear
{
    BOOL isLinearFunction = YES;
    switch (self.activeFunction)
    {
        case KRDeltaActivationTanh:
        case KRDeltaActivationSigmoid:
        case KRDeltaActivationRBF:
            isLinearFunction = NO;
            break;
        default:
            break;
    }
    return isLinearFunction;
}

- (void)turningWeightsByInputs:(NSArray *)inputs targetValue:(double)targetValue
{
    double _netInput          = [self sumNetInput:inputs];
    double _netOutput         = [self activate:_netInput];
    double _errorValue        = targetValue - _netOutput;
    double _derivedActivation = [self partialOfNet:([self isLinear] ? _netInput : _netOutput)];
    
    // new weights = learning rate * (target value - net output) * f'(net) * x1 + w1
    double _deltaValue     = _errorValue * _derivedActivation;
    NSArray *_deltaWeights = [self.math multiplyMatrix:inputs byNumber:(self.learningRate * _deltaValue)];
    
    // Before updating the weights, the block can return NO to stop the update missions.
    if( nil != self.beforeUpdate )
    {
        BOOL stopUpdate = !self.beforeUpdate(self.iteration, _deltaWeights);
        if( stopUpdate )
        {
            return;
        }
    }
    
    [self updateWeightsFromChanges:_deltaWeights];
    [self sumError:_errorValue];
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
        
        _activeFunction   = KRDeltaActivationTanh;
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

- (void)updateWeightsFromChanges:(NSArray *)_weightChanges
{
    NSArray *_newWeights = [_math plusMatrix:_weights anotherMatrix:_weightChanges];
    [_weights removeAllObjects];
    [_weights addObjectsFromArray:_newWeights];
}

- (void)training
{
    ++_iteration;
    _sumError               = 0.0f;
    NSInteger _patternIndex = -1;
    for( NSArray *_inputs in _patterns )
    {
        ++_patternIndex;
        [self turningWeightsByInputs:_inputs targetValue:[[_targets objectAtIndex:_patternIndex] doubleValue]];
    }
    
    // One iteration done then doing next adjust conditions
    if( _iteration >= _maxIteration || [self calculateIterationError] <= _convergenceValue )
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

- (void)trainingWithBeforeUpdate:(KRDeltaBeforeUpdate)beforeUpdate completion:(KRDeltaCompletion)completion
{
    _beforeUpdate = beforeUpdate;
    [self trainingWithCompletion:completion];
}

- (void)trainingWithIteration:(KRDeltaIteration)_iterationBlock completion:(KRDeltaCompletion)_completionBlock
{
    _trainingIteraion = _iterationBlock;
    [self trainingWithCompletion:_completionBlock];
}

- (void)directOutputByPatterns:(NSArray *)_inputs completion:(KRDeltaDirectOutput)_completionBlock
{
    double _netOutput = [self activate:[self sumNetInput:_inputs]];
    if( nil != _completionBlock )
    {
        _completionBlock(@[[NSNumber numberWithDouble:_netOutput]]);
    }
}

#pragma --mark Block Setters
- (void)setTrainingCompletion:(KRDeltaCompletion)block
{
    _trainingCompletion = block;
}

- (void)setTrainingIteraion:(KRDeltaIteration)block
{
    _trainingIteraion = block;
}

- (void)setBeforeUpdate:(KRDeltaBeforeUpdate)block
{
    _beforeUpdate = block;
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


